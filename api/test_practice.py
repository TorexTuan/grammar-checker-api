import json
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.db.models import Count
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient, APITestCase

from .models import PracticeExercise, PracticeMistake, PracticeProfile
from .services.practice import classify_concept, enrich_practice_issues, record_detected_mistakes
from .services.practice import (
    VERB_SEEDS,
    _personalized_seed_item,
    generate_with_gemini,
    prompt_template_signature,
)


User = get_user_model()


class PracticeApiTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user('practice@example.com', 'practice@example.com', 'StrongPass!2026')
        self.other = User.objects.create_user('other@example.com', 'other@example.com', 'StrongPass!2026')
        self.client.force_authenticate(self.user)

    def mistake_payload(self):
        return {
            'category': 'Grammar • Subject Verb Agreement',
            'sentence_context': 'She go to school every day.',
            'original': 'go',
            'correction': 'goes',
            'explanation': 'Chủ ngữ số ít cần động từ thêm s.',
        }

    def test_records_and_deduplicates_private_mistakes(self):
        first = self.client.post('/api/practice/mistakes/', self.mistake_payload(), format='json')
        second = self.client.post('/api/practice/mistakes/', self.mistake_payload(), format='json')
        self.assertEqual(first.status_code, status.HTTP_201_CREATED)
        self.assertEqual(second.status_code, status.HTTP_200_OK)
        mistake = PracticeMistake.objects.get(owner=self.user)
        self.assertEqual(mistake.category, 'subject_verb_agreement')
        self.assertEqual(mistake.occurrence_count, 2)
        self.assertFalse(PracticeMistake.objects.filter(owner=self.other).exists())

    def test_detected_past_simple_issue_is_recorded_without_accept(self):
        issues = [{
            'category': 'GRAMMAR • Verb:Tense',
            'error_type': 'Verb:Tense',
            'original': 'go',
            'replacement': 'went',
            'offset': 5,
            'explanation': 'Use the past simple.',
        }]
        enrich_practice_issues(issues, 'She go to school yesterday.')
        self.assertEqual(issues[0]['concept_code'], 'past_simple')
        self.assertEqual(record_detected_mistakes(self.user, issues), 1)
        mistake = PracticeMistake.objects.get(owner=self.user)
        self.assertEqual(mistake.source, 'editor_live')
        self.assertEqual(mistake.concept_code, 'past_simple')

    @patch('api.views.explain_issues_in_vietnamese', side_effect=lambda issues, _text: (issues, 0))
    @patch('api.views.check_text_with_sapling')
    def test_check_endpoint_auto_captures_grammar_but_not_spelling(self, sapling, _explain):
        sapling.return_value = [
            {'id': 'g1', 'category': 'GRAMMAR • Verb:Tense', 'original': 'go', 'replacement': 'went', 'offset': 4},
            {'id': 's1', 'category': 'SPELLING • Spelling', 'original': 'teh', 'replacement': 'the', 'offset': 0},
        ]
        response = self.client.post('/api/check/', {
            'text': 'She go yesterday.',
            'capture_practice': True,
            'excluded_practice_fingerprints': [],
        }, format='json')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['sources']['practice_recorded'], 1)
        self.assertEqual(PracticeMistake.objects.filter(owner=self.user, source='editor_live').count(), 1)

    @patch('api.practice_views.generate_with_gemini', return_value=([], None))
    def test_personalized_session_uses_live_editor_mistakes(self, _generate):
        mistake = PracticeMistake.objects.create(
            owner=self.user,
            category='verb_tense',
            concept_code='past_simple',
            concept_label='Past simple',
            source='editor_live',
            sentence_context='She go yesterday.',
            original='go',
            correction='went',
            fingerprint='live-mistake',
        )
        response = self.client.post('/api/practice/sessions/', {'mode': 'personalized'}, format='json')
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data['mode'], 'personalized')
        self.assertEqual(response.data['total'], 10)
        self.assertEqual(len(response.data['items']), 10)
        difficulties = [item['difficulty'] for item in response.data['items']]
        exercise_types = [item['exercise_type'] for item in response.data['items']]
        self.assertEqual(difficulties.count('easy'), 3)
        self.assertEqual(difficulties.count('medium'), 4)
        self.assertEqual(difficulties.count('hard'), 3)
        self.assertEqual(exercise_types.count('multiple_choice'), 6)
        self.assertEqual(exercise_types.count('fill_blank'), 4)
        self.assertEqual(difficulties, sorted(difficulties, key=('easy', 'medium', 'hard').index))
        self.assertTrue(all(exercise.mistake_id == mistake.id for exercise in PracticeExercise.objects.filter(session_items__session_id=response.data['id'])))

    @patch('api.practice_views.generate_with_gemini', return_value=([], None))
    def test_personalized_groups_duplicate_mistakes_by_concept(self, _generate):
        for index in range(2):
            PracticeMistake.objects.create(
                owner=self.user,
                category='verb_tense',
                concept_code='past_simple',
                concept_label='Past simple',
                source='editor_live',
                sentence_context=f'Private source sentence {index}.',
                original='go',
                correction='went',
                occurrence_count=index + 1,
                fingerprint=f'same-concept-{index}',
            )
        response = self.client.post('/api/practice/sessions/', {'mode': 'personalized'}, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED, response.data)
        self.assertEqual(response.data['focus_concepts'], [{'code': 'past_simple', 'label': 'Past simple'}])
        self.assertEqual(len(response.data['items']), 10)
        self.assertFalse(any('Private source sentence' in item['prompt'] for item in response.data['items']))

    def test_personalized_generation_key_ignores_decorative_prefix(self):
        plain = _personalized_seed_item('past_simple', 'multiple_choice', 'medium', 0)
        prefixed = _personalized_seed_item('past_simple', 'multiple_choice', 'medium', len(VERB_SEEDS))
        self.assertNotEqual(plain['prompt'], prefixed['prompt'])
        self.assertEqual(plain['generation_key'], prefixed['generation_key'])

    @patch('api.practice_views.generate_with_gemini', return_value=([], None))
    def test_single_concept_has_no_repeat_in_rolling_thirty_questions(self, _generate):
        PracticeMistake.objects.create(
            owner=self.user,
            category='verb_tense',
            concept_code='past_simple',
            concept_label='Past simple',
            source='editor_live',
            sentence_context='A private sentence that must not be copied.',
            original='leave',
            correction='left',
            fingerprint='single-concept-history',
        )
        recent_prompts = []
        recent_keys = []
        for _ in range(6):
            response = self.client.post('/api/practice/sessions/', {'mode': 'personalized'}, format='json')
            self.assertEqual(response.status_code, status.HTTP_201_CREATED, response.data)
            ids = [item['exercise_id'] for item in response.data['items']]
            exercises = list(PracticeExercise.objects.filter(id__in=ids))
            prompts = [exercise.prompt for exercise in exercises]
            keys = [exercise.generation_key for exercise in exercises]
            self.assertEqual(len(prompts), 10)
            self.assertEqual(len(set(keys)), 10)
            self.assertFalse(set(prompts) & set(recent_prompts[-30:]))
            self.assertFalse(set(keys) & set(recent_keys[-30:]))
            recent_prompts.extend(prompts)
            recent_keys.extend(keys)

    @patch('api.practice_views.generate_with_gemini', return_value=([], None))
    def test_personalized_uses_matching_unused_cache_item(self, _generate):
        mistake = PracticeMistake.objects.create(
            owner=self.user,
            category='verb_tense',
            concept_code='past_simple',
            concept_label='Past simple',
            source='editor_live',
            sentence_context='She leave yesterday.',
            original='leave',
            correction='left',
            fingerprint='cache-mistake',
        )
        cached = PracticeExercise.objects.create(
            owner=self.user,
            mistake=mistake,
            exercise_type='multiple_choice',
            mode='personalized',
            difficulty='easy',
            category='verb_tense',
            prompt='Yesterday, the guide ___ the station before noon.',
            hint='',
            options=['leave', 'left', 'leaves', 'leaving'],
            accepted_answers=['left'],
            explanation='“Yesterday” requires the past simple form “left”.',
            source='gemini_primary',
            fingerprint='cached-content',
            generation_key='cached-structure',
        )
        response = self.client.post('/api/practice/sessions/', {'mode': 'personalized'}, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED, response.data)
        self.assertIn(str(cached.id), [item['exercise_id'] for item in response.data['items']])
        cached.refresh_from_db()
        self.assertEqual(cached.used_count, 1)

    @patch('api.practice_views.generate_with_gemini', return_value=([], None))
    def test_personalized_fallback_stays_complete_after_recent_templates_are_used(self, _generate):
        mistakes = (
            ('subject_verb_agreement', 'subject_verb_agreement', 'have', 'has'),
            ('other_grammar', 'verb_tense', 'bring', 'brought'),
            ('verb_tense', 'past_simple', 'have', 'had'),
        )
        for index, (category, concept, original, correction) in enumerate(mistakes):
            PracticeMistake.objects.create(
                owner=self.user,
                category=category,
                concept_code=concept,
                concept_label=concept.replace('_', ' ').title(),
                source='editor_live',
                sentence_context='A neutral editor sentence.',
                original=original,
                correction=correction,
                occurrence_count=3 - index,
                fingerprint=f'live-history-{index}',
            )

        recent_prompts = []
        recent_generation_keys = []
        for session_index in range(7):
            response = self.client.post('/api/practice/sessions/', {'mode': 'personalized'}, format='json')
            self.assertEqual(response.status_code, status.HTTP_201_CREATED, response.data)
            prompts = [item['prompt'] for item in response.data['items']]
            exercise_ids = [item['exercise_id'] for item in response.data['items']]
            generation_keys = list(PracticeExercise.objects.filter(id__in=exercise_ids).values_list('generation_key', flat=True))
            self.assertEqual(len(prompts), 10)
            self.assertEqual(len(set(generation_keys)), 10)
            self.assertTrue(all(generation_keys))
            self.assertFalse(set(prompts) & set(recent_prompts[-30:]))
            self.assertFalse(set(generation_keys) & set(recent_generation_keys[-30:]))
            if session_index == 0:
                exercises = PracticeExercise.objects.filter(id__in=exercise_ids)
                concept_counts = sorted(
                    exercises.values('mistake__concept_code').annotate(total=Count('id')).values_list('total', flat=True),
                    reverse=True,
                )
                self.assertEqual(concept_counts, [4, 3, 3])
            recent_prompts.extend(prompts)
            recent_generation_keys.extend(generation_keys)

        verb_exercises = PracticeExercise.objects.filter(mistake__concept_code='verb_tense')
        self.assertTrue(verb_exercises.exists())
        self.assertTrue(all(exercise.category == 'verb_tense' for exercise in verb_exercises))

    @patch('api.practice_views.generate_with_gemini', return_value=([], None))
    def test_general_level_increases_after_ten_correct_answers(self, generate):
        seen_prompts = set()
        for _ in range(1):
            session = self.client.post('/api/practice/sessions/', {'mode': 'general'}, format='json')
            self.assertEqual(session.status_code, 201)
            for item in session.data['items']:
                exercise = PracticeExercise.objects.get(id=item['exercise_id'])
                self.assertNotIn(exercise.prompt, seen_prompts)
                seen_prompts.add(exercise.prompt)
                answer = self.client.post(
                    f"/api/practice/sessions/{session.data['id']}/answer/",
                    {'exercise_id': item['exercise_id'], 'answer': exercise.accepted_answers[0]},
                    format='json',
                )
                self.assertTrue(answer.data['is_correct'])
        profile = PracticeProfile.objects.get(owner=self.user)
        self.assertEqual(profile.general_level, 'medium')
        self.assertTrue(all(call.args[0] == [] for call in generate.call_args_list))

    @patch('api.practice_views.generate_with_gemini', return_value=([], None))
    def test_general_fallback_does_not_repeat_templates_in_consecutive_sessions(self, _generate):
        recent_prompts = []
        previous_signatures = set()
        for _ in range(6):
            response = self.client.post('/api/practice/sessions/', {'mode': 'general'}, format='json')
            self.assertEqual(response.status_code, status.HTTP_201_CREATED, response.data)
            prompts = [item['prompt'] for item in response.data['items']]
            signatures = {prompt_template_signature(prompt) for prompt in prompts}
            self.assertEqual(len(prompts), 10)
            self.assertEqual(len(signatures), 10)
            self.assertFalse(set(prompts) & set(recent_prompts[-30:]))
            self.assertFalse(signatures & previous_signatures)
            previous_signatures = signatures
            recent_prompts.extend(prompts)

    @patch('api.practice_views.generate_with_gemini', return_value=([], None))
    def test_fallback_session_hides_answers_and_grades_idempotently(self, _generate):
        created = self.client.post('/api/practice/sessions/', {'topic': 'verb_tense'}, format='json')
        self.assertEqual(created.status_code, status.HTTP_201_CREATED)
        self.assertEqual(created.data['total'], 10)
        self.assertEqual(len(created.data['items']), 10)
        self.assertEqual(created.data['source_summary'], {'template': 10})
        exercise_types = [item['exercise_type'] for item in created.data['items']]
        self.assertEqual(exercise_types.count('multiple_choice'), 6)
        self.assertEqual(exercise_types.count('fill_blank'), 4)
        for item in created.data['items']:
            self.assertNotIn('correct_answer', item)
            self.assertNotIn('accepted_answers', item)
            self.assertNotIn('explanation', item)
            if item['exercise_type'] == 'fill_blank':
                self.assertTrue(item['hint'])

        item = created.data['items'][0]
        exercise = PracticeExercise.objects.get(id=item['exercise_id'])
        response = self.client.post(
            f"/api/practice/sessions/{created.data['id']}/answer/",
            {'exercise_id': item['exercise_id'], 'answer': exercise.accepted_answers[0].upper()},
            format='json',
        )
        repeated = self.client.post(
            f"/api/practice/sessions/{created.data['id']}/answer/",
            {'exercise_id': item['exercise_id'], 'answer': 'a different answer'},
            format='json',
        )
        self.assertTrue(response.data['is_correct'])
        self.assertTrue(repeated.data['is_correct'])
        self.assertEqual(repeated.data['user_answer'], exercise.accepted_answers[0].upper())

    @patch('api.practice_views.generate_with_gemini', return_value=([], None))
    def test_session_and_progress_are_isolated_by_owner(self, _generate):
        created = self.client.post('/api/practice/sessions/', {'topic': 'articles'}, format='json')
        self.client.force_authenticate(self.other)
        hidden = self.client.get(f"/api/practice/sessions/{created.data['id']}/")
        self.assertEqual(hidden.status_code, status.HTTP_404_NOT_FOUND)
        progress = self.client.get('/api/practice/progress/')
        self.assertEqual(progress.data['answered'], 0)

    def test_practice_requires_authentication(self):
        self.client.force_authenticate(user=None)
        self.assertIn(self.client.get('/api/practice/overview/').status_code, (401, 403))
        self.assertIn(self.client.post('/api/practice/sessions/', {}, format='json').status_code, (401, 403))

    def test_practice_write_requires_csrf_for_session_auth(self):
        client = APIClient(enforce_csrf_checks=True)
        client.force_login(self.user)
        denied = client.post('/api/practice/mistakes/', self.mistake_payload(), format='json')
        self.assertEqual(denied.status_code, status.HTTP_403_FORBIDDEN)
        csrf = client.get(reverse('auth-csrf')).json()['csrfToken']
        client.credentials(HTTP_X_CSRFTOKEN=csrf)
        allowed = client.post('/api/practice/mistakes/', self.mistake_payload(), format='json')
        self.assertEqual(allowed.status_code, status.HTTP_201_CREATED)


class PracticeGeminiFallbackTests(APITestCase):
    def setUp(self):
        cache.clear()
        self.mistake = MagicMock(
            category='verb_tense',
            sentence_context='She go yesterday.',
            original='go',
            correction='went',
            explanation='Past tense.',
        )

    def valid_payload(self):
        items = []
        for index in range(6):
            items.append({
                'exercise_type': 'multiple_choice',
                'category': 'verb_tense',
                'prompt': f'Choose answer {index}.',
                'options': ['went', 'go', 'goes', 'going'],
                'accepted_answers': ['went'],
                'hint': '',
                'explanation': 'Dùng quá khứ đơn.',
            })
        for index in range(4):
            items.append({
                'exercise_type': 'fill_blank',
                'category': 'verb_tense',
                'prompt': f'They ___ yesterday {index}.',
                'options': [],
                'accepted_answers': ['went'],
                'hint': 'go',
                'explanation': 'Dùng quá khứ đơn.',
            })
        return items

    @patch('api.services.practice._practice_client')
    def test_primary_timeout_uses_secondary_model(self, client_factory):
        primary = MagicMock()
        primary.models.generate_content.side_effect = TimeoutError('timed out')
        secondary = MagicMock()
        secondary.models.generate_content.return_value.text = json.dumps(self.valid_payload())
        client_factory.side_effect = [primary, secondary]
        exercises, source = generate_with_gemini([self.mistake], 'verb_tense')
        self.assertEqual(source, 'gemini_secondary')
        self.assertEqual(len(exercises), 10)

    @patch('api.services.practice._practice_client')
    def test_invalid_responses_open_circuit_and_return_empty(self, client_factory):
        broken = MagicMock()
        broken.models.generate_content.return_value.text = '{}'
        client_factory.return_value = broken
        generate_with_gemini([self.mistake], 'verb_tense')
        generate_with_gemini([self.mistake], 'verb_tense')
        before = client_factory.call_count
        exercises, source = generate_with_gemini([self.mistake], 'verb_tense')
        self.assertEqual((exercises, source), ([], None))
        self.assertEqual(client_factory.call_count, before)
