import json
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from .services.gemini import _contextual_fallback_explanation, explain_issues_in_vietnamese


class ContextualExplanationTests(SimpleTestCase):
    def issue(self, text, original, replacement, category='GRAMMAR • Verb:Tense'):
        return {
            'id': 'issue-1',
            'category': category,
            'error_type': category,
            'original': original,
            'replacement': replacement,
            'offset': text.index(original),
        }

    def test_perfect_tense_explains_v3_instead_of_generic_verb_form(self):
        text = 'I have forgot my keys.'
        explanation = _contextual_fallback_explanation(
            self.issue(text, 'forgot', 'forgotten'), text,
        )
        self.assertIn('Sau “have”', explanation)
        self.assertIn('quá khứ phân từ (V3)', explanation)
        self.assertIn('“forgotten” là V3 của “forget”', explanation)
        self.assertIn('“forgot” là dạng quá khứ đơn (V2)', explanation)

    def test_past_simple_uses_the_actual_time_cue(self):
        text = 'She leave the office yesterday.'
        explanation = _contextual_fallback_explanation(
            self.issue(text, 'leave', 'left'), text,
        )
        self.assertIn('Dấu hiệu “yesterday”', explanation)
        self.assertIn('thì quá khứ đơn', explanation)
        self.assertIn('“left” là dạng quá khứ của “leave”', explanation)

    def test_regular_past_form_uses_the_actual_time_cue(self):
        text = 'They walk home last week.'
        explanation = _contextual_fallback_explanation(
            self.issue(text, 'walk', 'walked'), text,
        )
        self.assertIn('Dấu hiệu “last week”', explanation)
        self.assertIn('“walked” là dạng quá khứ của “walk”', explanation)

    def test_singular_subject_is_named_for_hasnt(self):
        text = "My brother haven't finished the report."
        explanation = _contextual_fallback_explanation(
            self.issue(
                text, "haven't", "hasn't",
                'GRAMMAR • Subject Verb Agreement',
            ),
            text,
        )
        self.assertIn('Chủ ngữ “My brother” là ngôi thứ ba số ít', explanation)
        self.assertIn('“has not/hasn’t”', explanation)
        self.assertIn('“I”, “you”, “we”', explanation)

    def test_plural_subject_is_named_for_havent(self):
        text = "They hasn't finished the report."
        explanation = _contextual_fallback_explanation(
            self.issue(
                text, "hasn't", "haven't",
                'GRAMMAR • Subject Verb Agreement',
            ),
            text,
        )
        self.assertIn('Chủ ngữ “They” là số nhiều', explanation)
        self.assertIn('“have not/haven’t”', explanation)

    def test_plural_subject_is_named_for_have(self):
        text = 'They has finished the report.'
        explanation = _contextual_fallback_explanation(
            self.issue(
                text, 'has', 'have',
                'GRAMMAR • Subject Verb Agreement',
            ),
            text,
        )
        self.assertIn('Chủ ngữ “They” là số nhiều', explanation)
        self.assertIn('dùng trợ động từ “have”, không dùng “has”', explanation)

    def test_have_rule_covers_personal_and_plural_subjects(self):
        for subject in ('I', 'You', 'We', 'They', 'My friends'):
            with self.subTest(subject=subject):
                text = f'{subject} has finished the report.'
                explanation = _contextual_fallback_explanation(
                    self.issue(
                        text, 'has', 'have',
                        'GRAMMAR • Subject Verb Agreement',
                    ),
                    text,
                )
                self.assertIn(f'“{subject}”', explanation)
                self.assertIn('dùng trợ động từ “have”, không dùng “has”', explanation)

    def test_singular_subject_is_named_for_has(self):
        text = 'My brother have finished the report.'
        explanation = _contextual_fallback_explanation(
            self.issue(
                text, 'have', 'has',
                'GRAMMAR • Subject Verb Agreement',
            ),
            text,
        )
        self.assertIn('Chủ ngữ “My brother” là ngôi thứ ba số ít', explanation)
        self.assertIn('dùng trợ động từ “has”, không dùng “have”', explanation)

    def test_regular_participle_explains_why_v3_is_required(self):
        text = 'They have finish the report.'
        explanation = _contextual_fallback_explanation(
            self.issue(text, 'finish', 'finished'), text,
        )
        self.assertIn('Sau “have”', explanation)
        self.assertIn('phải ở dạng quá khứ phân từ (V3)', explanation)
        self.assertIn('“finished” là V3 của “finish”', explanation)
        self.assertNotIn('V2) hoặc', explanation)

    def test_for_is_explained_with_the_actual_duration(self):
        for duration in ('two years', 'three months', 'a long time'):
            with self.subTest(duration=duration):
                text = f'I have lived here since {duration}.'
                explanation = _contextual_fallback_explanation(
                    self.issue(text, 'since', 'for', 'GRAMMAR • Prepositions'), text,
                )
                self.assertIn(f'“{duration}” diễn tả một khoảng thời gian', explanation)
                self.assertIn('“Since” dùng với một mốc bắt đầu cụ thể', explanation)

    def test_since_is_explained_with_the_actual_start_point(self):
        text = 'I have lived here for 2024.'
        explanation = _contextual_fallback_explanation(
            self.issue(text, 'for', 'since', 'GRAMMAR • Prepositions'), text,
        )
        self.assertIn('“2024” là mốc bắt đầu', explanation)
        self.assertIn('“For” dùng với một khoảng thời gian', explanation)

    def test_missing_time_cue_does_not_invent_one(self):
        text = 'She leave the office.'
        explanation = _contextual_fallback_explanation(
            self.issue(text, 'leave', 'left'), text,
        )
        self.assertNotIn('Dấu hiệu', explanation)
        self.assertNotIn('yesterday', explanation)
        self.assertIn('“left” là dạng quá khứ đơn (V2)', explanation)

    @patch('api.services.gemini.get_gemini_client')
    def test_generic_gemini_explanation_does_not_replace_specific_fallback(self, client_factory):
        text = "My brother haven't finished the report."
        issue = self.issue(
            text, "haven't", "hasn't",
            'GRAMMAR • Subject Verb Agreement',
        )
        client = MagicMock()
        client.models.generate_content.return_value.text = json.dumps([{
            'id': 'issue-1',
            'explanation': 'Thay “haven’t” bằng “hasn’t”. Động từ phải hòa hợp với chủ ngữ về số.',
            'concept_code': 'subject_verb_agreement',
        }], ensure_ascii=False)
        client_factory.return_value = client

        issues, explained = explain_issues_in_vietnamese([issue], text)

        self.assertEqual(explained, 0)
        self.assertIn('Chủ ngữ “My brother” là ngôi thứ ba số ít', issues[0]['explanation'])
        payload = json.loads(client.models.generate_content.call_args.kwargs['contents'])
        self.assertEqual(payload[0]['grammar_facts']['subject'], 'My brother')
        self.assertEqual(payload[0]['grammar_facts']['subject_number'], 'ngôi thứ ba số ít')

    @patch('api.services.gemini.get_gemini_client')
    def test_ambiguous_v2_or_v3_gemini_explanation_is_rejected(self, client_factory):
        text = 'They have finish the report.'
        issue = self.issue(text, 'finish', 'finished')
        client = MagicMock()
        client.models.generate_content.return_value.text = json.dumps([{
            'id': 'issue-1',
            'explanation': 'Thay “finish” bằng “finished”. “Finished” là dạng V2 hoặc V3 của “finish”.',
            'concept_code': 'present_perfect',
        }], ensure_ascii=False)
        client_factory.return_value = client

        issues, explained = explain_issues_in_vietnamese([issue], text)

        self.assertEqual(explained, 0)
        self.assertIn('Sau “have”', issues[0]['explanation'])
        self.assertIn('“finished” là V3 của “finish”', issues[0]['explanation'])

    @patch('api.services.gemini.get_gemini_client')
    def test_generic_since_for_gemini_explanation_is_rejected(self, client_factory):
        text = 'I have lived here since three months.'
        issue = self.issue(text, 'since', 'for', 'GRAMMAR • Prepositions')
        client = MagicMock()
        client.models.generate_content.return_value.text = json.dumps([{
            'id': 'issue-1',
            'explanation': 'Thay “since” bằng “for”. Cách sửa này giúp cấu trúc câu đúng ngữ pháp hơn.',
            'concept_code': 'prepositions',
        }], ensure_ascii=False)
        client_factory.return_value = client

        issues, explained = explain_issues_in_vietnamese([issue], text)

        self.assertEqual(explained, 0)
        self.assertIn('“three months” diễn tả một khoảng thời gian', issues[0]['explanation'])
        payload = json.loads(client.models.generate_content.call_args.kwargs['contents'])
        self.assertEqual(payload[0]['grammar_facts']['temporal_expression'], 'three months')
        self.assertEqual(payload[0]['grammar_facts']['temporal_kind'], 'duration')

