import random

from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import Count, F, Q, Sum
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import (
    PracticeExercise,
    PracticeMistake,
    PracticeProfile,
    PracticeSession,
    PracticeSessionItem,
)
from .services.practice import (
    SESSION_SIZE,
    TOPICS,
    TYPE_TARGETS,
    CONCEPT_LABELS,
    CONCEPT_TOPICS,
    fingerprint,
    general_fallback_exercises,
    generate_with_gemini,
    is_recent_prompt,
    normalize_answer,
    normalize_topic,
    personalized_fallback_exercises,
    prompt_template_signature,
)


def _limited_string(data, field, maximum, required=False):
    value = data.get(field, '')
    if not isinstance(value, str):
        raise ValueError(f'{field} must be a string.')
    value = value.strip()
    if required and not value:
        raise ValueError(f'{field} is required.')
    if len(value) > maximum:
        raise ValueError(f'{field} must not exceed {maximum} characters.')
    return value


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def record_mistake(request):
    try:
        sentence = _limited_string(request.data, 'sentence_context', 500, required=True)
        original = _limited_string(request.data, 'original', 300)
        correction = _limited_string(request.data, 'correction', 300)
        explanation = _limited_string(request.data, 'explanation', 2000)
        if not original and not correction:
            raise ValueError('original or correction is required.')
        category = normalize_topic(_limited_string(request.data, 'category', 160))
    except ValueError as exc:
        return Response({'error': str(exc)}, status=status.HTTP_400_BAD_REQUEST)

    key = fingerprint(category, original, correction, sentence)
    with transaction.atomic():
        mistake = PracticeMistake.objects.select_for_update().filter(
            owner=request.user,
            fingerprint=key,
        ).first()
        created = mistake is None
        if created:
            mistake = PracticeMistake.objects.create(
                owner=request.user,
                category=category,
                concept_code=category,
                concept_label=CONCEPT_LABELS.get(category, TOPICS.get(category, 'Other grammar')),
                source='editor_live',
                sentence_context=sentence,
                original=original,
                correction=correction,
                explanation=explanation,
                fingerprint=key,
            )
        else:
            mistake.explanation = explanation or mistake.explanation
            mistake.occurrence_count = F('occurrence_count') + 1
            mistake.save(update_fields=['explanation', 'occurrence_count', 'last_seen_at'])
            mistake.refresh_from_db()

    return Response({
        'id': str(mistake.id),
        'category': mistake.category,
        'occurrence_count': mistake.occurrence_count,
        'created': created,
    }, status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)


def _topic_stats(user):
    mistake_rows = PracticeMistake.objects.filter(owner=user).values('category').annotate(
        mistakes=Sum('occurrence_count'),
    )
    mistake_counts = {row['category']: row['mistakes'] or 0 for row in mistake_rows}
    answer_rows = PracticeSessionItem.objects.filter(
        session__owner=user,
        is_correct__isnull=False,
    ).values('exercise__category').annotate(
        answered=Count('id'),
        correct=Count('id', filter=Q(is_correct=True)),
    )
    answer_counts = {
        row['exercise__category']: (row['answered'], row['correct'])
        for row in answer_rows
    }
    return [
        {
            'code': code,
            'label': label,
            'mistakes': mistake_counts.get(code, 0),
            'answered': answer_counts.get(code, (0, 0))[0],
            'correct': answer_counts.get(code, (0, 0))[1],
            'accuracy': round(
                answer_counts.get(code, (0, 0))[1] * 100 / answer_counts.get(code, (0, 0))[0]
            ) if answer_counts.get(code, (0, 0))[0] else None,
        }
        for code, label in TOPICS.items()
    ]


def _recommended_topic(user):
    stats = _topic_stats(user)
    attempted = [item for item in stats if item['answered']]
    if attempted:
        return min(attempted, key=lambda item: (item['accuracy'], -item['mistakes']))['code']
    with_mistakes = [item for item in stats if item['mistakes']]
    if with_mistakes:
        return max(with_mistakes, key=lambda item: item['mistakes'])['code']
    return 'other_grammar'


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def practice_overview(request):
    mistakes = PracticeMistake.objects.filter(owner=request.user, source='editor_live')
    profile, _ = PracticeProfile.objects.get_or_create(owner=request.user)
    concept_rows = list(mistakes.values('concept_code', 'concept_label').annotate(
        mistakes=Sum('occurrence_count'),
    ).order_by('-mistakes', 'concept_label')[:6])
    return Response({
        'personalized': {
            'available': mistakes.exists(),
            'mistake_count': mistakes.aggregate(total=Sum('occurrence_count'))['total'] or 0,
            'concept_count': mistakes.values('concept_code').distinct().count(),
            'concepts': concept_rows,
        },
        'general': {
            'level': profile.general_level,
            'level_answered': profile.level_answered,
            'level_correct': profile.level_correct,
        },
    })


def _public_item(item):
    exercise = item.exercise
    result = {
        'item_id': str(item.id),
        'exercise_id': str(exercise.id),
        'order': item.order,
        'exercise_type': exercise.exercise_type,
        'category': exercise.category,
        'category_label': exercise.mistake.concept_label if exercise.mistake_id else TOPICS.get(exercise.category, TOPICS['other_grammar']),
        'difficulty': exercise.difficulty,
        'prompt': exercise.prompt,
        'hint': exercise.hint if exercise.exercise_type == PracticeExercise.TYPE_FILL_BLANK else '',
        'options': exercise.options if exercise.exercise_type == PracticeExercise.TYPE_MULTIPLE_CHOICE else [],
        'answered': item.is_correct is not None,
        'user_answer': item.user_answer if item.is_correct is not None else '',
    }
    if item.is_correct is not None:
        result.update({
            'is_correct': item.is_correct,
            'correct_answer': exercise.accepted_answers[0],
            'accepted_answers': exercise.accepted_answers,
            'explanation': exercise.explanation,
        })
    return result


def _session_payload(session):
    items = list(session.items.select_related('exercise', 'exercise__mistake').all())
    answered = sum(item.is_correct is not None for item in items)
    correct = sum(item.is_correct is True for item in items)
    return {
        'id': str(session.id),
        'topic': session.topic,
        'mode': session.mode,
        'topic_label': 'Practice your mistakes' if session.mode == 'personalized' else 'General grammar practice',
        'difficulty': session.difficulty,
        'focus_concepts': session.focus_concepts,
        'total': session.total_questions,
        'answered': answered,
        'correct': correct,
        'completed': session.completed_at is not None,
        'source_summary': session.source_summary,
        'items': [_public_item(item) for item in items],
    }


def _pick_by_targets(exercises, targets):
    chosen = []
    for exercise in exercises:
        if targets.get(exercise.exercise_type, 0) > 0:
            chosen.append(exercise)
            targets[exercise.exercise_type] -= 1
    return chosen


def _save_exercise(user, data, source, mistake=None, mode='personalized', difficulty='medium'):
    item_difficulty = data.get('difficulty') or difficulty
    generation_key = data.get('generation_key') or fingerprint(
        mode,
        mistake.concept_code if mistake else data.get('category', ''),
        item_difficulty,
        data['exercise_type'],
        prompt_template_signature(data['prompt']),
        data['accepted_answers'][0],
    )
    key = fingerprint(mode, item_difficulty, mistake.id if mistake else '', data['exercise_type'], data['prompt'], data['accepted_answers'][0])
    existing = PracticeExercise.objects.filter(owner=user, fingerprint=key).first()
    if existing:
        if not existing.generation_key:
            existing.generation_key = generation_key
            existing.save(update_fields=['generation_key'])
        return existing
    return PracticeExercise.objects.create(
        owner=user,
        mistake=mistake,
        exercise_type=data['exercise_type'],
        mode=mode,
        difficulty=item_difficulty,
        category=data['category'],
        prompt=data['prompt'],
        hint=data.get('hint', ''),
        options=data['options'],
        accepted_answers=data['accepted_answers'],
        explanation=data['explanation'],
        source=source,
        fingerprint=key,
        generation_key=generation_key,
    )


PERSONALIZED_SLOT_PLANS = {
    1: [[
        {'exercise_type': 'multiple_choice', 'difficulty': 'easy'},
        {'exercise_type': 'fill_blank', 'difficulty': 'easy'},
        {'exercise_type': 'multiple_choice', 'difficulty': 'easy'},
        {'exercise_type': 'multiple_choice', 'difficulty': 'medium'},
        {'exercise_type': 'fill_blank', 'difficulty': 'medium'},
        {'exercise_type': 'multiple_choice', 'difficulty': 'medium'},
        {'exercise_type': 'fill_blank', 'difficulty': 'medium'},
        {'exercise_type': 'multiple_choice', 'difficulty': 'hard'},
        {'exercise_type': 'fill_blank', 'difficulty': 'hard'},
        {'exercise_type': 'multiple_choice', 'difficulty': 'hard'},
    ]],
    2: [
        [
            {'exercise_type': 'multiple_choice', 'difficulty': 'easy'},
            {'exercise_type': 'fill_blank', 'difficulty': 'easy'},
            {'exercise_type': 'multiple_choice', 'difficulty': 'medium'},
            {'exercise_type': 'fill_blank', 'difficulty': 'medium'},
            {'exercise_type': 'multiple_choice', 'difficulty': 'hard'},
        ],
        [
            {'exercise_type': 'multiple_choice', 'difficulty': 'easy'},
            {'exercise_type': 'fill_blank', 'difficulty': 'medium'},
            {'exercise_type': 'multiple_choice', 'difficulty': 'medium'},
            {'exercise_type': 'fill_blank', 'difficulty': 'hard'},
            {'exercise_type': 'multiple_choice', 'difficulty': 'hard'},
        ],
    ],
    3: [
        [
            {'exercise_type': 'multiple_choice', 'difficulty': 'easy'},
            {'exercise_type': 'fill_blank', 'difficulty': 'medium'},
            {'exercise_type': 'multiple_choice', 'difficulty': 'medium'},
            {'exercise_type': 'fill_blank', 'difficulty': 'hard'},
        ],
        [
            {'exercise_type': 'multiple_choice', 'difficulty': 'easy'},
            {'exercise_type': 'multiple_choice', 'difficulty': 'medium'},
            {'exercise_type': 'fill_blank', 'difficulty': 'hard'},
        ],
        [
            {'exercise_type': 'multiple_choice', 'difficulty': 'easy'},
            {'exercise_type': 'multiple_choice', 'difficulty': 'medium'},
            {'exercise_type': 'fill_blank', 'difficulty': 'hard'},
        ],
    ],
}


def _rank_personalized_concepts(user):
    mistakes = list(PracticeMistake.objects.filter(
        owner=user,
        source='editor_live',
    ).annotate(
        wrong_count=Count('exercises__session_items', filter=Q(exercises__session_items__is_correct=False)),
    ).order_by('-wrong_count', '-occurrence_count', '-last_seen_at'))
    grouped = {}
    for mistake in mistakes:
        group = grouped.setdefault(mistake.concept_code, {
            'code': mistake.concept_code,
            'label': mistake.concept_label,
            'wrong_count': 0,
            'occurrences': 0,
            'last_seen_at': mistake.last_seen_at,
            'mistakes': [],
        })
        group['wrong_count'] += mistake.wrong_count
        group['occurrences'] += mistake.occurrence_count
        group['last_seen_at'] = max(group['last_seen_at'], mistake.last_seen_at)
        group['mistakes'].append(mistake)
    return sorted(
        grouped.values(),
        key=lambda group: (group['wrong_count'], group['occurrences'], group['last_seen_at']),
        reverse=True,
    )[:3]


def _matches_slot(exercise, slot):
    return exercise.exercise_type == slot['exercise_type'] and exercise.difficulty == slot['difficulty']


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_practice_session(request):
    mode = request.data.get('mode')
    # Compatibility for clients created before the two-card Practice redesign.
    if mode not in {'personalized', 'general'}:
        mode = 'personalized' if request.data.get('topic') == 'recommended' else 'general'
    profile, _ = PracticeProfile.objects.get_or_create(owner=request.user)
    difficulty = 'mixed' if mode == 'personalized' else profile.general_level
    recent_rows = list(PracticeSessionItem.objects.filter(
        session__owner=request.user,
        **({'session__mode': 'personalized'} if mode == 'personalized' else {}),
    ).order_by('-session__created_at', '-order').values_list(
        'exercise__prompt', 'exercise__generation_key',
    )[:30])
    recent_prompts = [row[0] for row in recent_rows]
    recent_generation_keys = {row[1] for row in recent_rows if row[1]}
    selected = []
    source_summary = {}
    focus_concepts = []

    if mode == 'personalized':
        concept_groups = _rank_personalized_concepts(request.user)
        if not concept_groups:
            return Response({'error': 'Write in the Editor first so Practice can learn from your grammar mistakes.'}, status=status.HTTP_400_BAD_REQUEST)
        slot_groups = PERSONALIZED_SLOT_PLANS[len(concept_groups)]
        selected_keys = set(recent_generation_keys)

        def select_personalized(exercise):
            selected.append(exercise)
            selected_keys.add(exercise.generation_key)
            source_summary[exercise.source] = source_summary.get(exercise.source, 0) + 1

        for group, slots in zip(concept_groups, slot_groups):
            representatives = group['mistakes'][:3]
            representative = representatives[0]
            concept_code = group['code']
            topic = CONCEPT_TOPICS.get(concept_code, representative.category)
            focus_concepts.append({'code': concept_code, 'label': group['label']})
            pending_slots = [dict(slot) for slot in slots]

            cached = list(PracticeExercise.objects.filter(
                owner=request.user,
                mode='personalized',
                used_count=0,
                mistake__concept_code=concept_code,
            ).exclude(generation_key='').order_by('created_at'))
            for slot in list(pending_slots):
                exercise = next((item for item in cached if (
                    _matches_slot(item, slot) and
                    item.generation_key not in selected_keys and
                    not is_recent_prompt(item.prompt, [*recent_prompts, *[chosen.prompt for chosen in selected]])
                )), None)
                if exercise:
                    select_personalized(exercise)
                    cached.remove(exercise)
                    pending_slots.remove(slot)

            if pending_slots:
                extra_slots = [dict(pending_slots[index % len(pending_slots)]) for index in range(min(2, len(pending_slots)))]
                requested_slots = [*pending_slots, *extra_slots]
                requested_types = {
                    exercise_type: sum(slot['exercise_type'] == exercise_type for slot in requested_slots)
                    for exercise_type in TYPE_TARGETS
                }
                generated, gemini_source = generate_with_gemini(
                    representatives,
                    topic,
                    len(requested_slots),
                    mode,
                    'mixed',
                    recent_prompts,
                    requested_types,
                    requested_slots,
                )
                generated_exercises = []
                for index, data in enumerate(generated):
                    data['category'] = topic
                    data['generation_key'] = fingerprint(
                        concept_code,
                        data['difficulty'],
                        data['exercise_type'],
                        prompt_template_signature(data['prompt']),
                        data['accepted_answers'][0],
                    )
                    if (
                        data['generation_key'] in selected_keys or
                        is_recent_prompt(data['prompt'], [*recent_prompts, *[chosen.prompt for chosen in selected]])
                    ):
                        continue
                    exercise = _save_exercise(
                        request.user,
                        data,
                        gemini_source or 'gemini_primary',
                        representatives[index % len(representatives)],
                        mode,
                        data['difficulty'],
                    )
                    generated_exercises.append(exercise)
                    selected_keys.add(exercise.generation_key)

                for slot in list(pending_slots):
                    exercise = next((item for item in generated_exercises if _matches_slot(item, slot)), None)
                    if exercise:
                        select_personalized(exercise)
                        generated_exercises.remove(exercise)
                        pending_slots.remove(slot)

            if pending_slots:
                fallback_items = personalized_fallback_exercises(
                    representative,
                    pending_slots,
                    [*recent_prompts, *[chosen.prompt for chosen in selected]],
                    selected_keys,
                )
                for slot, data in zip(list(pending_slots), fallback_items):
                    exercise = _save_exercise(
                        request.user,
                        data,
                        'template',
                        representative,
                        mode,
                        data['difficulty'],
                    )
                    select_personalized(exercise)
                    pending_slots.remove(slot)
    else:
        targets = dict(TYPE_TARGETS)
        generated, gemini_source = generate_with_gemini(
            [], 'mixed_grammar', SESSION_SIZE, mode, difficulty, recent_prompts, targets
        )
        for data in generated:
            if (
                targets.get(data['exercise_type'], 0) <= 0 or
                is_recent_prompt(data['prompt'], recent_prompts, compare_templates=True) or
                is_recent_prompt(data['prompt'], [x.prompt for x in selected], compare_templates=True)
            ):
                continue
            exercise = _save_exercise(request.user, data, gemini_source or 'gemini_primary', None, mode, difficulty)
            if exercise.session_items.exists():
                continue
            selected.append(exercise)
            targets[data['exercise_type']] -= 1
            source_summary[exercise.source] = source_summary.get(exercise.source, 0) + 1
        missing = sum(targets.values())
        for data in general_fallback_exercises(difficulty, missing, targets, [*recent_prompts, *[x.prompt for x in selected]]):
            exercise = _save_exercise(request.user, data, 'template', None, mode, difficulty)
            if exercise.session_items.exists():
                exercise.pk = None
                exercise.fingerprint = fingerprint(data['exercise_type'], data['prompt'], timezone.now().isoformat())
                exercise.save()
            selected.append(exercise)
            source_summary['template'] = source_summary.get('template', 0) + 1

    selected = selected[:SESSION_SIZE]
    if len(selected) != SESSION_SIZE:
        return Response({'error': 'Unable to prepare a complete practice session.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    if mode == 'personalized':
        ordered = []
        for level in ('easy', 'medium', 'hard'):
            level_items = [exercise for exercise in selected if exercise.difficulty == level]
            random.shuffle(level_items)
            ordered.extend(level_items)
        selected = ordered
    else:
        random.shuffle(selected)

    with transaction.atomic():
        session = PracticeSession.objects.create(
            owner=request.user,
            topic=mode,
            mode=mode,
            difficulty=difficulty,
            focus_concepts=focus_concepts,
            total_questions=SESSION_SIZE,
            source_summary=source_summary,
        )
        for order, exercise in enumerate(selected, start=1):
            PracticeSessionItem.objects.create(session=session, exercise=exercise, order=order)
            PracticeExercise.objects.filter(pk=exercise.pk).update(used_count=F('used_count') + 1)
    return Response(_session_payload(session), status=status.HTTP_201_CREATED)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def retrieve_practice_session(request, session_id):
    session = PracticeSession.objects.filter(id=session_id, owner=request.user).first()
    if session is None:
        return Response({'error': 'Practice session not found.'}, status=status.HTTP_404_NOT_FOUND)
    return Response(_session_payload(session))


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def answer_practice_session(request, session_id):
    session = PracticeSession.objects.filter(id=session_id, owner=request.user).first()
    if session is None:
        return Response({'error': 'Practice session not found.'}, status=status.HTTP_404_NOT_FOUND)
    exercise_id = request.data.get('exercise_id')
    answer = request.data.get('answer')
    if not isinstance(answer, str) or not answer.strip() or len(answer) > 500:
        return Response({'error': 'answer must be a non-empty string of at most 500 characters.'}, status=status.HTTP_400_BAD_REQUEST)
    try:
        item = session.items.select_related('exercise').filter(exercise_id=exercise_id).first()
    except (ValueError, ValidationError):
        item = None
    if item is None:
        return Response({'error': 'Exercise not found in this session.'}, status=status.HTTP_404_NOT_FOUND)

    if item.is_correct is None:
        accepted = {normalize_answer(value) for value in item.exercise.accepted_answers}
        item.user_answer = answer.strip()
        item.is_correct = normalize_answer(answer) in accepted
        item.answered_at = timezone.now()
        item.save(update_fields=['user_answer', 'is_correct', 'answered_at'])
        if session.mode == 'general':
            profile, _ = PracticeProfile.objects.get_or_create(owner=request.user)
            profile.level_answered += 1
            if item.is_correct:
                profile.level_correct += 1
            if profile.level_answered >= 10:
                accuracy = profile.level_correct / profile.level_answered
                levels = ['easy', 'medium', 'hard']
                current = levels.index(profile.general_level)
                if accuracy >= 0.8:
                    profile.general_level = levels[min(current + 1, len(levels) - 1)]
                elif accuracy <= 0.5:
                    profile.general_level = levels[max(current - 1, 0)]
                profile.level_answered = 0
                profile.level_correct = 0
            profile.save()
        if not session.items.filter(is_correct__isnull=True).exists():
            session.completed_at = timezone.now()
            session.save(update_fields=['completed_at'])

    item.refresh_from_db()
    payload = _public_item(item)
    payload['session'] = _session_payload(session)
    return Response(payload)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def practice_progress(request):
    items = PracticeSessionItem.objects.filter(session__owner=request.user, session__mode__in=['personalized', 'general'], is_correct__isnull=False)
    totals = items.aggregate(
        answered=Count('id'),
        correct=Count('id', filter=Q(is_correct=True)),
    )
    answered = totals['answered'] or 0
    correct = totals['correct'] or 0
    return Response({
        'answered': answered,
        'correct': correct,
        'accuracy': round(correct * 100 / answered) if answered else None,
        'completed_sessions': PracticeSession.objects.filter(owner=request.user, completed_at__isnull=False).count(),
        'modes': {
            mode: {
                'answered': PracticeSessionItem.objects.filter(session__owner=request.user, session__mode=mode, is_correct__isnull=False).count(),
                'correct': PracticeSessionItem.objects.filter(session__owner=request.user, session__mode=mode, is_correct=True).count(),
            }
            for mode in ('personalized', 'general')
        },
    })
