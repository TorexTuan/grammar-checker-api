import hashlib
import json
import os
import random
import re

from django.core.cache import cache
from google import genai
from google.genai import types

from .gemini import GEMINI_FALLBACK_MODEL, GEMINI_MODEL


TOPICS = {
    'verb_tense': 'Verb tenses',
    'subject_verb_agreement': 'Subject–verb agreement',
    'articles': 'Articles & determiners',
    'prepositions': 'Prepositions',
    'pronouns': 'Pronouns',
    'word_form': 'Word forms',
    'sentence_structure': 'Sentence structure',
    'modals': 'Modal verbs',
    'conditionals': 'Conditionals',
    'passive_voice': 'Passive voice',
    'reported_speech': 'Reported speech',
    'gerunds_infinitives': 'Gerunds & infinitives',
    'relative_clauses': 'Relative clauses',
    'comparisons': 'Comparatives & superlatives',
    'quantifiers': 'Quantifiers',
    'spelling_punctuation': 'Spelling & punctuation',
    'other_grammar': 'Other grammar',
}

SESSION_SIZE = 10
TYPE_TARGETS = {'multiple_choice': 6, 'fill_blank': 4}
BREAKER_KEY = 'practice:gemini:failures'
BREAKER_OPEN_KEY = 'practice:gemini:open'

CONCEPT_LABELS = {
    'past_simple': 'Past simple', 'past_continuous': 'Past continuous',
    'present_simple': 'Present simple', 'present_continuous': 'Present continuous',
    'present_perfect': 'Present perfect', 'future_forms': 'Future forms',
    'verb_tense': 'Verb tenses', 'subject_verb_agreement': 'Subject–verb agreement',
    'articles': 'Articles & determiners', 'prepositions': 'Prepositions',
    'pronouns': 'Pronouns', 'word_form': 'Word forms',
    'sentence_structure': 'Sentence structure', 'other_grammar': 'Other grammar',
    'modals': 'Modal verbs', 'conditionals': 'Conditionals',
    'passive_voice': 'Passive voice', 'reported_speech': 'Reported speech',
    'gerunds_infinitives': 'Gerunds & infinitives',
    'relative_clauses': 'Relative clauses', 'comparisons': 'Comparatives & superlatives',
    'quantifiers': 'Quantifiers',
}

CONCEPT_TOPICS = {
    'past_simple': 'verb_tense',
    'past_continuous': 'verb_tense',
    'present_simple': 'verb_tense',
    'present_continuous': 'verb_tense',
    'present_perfect': 'verb_tense',
    'future_forms': 'verb_tense',
    'verb_tense': 'verb_tense',
    'subject_verb_agreement': 'subject_verb_agreement',
    'articles': 'articles',
    'prepositions': 'prepositions',
    'pronouns': 'pronouns',
    'word_form': 'word_form',
    'sentence_structure': 'sentence_structure',
    'modals': 'modals',
    'conditionals': 'conditionals',
    'passive_voice': 'passive_voice',
    'reported_speech': 'reported_speech',
    'gerunds_infinitives': 'gerunds_infinitives',
    'relative_clauses': 'relative_clauses',
    'comparisons': 'comparisons',
    'quantifiers': 'quantifiers',
}


def normalize_topic(category):
    value = (category or '').casefold()
    mappings = (
        ('subject_verb_agreement', ('subject-verb', 'subject verb', 'agreement', 'sva')),
        ('verb_tense', ('tense', 'verb:tense')),
        ('articles', ('article', 'determiner', 'det:art')),
        ('prepositions', ('preposition', 'prep')),
        ('pronouns', ('pronoun', 'pron')),
        ('word_form', ('word form', 'word_form', 'noun form', 'adjective form', 'adverb form')),
        ('sentence_structure', ('sentence', 'fragment', 'run-on', 'clause', 'structure')),
        ('spelling_punctuation', ('spelling', 'spell', 'punctuation', 'punct')),
    )
    for topic, markers in mappings:
        if any(marker in value for marker in markers):
            return topic
    return value if value in TOPICS else 'other_grammar'


def normalize_answer(value):
    return re.sub(r'\s+', ' ', str(value or '')).strip().casefold()


def infinitive_hint(value):
    """Return a useful base-form hint for subject-verb agreement exercises."""
    verb = str(value or '').strip()
    lowered = verb.casefold()
    irregular = {
        'am': 'be', 'is': 'be', 'are': 'be', 'was': 'be', 'were': 'be',
        'has': 'have', 'had': 'have',
        'does': 'do', 'did': 'do',
    }
    if lowered in irregular:
        return irregular[lowered]
    if lowered.endswith('ies') and len(verb) > 3:
        return f'{verb[:-3]}y'
    if lowered.endswith(('ches', 'shes', 'sses', 'xes', 'zes', 'oes')):
        return verb[:-2]
    if lowered.endswith('s') and not lowered.endswith('ss'):
        return verb[:-1]
    return verb


def fingerprint(*values):
    normalized = '|'.join(normalize_answer(value) for value in values)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def sentence_context(text, offset):
    offset = max(0, min(int(offset or 0), len(text)))
    start = max(text.rfind(mark, 0, offset) for mark in '.!?\n') + 1
    ends = [text.find(mark, offset) for mark in '.!?\n']
    ends = [position for position in ends if position >= 0]
    end = min(ends) + 1 if ends else len(text)
    return text[start:end].strip()[:500]


def classify_concept(issue, context=''):
    suggested = str(issue.get('concept_code') or '').strip()
    category = str(issue.get('category') or '').casefold()
    detail = f"{issue.get('error_type', '')} {category}".casefold()
    sample = f"{context} {issue.get('original', '')} {issue.get('replacement', '')}".casefold()
    # Sapling exposes spelling, punctuation and style through the category label.
    if (category and 'grammar' not in category) or any(marker in detail for marker in ('spelling', 'punctuation', 'style', 'clarity', 'fluency')):
        return None
    if suggested in CONCEPT_LABELS:
        return suggested, CONCEPT_LABELS[suggested]
    if any(marker in detail for marker in ('subject-verb', 'subject verb', 'agreement', 'sva')):
        code = 'subject_verb_agreement'
    elif any(marker in detail for marker in ('article', 'determiner')):
        code = 'articles'
    elif 'preposition' in detail:
        code = 'prepositions'
    elif 'pronoun' in detail:
        code = 'pronouns'
    elif any(marker in detail for marker in ('word form', 'noun form', 'adjective form', 'adverb form')):
        code = 'word_form'
    elif any(marker in detail for marker in ('fragment', 'run-on', 'clause', 'sentence structure')):
        code = 'sentence_structure'
    elif any(marker in detail for marker in ('tense', 'verb')):
        if re.search(r'\b(yesterday|last\s+\w+|ago|in \d{4})\b', sample):
            code = 'past_continuous' if re.search(r'\b(was|were)\s+\w+ing\b', sample) else 'past_simple'
        elif re.search(r'\b(since|for|already|yet|ever|never|just)\b', sample) and re.search(r'\b(has|have|had)\b', sample):
            code = 'present_perfect'
        elif re.search(r'\b(now|currently|at the moment)\b', sample):
            code = 'present_continuous'
        elif re.search(r'\b(tomorrow|next\s+\w+|will|going to)\b', sample):
            code = 'future_forms'
        elif re.search(r'\b(every|usually|often|always|sometimes)\b', sample):
            code = 'present_simple'
        else:
            code = 'verb_tense'
    else:
        code = normalize_topic(detail)
        if code == 'spelling_punctuation':
            return None
    return code, CONCEPT_LABELS.get(code, TOPICS.get(code, 'Other grammar'))


def enrich_practice_issues(issues, text):
    for issue in issues:
        context = sentence_context(text, issue.get('offset', 0))
        concept = classify_concept(issue, context)
        issue['sentence_context'] = context
        if concept:
            code, label = concept
            issue['concept_code'] = code
            issue['concept_label'] = label
            issue['practice_fingerprint'] = fingerprint(code, issue.get('original', ''), issue.get('replacement', ''), context)
        else:
            issue['practice_fingerprint'] = ''
    return issues


def record_detected_mistakes(user, issues, excluded=None):
    if not getattr(user, 'is_authenticated', False):
        return 0
    from django.db.models import F
    from django.utils import timezone
    from api.models import PracticeMistake

    excluded = set(excluded or [])
    recorded = 0
    for issue in issues:
        key = issue.get('practice_fingerprint')
        if not key or key in excluded:
            continue
        defaults = {
            'category': normalize_topic(issue.get('category', '')),
            'concept_code': issue.get('concept_code', 'other_grammar'),
            'concept_label': issue.get('concept_label', 'Other grammar'),
            'source': 'editor_live',
            'sentence_context': issue.get('sentence_context', '')[:500],
            'original': str(issue.get('original') or '')[:300],
            'correction': str(issue.get('replacement') or '')[:300],
            'explanation': str(issue.get('explanation') or '')[:2000],
        }
        mistake, created = PracticeMistake.objects.get_or_create(owner=user, fingerprint=key, defaults=defaults)
        if not created:
            mistake.occurrence_count = F('occurrence_count') + 1
            mistake.last_seen_at = timezone.now()
            mistake.save(update_fields=['occurrence_count', 'last_seen_at'])
        recorded += 1
    return recorded


PRACTICE_PROMPT = """
You create safe English grammar exercises for a Vietnamese learner at the requested difficulty.
Return JSON only as an array. Create exactly the requested count and match the
required_types mix and every exercise_type/difficulty pair in requested_slots.
Arrange the language from clear single-clause easy items, through distracting
medium structures, to multi-clause hard items. Do not copy personal names,
places, employers, contact details, or distinctive facts from the source sentence;
create analogous neutral examples instead.

For general mode, cover at least six distinct grammar categories in a ten-item
session. Test genuinely different rules rather than paraphrasing one sentence.
Use advanced categories such as conditionals, passive voice, reported speech,
relative clauses, modal verbs, gerunds/infinitives, comparisons, and quantifiers
when appropriate for the requested difficulty. Difficulty must come from the
grammar and sentence structure, not from adding a decorative introductory phrase.

Each item must have:
{"exercise_type":"multiple_choice|fill_blank","category":"allowed category code",
 "difficulty":"easy|medium|hard",
 "prompt":"English question","options":["four unique options for multiple choice"],
 "accepted_answers":["at least one exact answer"],"hint":"base word",
 "explanation":"short Vietnamese explanation"}

For fill_blank, options must be []. For multiple_choice, accepted_answers[0] must
appear in options. Include exactly one blank written as ___ in fill_blank prompts.
For a verb exercise, hint is required and must be the infinitive/base form of the
verb (for example, go instead of went/goes). For a non-verb blank, hint must be a
short base-word or word-class cue. For multiple_choice, hint must be an empty string.
"""


def _clean_strings(value, maximum, item_maximum=300):
    if not isinstance(value, list):
        return []
    result = []
    for item in value[:maximum]:
        if isinstance(item, str) and item.strip():
            cleaned = item.strip()[:item_maximum]
            if normalize_answer(cleaned) not in {normalize_answer(x) for x in result}:
                result.append(cleaned)
    return result


def validate_generated_exercise(item, requested_topic, default_difficulty='medium'):
    if not isinstance(item, dict):
        return None
    exercise_type = item.get('exercise_type')
    if exercise_type not in TYPE_TARGETS:
        return None
    category = normalize_topic(item.get('category') or requested_topic)
    difficulty = item.get('difficulty') or default_difficulty
    if difficulty not in {'easy', 'medium', 'hard'}:
        return None
    prompt = item.get('prompt')
    explanation = item.get('explanation')
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt.strip()) > 800:
        return None
    if not isinstance(explanation, str) or not explanation.strip() or len(explanation.strip()) > 1500:
        return None
    answers = _clean_strings(item.get('accepted_answers'), 5)
    if not answers:
        return None
    options = _clean_strings(item.get('options'), 4)
    hint = item.get('hint', '')
    if not isinstance(hint, str) or len(hint.strip()) > 120:
        return None
    hint = hint.strip()
    if exercise_type == 'multiple_choice':
        if len(options) != 4 or normalize_answer(answers[0]) not in {normalize_answer(x) for x in options}:
            return None
        hint = ''
    elif options or prompt.count('___') != 1 or not hint:
        return None
    return {
        'exercise_type': exercise_type,
        'category': category,
        'difficulty': difficulty,
        'prompt': prompt.strip(),
        'hint': hint,
        'options': options,
        'accepted_answers': answers,
        'explanation': explanation.strip(),
    }


def _retryable(exc):
    message = f'{type(exc).__name__} {exc}'.casefold()
    return any(marker in message for marker in (
        'timeout', 'timed out', 'deadline', '429', '500', '502', '503', '504',
        'resource_exhausted', 'unavailable', 'high demand',
    ))


def _record_provider_success():
    cache.delete(BREAKER_KEY)
    cache.delete(BREAKER_OPEN_KEY)


def _record_provider_failure():
    failures = cache.get(BREAKER_KEY, 0) + 1
    cache.set(BREAKER_KEY, failures, 90)
    if failures >= 3:
        cache.set(BREAKER_OPEN_KEY, True, 90)


def _practice_client(timeout_ms):
    api_key = os.getenv('GEMINI_API_KEY', '').strip()
    if not api_key:
        return None
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=timeout_ms,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def generate_with_gemini(
    mistakes, topic, count=SESSION_SIZE, mode='personalized', difficulty='medium',
    recent_prompts=None, required_types=None, requested_slots=None,
):
    if cache.get(BREAKER_OPEN_KEY):
        return [], None
    request_payload = {
        'count': count,
        'required_types': dict(required_types or TYPE_TARGETS),
        'difficulty': difficulty,
        'requested_slots': list(requested_slots or []),
        'mode': mode,
        'topic': topic,
        'avoid_prompts': list(recent_prompts or [])[:30],
        'allowed_categories': list(TOPICS),
        'mistakes': [
            {
                'category': mistake.category,
                'sentence_context': mistake.sentence_context,
                'original': mistake.original,
                'correction': mistake.correction,
                'explanation': mistake.explanation[:500],
            }
            for mistake in mistakes[:3]
        ],
    }
    model_attempts = ((GEMINI_MODEL, 6000, 'gemini_primary'), (GEMINI_FALLBACK_MODEL, 4000, 'gemini_secondary'))
    for model, timeout_ms, source in model_attempts:
        client = _practice_client(timeout_ms)
        if not client:
            _record_provider_failure()
            return [], None
        try:
            response = client.models.generate_content(
                model=model,
                contents=json.dumps(request_payload, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=PRACTICE_PROMPT,
                    response_mime_type='application/json',
                    temperature=0.65,
                    thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
                ),
            )
            payload = json.loads((response.text or '').strip())
            if isinstance(payload, dict):
                payload = payload.get('exercises')
            if not isinstance(payload, list):
                raise ValueError('Gemini returned an invalid exercise list.')
            valid = []
            seen = set()
            for item in payload:
                exercise = validate_generated_exercise(item, topic, 'medium' if difficulty == 'mixed' else difficulty)
                if not exercise:
                    continue
                key = fingerprint(exercise['exercise_type'], exercise['prompt'])
                if key not in seen:
                    seen.add(key)
                    valid.append(exercise)
            if valid:
                _record_provider_success()
                return valid, source
            raise ValueError('Gemini returned no valid exercises.')
        except Exception as exc:
            print(f'[Practice Gemini Warning] {model}: {exc}')
            _record_provider_failure()
            if model == GEMINI_MODEL and (_retryable(exc) or isinstance(exc, (ValueError, json.JSONDecodeError))):
                continue
            break
    return [], None


FALLBACK_BANK = {
    'verb_tense': [
        ('multiple_choice', 'Yesterday, Maya ___ the report before lunch.', ['finishes', 'finished', 'has finish', 'finishing'], 'finished', '“Yesterday” là dấu hiệu của thì quá khứ đơn, nên dùng “finished”.'),
        ('multiple_choice', 'By the time we arrived, the film ___.', ['starts', 'started', 'had started', 'has started'], 'had started', 'Hành động bắt đầu xảy ra trước một hành động khác trong quá khứ, nên dùng quá khứ hoàn thành.'),
        ('multiple_choice', 'She usually ___ coffee in the morning.', ['drink', 'drinks', 'drank', 'drinking'], 'drinks', 'Thói quen dùng hiện tại đơn; chủ ngữ số ít “she” đi với “drinks”.'),
        ('fill_blank', 'They ___ the museum last weekend.', [], 'visited', '“Last weekend” yêu cầu động từ ở thì quá khứ đơn.'),
        ('fill_blank', 'I have ___ this book twice.', [], 'read', 'Sau “have” trong hiện tại hoàn thành cần quá khứ phân từ “read”.'),
    ],
    'subject_verb_agreement': [
        ('multiple_choice', 'The list of items ___ on the desk.', ['are', 'were', 'is', 'be'], 'is', 'Chủ ngữ chính “list” là số ít nên dùng “is”.'),
        ('multiple_choice', 'My friends ___ near the university.', ['lives', 'live', 'living', 'has lived'], 'live', 'Chủ ngữ số nhiều “friends” dùng động từ nguyên mẫu “live”.'),
        ('multiple_choice', 'Each student ___ a notebook.', ['have', 'having', 'has', 'are having'], 'has', '“Each student” được xem là chủ ngữ số ít nên dùng “has”.'),
        ('fill_blank', 'The dog ___ loudly every night.', [], 'barks', 'Chủ ngữ số ít ở hiện tại đơn cần động từ thêm “s”.'),
        ('fill_blank', 'We ___ ready to begin.', [], 'are', 'Chủ ngữ “we” đi với động từ “are”.'),
    ],
    'articles': [
        ('multiple_choice', 'She bought ___ umbrella because it was raining.', ['a', 'an', 'the', 'no article'], 'an', '“Umbrella” bắt đầu bằng nguyên âm nên dùng “an”.'),
        ('multiple_choice', 'I saw ___ moon clearly last night.', ['a', 'an', 'the', 'no article'], 'the', 'Mặt trăng là đối tượng xác định và duy nhất nên dùng “the”.'),
        ('multiple_choice', 'He wants to become ___ engineer.', ['a', 'an', 'the', 'no article'], 'an', 'Nghề nghiệp số ít và “engineer” bắt đầu bằng nguyên âm nên dùng “an”.'),
        ('fill_blank', 'Please close ___ door next to you.', [], 'the', 'Cánh cửa được xác định bởi cụm “next to you”, nên dùng “the”.'),
        ('fill_blank', 'She adopted ___ small cat.', [], 'a', 'Danh từ số ít chưa xác định “small cat” dùng “a”.'),
    ],
    'prepositions': [
        ('multiple_choice', 'The meeting starts ___ 9 a.m.', ['in', 'on', 'at', 'by'], 'at', 'Dùng “at” trước một thời điểm cụ thể.'),
        ('multiple_choice', 'We will travel ___ Monday.', ['at', 'on', 'in', 'from'], 'on', 'Dùng “on” trước ngày trong tuần.'),
        ('multiple_choice', 'The keys are ___ the table.', ['on', 'at', 'into', 'during'], 'on', '“On” diễn tả vị trí trên một bề mặt.'),
        ('fill_blank', 'She is interested ___ photography.', [], 'in', 'Cụm đúng là “interested in”.'),
        ('fill_blank', 'They arrived ___ the airport early.', [], 'at', 'Dùng “arrive at” với một địa điểm cụ thể.'),
    ],
    'pronouns': [
        ('multiple_choice', 'Lena and I finished the project by ___.', ['ourselves', 'themselves', 'myself', 'us'], 'ourselves', 'Chủ ngữ “Lena and I” tương ứng với đại từ phản thân “ourselves”.'),
        ('multiple_choice', 'The teacher spoke to Sam and ___.', ['I', 'me', 'my', 'mine'], 'me', 'Sau giới từ “to” cần đại từ tân ngữ “me”.'),
        ('multiple_choice', 'This jacket belongs to ___.', ['she', 'her', 'hers', 'herself'], 'her', 'Sau giới từ “to” dùng đại từ tân ngữ “her”.'),
        ('fill_blank', 'Alex gave ___ a useful suggestion.', [], 'me', 'Vị trí tân ngữ gián tiếp cần đại từ tân ngữ “me”.'),
        ('fill_blank', 'The children prepared the meal by ___.', [], 'themselves', 'Chủ ngữ số nhiều “children” dùng “themselves”.'),
    ],
    'word_form': [
        ('multiple_choice', 'She completed the task ___.', ['success', 'successful', 'successfully', 'succeed'], 'successfully', 'Cần trạng từ “successfully” để bổ nghĩa cho động từ “completed”.'),
        ('multiple_choice', 'The instructions were very ___.', ['help', 'helpfully', 'helpful', 'helped'], 'helpful', 'Sau “were very” cần tính từ “helpful”.'),
        ('multiple_choice', 'His ___ surprised everyone.', ['decide', 'decision', 'decisive', 'deciding'], 'decision', 'Sau tính từ sở hữu “his” cần danh từ “decision”.'),
        ('fill_blank', 'The team worked ___ to meet the deadline.', [], 'quickly', 'Cần trạng từ “quickly” để bổ nghĩa cho “worked”.'),
        ('fill_blank', 'It was an ___ performance.', [], 'impressive', 'Trước danh từ “performance” cần tính từ “impressive”.'),
    ],
    'sentence_structure': [
        ('multiple_choice', 'Choose the complete sentence.', ['Because the rain was heavy.', 'When we reached the station.', 'We stayed inside because it rained.', 'Although feeling tired.'], 'We stayed inside because it rained.', 'Câu này có đầy đủ chủ ngữ, động từ và ý nghĩa hoàn chỉnh.'),
        ('multiple_choice', 'Choose the correctly joined sentence.', ['I was tired, I kept working.', 'I was tired, but I kept working.', 'I was tired but, I kept working.', 'I was tired; but kept working.'], 'I was tired, but I kept working.', 'Hai mệnh đề độc lập được nối đúng bằng dấu phẩy và “but”.'),
        ('multiple_choice', 'Which sentence avoids a fragment?', ['After the meeting ended.', 'The meeting ended, and we left.', 'While waiting for the bus.', 'Because she was late.'], 'The meeting ended, and we left.', 'Câu có hai mệnh đề hoàn chỉnh được nối đúng.'),
        ('fill_blank', 'I called Mia, ___ she did not answer.', [], 'but', '“But” nối hai ý tương phản trong hai mệnh đề.'),
        ('fill_blank', 'We left early ___ we could catch the train.', [], 'so that', '“So that” diễn tả mục đích của hành động rời đi sớm.'),
    ],
    'spelling_punctuation': [
        ('multiple_choice', 'Choose the correctly spelled word.', ['definately', 'definitely', 'definetly', 'definatly'], 'definitely', '“Definitely” là cách viết đúng.'),
        ('multiple_choice', 'Choose the correctly punctuated sentence.', ['Lets eat, Grandma!', "Let's eat Grandma!", "Let's eat, Grandma!", 'Lets eat Grandma.'], "Let's eat, Grandma!", 'Dấu nháy trong “Let’s” và dấu phẩy gọi “Grandma” đều cần thiết.'),
        ('multiple_choice', 'Which word is spelled correctly?', ['seperate', 'separate', 'seperete', 'separete'], 'separate', '“Separate” là cách viết đúng.'),
        ('fill_blank', 'Complete the spelling: neces___ary', [], 's', 'Từ “necessary” có một chữ c và hai chữ s.'),
        ('fill_blank', 'Add the missing punctuation: Where are you___', [], '?', 'Câu hỏi trực tiếp kết thúc bằng dấu chấm hỏi.'),
    ],
    'other_grammar': [
        ('multiple_choice', 'Choose the correct sentence.', ['She can sings well.', 'She can sing well.', 'She can to sing well.', 'She cans sing well.'], 'She can sing well.', 'Sau động từ khuyết thiếu “can” dùng động từ nguyên mẫu.'),
        ('multiple_choice', 'Choose the natural sentence.', ['There is many reasons.', 'There are many reasons.', 'There be many reasons.', 'There has many reasons.'], 'There are many reasons.', 'Danh từ số nhiều “reasons” đi với “there are”.'),
        ('multiple_choice', 'Choose the correct form.', ['I enjoy to read.', 'I enjoy read.', 'I enjoy reading.', 'I enjoying reading.'], 'I enjoy reading.', 'Sau “enjoy” dùng động từ dạng V-ing.'),
        ('fill_blank', 'You should ___ more water.', [], 'drink', 'Sau “should” dùng động từ nguyên mẫu.'),
        ('fill_blank', 'There ___ two books on the desk.', [], 'are', 'Danh từ số nhiều “two books” đi với “there are”.'),
    ],
}

FALLBACK_HINTS = {
    ('verb_tense', 'visited'): 'visit', ('verb_tense', 'read'): 'read',
    ('subject_verb_agreement', 'barks'): 'bark', ('subject_verb_agreement', 'are'): 'be',
    ('articles', 'the'): 'article', ('articles', 'a'): 'article',
    ('prepositions', 'in'): 'preposition', ('prepositions', 'at'): 'preposition',
    ('pronouns', 'me'): 'pronoun', ('pronouns', 'themselves'): 'pronoun',
    ('word_form', 'quickly'): 'quick', ('word_form', 'impressive'): 'impress',
    ('sentence_structure', 'but'): 'conjunction', ('sentence_structure', 'so that'): 'conjunction',
    ('spelling_punctuation', 's'): 'necessary', ('spelling_punctuation', '?'): 'punctuation',
    ('other_grammar', 'drink'): 'drink', ('other_grammar', 'are'): 'be',
}


def fallback_exercises(topic, count=SESSION_SIZE, required_types=None):
    topic = topic if topic in TOPICS else 'other_grammar'
    bank = list(FALLBACK_BANK[topic])
    random.shuffle(bank)
    targets = dict(required_types or TYPE_TARGETS)
    result = []
    for exercise_type, prompt, options, answer, explanation in bank:
        if targets.get(exercise_type, 0) <= 0:
            continue
        result.append({
            'exercise_type': exercise_type,
            'category': topic,
            'prompt': prompt,
            'hint': FALLBACK_HINTS.get((topic, normalize_answer(answer)), '') if exercise_type == 'fill_blank' else '',
            'options': options,
            'accepted_answers': [answer],
            'explanation': explanation,
        })
        targets[exercise_type] -= 1
        if len(result) >= count:
            break
    return result


FALLBACK_CONTEXTS = (
    'In a classroom exercise', 'For a travel journal', 'In a short email',
    'During a team meeting', 'For a daily conversation', 'In a school project',
    'At a community event', 'For a weekend plan', 'In a work update',
    'During an English lesson', 'For a personal diary', 'In a friendly message',
    'At a study group', 'For a short story', 'In an office conversation',
    'During a family discussion', 'For a class assignment', 'In a news summary',
    'At a local workshop', 'For a progress report', 'In a phone conversation',
    'During a club activity', 'For an online post', 'In a simple narrative',
)


def prompt_template_signature(prompt):
    value = re.sub(r'\s+', ' ', str(prompt or '')).strip()
    personalized_contexts = tuple(
        wrapper.split('{}', 1)[0].strip().rstrip(',:')
        for wrappers in globals().get('PERSONALIZED_WRAPPERS', {}).values()
        for wrapper in wrappers
        if wrapper != '{}'
    )
    for context in (*FALLBACK_CONTEXTS, *personalized_contexts):
        match = re.match(rf'^{re.escape(context)}\s*[:,]\s*', value, flags=re.IGNORECASE)
        if match:
            value = value[match.end():]
            break
    return fingerprint(normalize_answer(value))


def prompt_similarity(left, right):
    def grams(value):
        words = re.findall(r"[a-z0-9']+", normalize_answer(value))
        return set(zip(words, words[1:])) if len(words) > 1 else set(words)
    first, second = grams(left), grams(right)
    return len(first & second) / max(1, len(first | second))


def is_recent_prompt(prompt, recent_prompts, compare_templates=False):
    for index, recent in enumerate(recent_prompts or []):
        if fingerprint(prompt) == fingerprint(recent) or prompt_similarity(prompt, recent) >= 0.8:
            return True
        # The first SESSION_SIZE prompts are the immediately preceding session.
        # Ignore decorative fallback prefixes so the same underlying exercise
        # cannot return in back-to-back sessions.
        if compare_templates and index < SESSION_SIZE and prompt_template_signature(prompt) == prompt_template_signature(recent):
            return True
    return False


def _varied_prompt(prompt, index):
    replacements = (
        ('Maya', ('Nora', 'Liam', 'Ava', 'Noah')),
        ('She', ('He', 'My colleague', 'The student', 'Our neighbor')),
        ('They', ('We', 'The volunteers', 'The players', 'The guests')),
        ('the report', ('the assignment', 'the message', 'the project', 'the schedule')),
        ('the task', ('the exercise', 'the plan', 'the work', 'the activity')),
    )
    varied = prompt
    for source, choices in replacements:
        if source in varied:
            varied = varied.replace(source, choices[index % len(choices)])
    context = FALLBACK_CONTEXTS[index % len(FALLBACK_CONTEXTS)]
    if varied.lower().startswith(('choose ', 'which ', 'select ')):
        return f'{context}, {varied[0].lower()}{varied[1:]}'
    return f'{context}: {varied}'


def _fallback_candidates(topic):
    bank = list(FALLBACK_BANK[topic])
    random.shuffle(bank)
    return [{
        'exercise_type': exercise_type,
        'category': topic,
        'prompt': prompt,
        'hint': FALLBACK_HINTS.get((topic, normalize_answer(answer)), '') if exercise_type == 'fill_blank' else '',
        'options': options,
        'accepted_answers': [answer],
        'explanation': explanation,
    } for exercise_type, prompt, options, answer, explanation in bank]


def dynamic_fallback_exercises(topic, count, required_types, recent_prompts=None, compare_templates=False):
    topic = topic if topic in FALLBACK_BANK else 'other_grammar'
    result = []
    targets = dict(required_types)
    # Use the whole topic bank and many neutral contexts. Previously this only
    # varied the first few matching templates, so a user with enough history
    # could exhaust the bank and receive an incomplete session.
    candidates = _fallback_candidates(topic)
    context_offset = random.randrange(24)
    for round_index in range(32):
        for item in candidates:
            if targets.get(item['exercise_type'], 0) <= 0:
                continue
            candidate = dict(item)
            candidate['prompt'] = _varied_prompt(item['prompt'], context_offset + round_index)
            if (
                is_recent_prompt(candidate['prompt'], recent_prompts, compare_templates) or
                is_recent_prompt(candidate['prompt'], [x['prompt'] for x in result], compare_templates)
            ):
                continue
            result.append(candidate)
            targets[candidate['exercise_type']] -= 1
            if len(result) >= count:
                return result
    return result


PERSONALIZED_WRAPPERS = {
    'easy': (
        '{}', 'In a short classroom example, {}', 'For a daily conversation, {}',
        'In a simple note, {}', 'During a routine task, {}',
    ),
    'medium': (
        'During the weekly review, {}', 'While the group checked its work, {}',
        'In the update that arrived this morning, {}', 'Before the discussion ended, {}',
        'As part of the project summary, {}',
    ),
    'hard': (
        'Although the schedule had changed unexpectedly, {}',
        'After everyone had reviewed the earlier notes, {}',
        'Because the first explanation was incomplete, {}',
        'While the details in the final report were being checked, {}',
        'Even though several alternatives had been considered, {}',
    ),
}

VERB_SEEDS = (
    ('the editor', 'write', 'wrote', 'written', 'writes', 'writing', 'the summary'),
    ('the guide', 'leave', 'left', 'left', 'leaves', 'leaving', 'the station'),
    ('the team', 'take', 'took', 'taken', 'takes', 'taking', 'the final step'),
    ('the student', 'see', 'saw', 'seen', 'sees', 'seeing', 'the result'),
    ('the manager', 'choose', 'chose', 'chosen', 'chooses', 'choosing', 'a new approach'),
    ('the visitor', 'go', 'went', 'gone', 'goes', 'going', 'to the information desk'),
    ('the researcher', 'make', 'made', 'made', 'makes', 'making', 'a careful comparison'),
    ('the speaker', 'begin', 'began', 'begun', 'begins', 'beginning', 'the presentation'),
    ('the assistant', 'bring', 'brought', 'brought', 'brings', 'bringing', 'the requested file'),
    ('the coordinator', 'send', 'sent', 'sent', 'sends', 'sending', 'the confirmation'),
    ('the analyst', 'find', 'found', 'found', 'finds', 'finding', 'the missing detail'),
    ('the instructor', 'give', 'gave', 'given', 'gives', 'giving', 'clear feedback'),
)

SVA_SEEDS = (
    ('The list of revised items', 'is', ('are', 'were', 'be'), 'on the desk'),
    ('Each of the new participants', 'has', ('have', 'having', 'are'), 'a numbered badge'),
    ('The volunteers from the local center', 'work', ('works', 'working', 'has worked'), 'every Saturday'),
    ('Neither of the proposed answers', 'seems', ('seem', 'seeming', 'have seemed'), 'complete'),
    ('The quality of these materials', 'depends', ('depend', 'depending', 'have depended'), 'on careful storage'),
    ('My colleagues in the design group', 'prefer', ('prefers', 'preferring', 'has preferred'), 'the second version'),
    ('One of the main reasons', 'is', ('are', 'were', 'be'), 'explained below'),
    ('The results of both experiments', 'show', ('shows', 'showing', 'has shown'), 'the same pattern'),
    ('Every student and teacher', 'receives', ('receive', 'receiving', 'have received'), 'a safety guide'),
    ('The information in those reports', 'remains', ('remain', 'remaining', 'have remained'), 'confidential'),
    ('A series of short workshops', 'begins', ('begin', 'beginning', 'have begun'), 'next week'),
    ('The players who joined yesterday', 'are', ('is', 'was', 'be'), 'ready to practise'),
)

ARTICLE_SEEDS = (
    ('an', 'unusual idea', 'was discussed during the meeting'), ('an', 'online account', 'is required for registration'),
    ('a', 'careful editor', 'noticed the inconsistency'), ('a', 'useful reference', 'was added to the guide'),
    ('the', 'final version', 'that you approved is ready'), ('the', 'only entrance', 'to the building is locked'),
    ('an', 'honest response', 'would be more helpful'), ('a', 'European researcher', 'presented the results'),
    ('the', 'equipment in this room', 'must remain here'), ('a', 'one-page summary', 'would be enough'),
    ('an', 'hour-long session', 'starts at noon'), ('the', 'first chapter', 'explains the main concept'),
)

PREPOSITION_SEEDS = (
    ('at', 'The interview begins ___ 10:30 a.m.'), ('on', 'The revised schedule takes effect ___ Monday.'),
    ('in', 'The conference will be held ___ September.'), ('for', 'They have worked on the project ___ three months.'),
    ('since', 'The office has used this system ___ 2024.'), ('between', 'The responsibility is shared ___ the two departments.'),
    ('among', 'The teacher divided the materials ___ five groups.'), ('by', 'The application must be submitted ___ Friday.'),
    ('with', 'The conclusion is consistent ___ the available evidence.'), ('from', 'This approach differs ___ the earlier method.'),
    ('into', 'The report is divided ___ four sections.'), ('during', 'No changes were made ___ the final review.'),
)

PRONOUN_SEEDS = (
    ('me', 'The coordinator sent the revised schedule to Maya and ___.'),
    ('I', 'Jordan and ___ will present the findings tomorrow.'),
    ('themselves', 'The participants organized the event by ___.'),
    ('ourselves', 'My colleague and I completed the final check by ___.'),
    ('her', 'The new responsibility was assigned to ___.'),
    ('they', 'The applicants said that ___ had submitted every document.'),
    ('whom', 'The researcher to ___ you spoke has published the results.'),
    ('whose', 'The student ___ project won the prize thanked the group.'),
    ('it', 'The device stopped working, so the technician inspected ___.'),
    ('their', 'Each team presented ___ solution to the panel.'),
    ('himself', 'The author revised the entire chapter by ___.'),
    ('us', 'The instructor gave ___ additional time to finish.'),
)

WORD_FORM_SEEDS = (
    ('successfully', ('success', 'successful', 'succeed'), 'The team completed the complex task ___.'),
    ('effective', ('effect', 'effectively', 'effectiveness'), 'The revised method proved highly ___.'),
    ('decision', ('decide', 'decisive', 'deciding'), 'Their final ___ surprised the committee.'),
    ('carefully', ('careful', 'care', 'carefulness'), 'The evidence must be examined ___.'),
    ('reliable', ('rely', 'reliably', 'reliability'), 'We need a more ___ source of information.'),
    ('improvement', ('improve', 'improved', 'improving'), 'The latest results show a clear ___.'),
    ('persuasive', ('persuade', 'persuasively', 'persuasion'), 'She presented a highly ___ argument.'),
    ('independently', ('independent', 'independence', 'depend'), 'The two reviewers worked ___.'),
    ('accuracy', ('accurate', 'accurately', 'inaccurate'), 'The editor checked the ___ of every figure.'),
    ('practical', ('practice', 'practically', 'practise'), 'The proposal offers a ___ solution.'),
    ('unexpectedly', ('unexpected', 'expectation', 'expected'), 'The system stopped ___ during the test.'),
    ('clarification', ('clarify', 'clear', 'clearly'), 'The final paragraph needs further ___.'),
)

STRUCTURE_SEEDS = (
    ('although', '___ the evidence was limited, the team reached a careful conclusion.'),
    ('because', 'The meeting was postponed ___ several key members were absent.'),
    ('so that', 'The instructions were rewritten ___ new users could follow them easily.'),
    ('whereas', 'The first method is quick, ___ the second is more accurate.'),
    ('unless', 'The application will not be reviewed ___ every section is complete.'),
    ('while', '___ one group collected the data, another checked the calculations.'),
    ('therefore', 'The sample was too small; ___, the result was treated cautiously.'),
    ('even though', '___ the deadline was close, the editor checked every reference.'),
    ('after', '___ the final draft had been approved, the team prepared the presentation.'),
    ('which', 'The report contains three recommendations, all of ___ require further testing.'),
    ('but', 'The explanation was brief, ___ it addressed the central issue.'),
    ('before', 'Please verify every figure ___ you submit the document.'),
)

ADVANCED_GRAMMAR_SEEDS = {
    'modals': (
        ('must', ('must to', 'have', 'ought'), 'All visitors ___ wear an identification badge.', 'modal verb', '“Must” diễn tả quy định bắt buộc và đi với động từ nguyên mẫu.'),
        ('might have', ('must', 'should be', 'can'), 'The parcel is late; it ___ been sent to the wrong address.', 'modal perfect', '“Might have + V3” diễn tả khả năng đã xảy ra trong quá khứ.'),
        ('should have', ('should', 'had better', 'must be'), 'You ___ checked the figures before publishing the report.', 'modal perfect', '“Should have + V3” diễn tả việc đáng lẽ nên làm trong quá khứ.'),
        ('cannot be', ('must be', 'should have', 'need to'), 'That explanation ___ correct because it contradicts the evidence.', 'modal deduction', '“Cannot be” diễn tả suy luận chắc chắn rằng điều gì đó không đúng.'),
        ('need not', ('must not', 'cannot', 'would not'), 'You ___ print the form; a digital copy is acceptable.', 'modal verb', '“Need not” diễn tả việc không cần thiết phải làm.'),
        ('would', ('will', 'shall', 'must'), 'When we lived nearby, we ___ walk to the library every evening.', 'modal verb', '“Would” có thể diễn tả thói quen lặp lại trong quá khứ.'),
        ('could have', ('can', 'could', 'would have to'), 'With better planning, the team ___ avoided the delay.', 'modal perfect', '“Could have + V3” diễn tả một khả năng trong quá khứ đã không xảy ra.'),
        ('must have', ('should', 'can have', 'would'), 'The lights are off; everyone ___ left the office already.', 'modal deduction', '“Must have + V3” diễn tả suy luận chắc chắn về quá khứ.'),
    ),
    'conditionals': (
        ('will finish', ('finished', 'would finish', 'finish'), 'If the team receives the data today, it ___ the analysis tomorrow.', 'conditional', 'Điều kiện loại một dùng hiện tại đơn ở mệnh đề if và “will” ở mệnh đề chính.'),
        ('would accept', ('will accept', 'accepted', 'accepts'), 'If the terms were clearer, more clients ___ the proposal.', 'conditional', 'Điều kiện loại hai dùng “would + V” cho tình huống giả định hiện tại.'),
        ('would have arrived', ('will arrive', 'would arrive', 'had arrived'), 'If the train had left on time, we ___ before noon.', 'conditional', 'Điều kiện loại ba dùng “would have + V3” ở mệnh đề chính.'),
        ('had known', ('knew', 'would know', 'have known'), 'Had I ___ about the change, I would have updated the schedule.', 'conditional inversion', 'Đảo ngữ điều kiện loại ba dùng “Had + chủ ngữ + V3”.'),
        ('unless', ('if', 'because', 'although'), 'The application will be rejected ___ all required documents are included.', 'conjunction', '“Unless” mang nghĩa “nếu không”, phù hợp với điều kiện trong câu.'),
        ('provided that', ('despite', 'whereas', 'as if'), 'You may work remotely ___ you remain available during office hours.', 'conditional linker', '“Provided that” đặt ra một điều kiện cần được đáp ứng.'),
        ('were', ('was', 'had been', 'would be'), 'If I ___ responsible for the project, I would revise the timeline.', 'conditional', 'Trong câu điều kiện giả định, “were” có thể dùng với mọi chủ ngữ.'),
        ('should', ('would', 'had', 'will'), '___ you need further assistance, contact the support team.', 'conditional inversion', 'Đảo ngữ điều kiện trang trọng có thể bắt đầu bằng “Should”.'),
    ),
    'passive_voice': (
        ('is reviewed', ('reviews', 'is reviewing', 'has review'), 'Every application ___ by two independent assessors.', 'passive voice', 'Bị động hiện tại đơn dùng “is/are + V3”.'),
        ('was postponed', ('postponed', 'was postponing', 'has postponed'), 'The conference ___ because of the severe weather.', 'passive voice', 'Bị động quá khứ đơn dùng “was/were + V3”.'),
        ('has been approved', ('has approved', 'was approving', 'is approve'), 'The revised budget ___ by the board.', 'passive voice', 'Bị động hiện tại hoàn thành dùng “has/have been + V3”.'),
        ('will be announced', ('will announce', 'is announced', 'will be announce'), 'The final results ___ next Monday.', 'passive voice', 'Bị động tương lai dùng “will be + V3”.'),
        ('is being repaired', ('is repairing', 'has repaired', 'was repair'), 'The main elevator ___ at the moment.', 'passive voice', 'Bị động hiện tại tiếp diễn dùng “is/are being + V3”.'),
        ('had been completed', ('had completed', 'was completed', 'has been completing'), 'The inspection ___ before the building reopened.', 'passive voice', 'Bị động quá khứ hoàn thành dùng “had been + V3”.'),
        ('to be submitted', ('to submit', 'submitting', 'to be submit'), 'The documents are expected ___ by Friday.', 'passive infinitive', 'Sau “are expected”, dạng bị động là “to be + V3”.'),
        ('being interrupted', ('interrupting', 'to interrupt', 'been interrupted'), 'The speaker dislikes ___ during a presentation.', 'passive gerund', 'Danh động từ bị động dùng “being + V3”.'),
    ),
    'reported_speech': (
        ('was', ('is', 'has been', 'will be'), 'Lena said that the meeting ___ cancelled.', 'reported speech', 'Trong lời nói gián tiếp ở quá khứ, “is” thường lùi thành “was”.'),
        ('had completed', ('completed', 'has completed', 'would complete'), 'He said that he ___ the task before lunch.', 'reported speech', 'Hành động xảy ra trước lời nói trong quá khứ dùng quá khứ hoàn thành.'),
        ('would send', ('will send', 'sent', 'sends'), 'The manager promised that she ___ the details later.', 'reported speech', 'Trong lời nói gián tiếp, “will” thường lùi thành “would”.'),
        ('whether', ('that', 'what', 'which'), 'They asked ___ the new policy applied to temporary staff.', 'reported question', 'Câu hỏi yes/no gián tiếp dùng “if” hoặc “whether”.'),
        ('not to touch', ('to not touch', 'do not touch', 'not touching'), 'The technician warned us ___ the exposed wires.', 'reported command', 'Mệnh lệnh phủ định gián tiếp dùng “not to + V”.'),
        ('where the archive was', ('where was the archive', 'where is the archive', 'the archive where was'), 'She asked me ___.', 'reported question', 'Câu hỏi gián tiếp dùng trật tự chủ ngữ trước động từ.'),
        ('the following day', ('tomorrow', 'yesterday', 'that day before'), 'In reported speech, “tomorrow” often becomes ___.', 'reported time expression', '“Tomorrow” thường đổi thành “the following day” trong lời nói gián tiếp.'),
        ('had to', ('must', 'will have to', 'has to'), 'He explained that he ___ leave early that afternoon.', 'reported speech', '“Must” chỉ nghĩa bắt buộc thường lùi thành “had to”.'),
    ),
    'gerunds_infinitives': (
        ('reviewing', ('review', 'to reviewing', 'reviewed'), 'The editor suggested ___ the final section again.', 'verb form', 'Sau “suggest” dùng danh động từ V-ing.'),
        ('to submit', ('submitting', 'submit', 'submitted'), 'The applicant agreed ___ an additional reference.', 'verb form', 'Sau “agree” dùng động từ nguyên mẫu có “to”.'),
        ('taking', ('to take', 'take', 'taken'), 'Please avoid ___ personal calls during the meeting.', 'verb form', 'Sau “avoid” dùng danh động từ V-ing.'),
        ('to inform', ('informing', 'inform', 'informed'), 'We regret ___ you that your request was unsuccessful.', 'verb form', '“Regret to inform” dùng khi thông báo một tin không mong muốn.'),
        ('locking', ('to lock', 'lock', 'locked'), 'I remember ___ the door before I left.', 'verb form', '“Remember + V-ing” nói về ký ức của một hành động đã xảy ra.'),
        ('to buy', ('buying', 'buy', 'bought'), 'Remember ___ enough paper on your way to the office.', 'verb form', '“Remember + to V” nói về việc cần làm.'),
        ('working', ('to work', 'work', 'worked'), 'The new procedure means ___ longer hours this week.', 'verb form', '“Mean + V-ing” mang nghĩa kéo theo hoặc bao hàm điều gì.'),
        ('to improve', ('improving', 'improve', 'improved'), 'The department aims ___ response times this year.', 'verb form', 'Sau “aim” dùng động từ nguyên mẫu có “to”.'),
    ),
    'relative_clauses': (
        ('who', ('which', 'whose', 'whom'), 'The researcher ___ presented the findings works at the university.', 'relative pronoun', '“Who” thay cho người và làm chủ ngữ trong mệnh đề quan hệ.'),
        ('whose', ('who', 'which', 'whom'), 'The candidate ___ application impressed the panel was shortlisted.', 'relative pronoun', '“Whose” diễn tả quan hệ sở hữu.'),
        ('which', ('who', 'where', 'what'), 'The software, ___ was updated yesterday, now runs faster.', 'relative pronoun', '“Which” dùng cho vật trong mệnh đề quan hệ không xác định.'),
        ('where', ('which', 'when', 'whose'), 'This is the laboratory ___ the samples are stored.', 'relative adverb', '“Where” thay cho cụm chỉ nơi chốn trong mệnh đề quan hệ.'),
        ('to whom', ('to who', 'who to', 'whose'), 'The consultant ___ I spoke recommended a different approach.', 'object form of “who” after a preposition', 'Sau giới từ trong văn phong trang trọng dùng “whom”.'),
        ('that', ('what', 'where', 'whose'), 'Everything ___ was discussed must remain confidential.', 'relative pronoun', 'Sau “everything”, có thể dùng “that” làm đại từ quan hệ.'),
        ('when', ('where', 'which', 'who'), 'There are occasions ___ a written response is more appropriate.', 'relative adverb', '“When” dùng thay cho cụm chỉ thời gian.'),
        ('most of whom', ('most of which', 'most who', 'whom most'), 'The course has twenty participants, ___ work full-time.', 'relative clause', 'Sau dấu phẩy có thể dùng “most of whom” để nói về người.'),
    ),
    'comparisons': (
        ('more reliable than', ('most reliable than', 'reliabler than', 'as reliable than'), 'The revised method is ___ the original one.', 'comparison', 'Tính từ dài dùng cấu trúc “more + adjective + than”.'),
        ('the most efficient', ('more efficient', 'the efficientest', 'most efficiently'), 'This is ___ solution we have tested so far.', 'superlative', 'So sánh nhất với tính từ dài dùng “the most + adjective”.'),
        ('as accurate as', ('more accurate as', 'so accurate than', 'as accurately than'), 'The preliminary estimate was ___ the final calculation.', 'comparison', 'So sánh bằng dùng “as + adjective + as”.'),
        ('the fewer', ('fewer', 'the fewest', 'less'), 'The more carefully you plan, ___ mistakes you make.', 'parallel comparison', 'Cấu trúc song hành dùng “the more…, the fewer…”.'),
        ('far more complex', ('very complex than', 'far most complex', 'much complex'), 'The legal issue is ___ than we initially expected.', 'comparison', '“Far” có thể nhấn mạnh dạng so sánh hơn “more complex”.'),
        ('less expensive than', ('least expensive than', 'less expensively as', 'not expensive than'), 'The digital subscription is ___ the printed edition.', 'comparison', 'So sánh kém dùng “less + adjective + than”.'),
        ('better and better', ('more and better', 'best and best', 'better than better'), 'The team’s performance is getting ___.', 'progressive comparison', 'Sự thay đổi tăng dần có thể dùng “comparative and comparative”.'),
        ('no sooner', ('not sooner', 'no soon', 'hardly sooner'), '___ had the meeting begun than the fire alarm sounded.', 'inverted comparison', 'Cấu trúc “No sooner… than…” yêu cầu đảo ngữ.'),
    ),
    'quantifiers': (
        ('few', ('little', 'much', 'every'), 'Very ___ applicants had the required experience.', 'quantifier', '“Few” dùng với danh từ đếm được số nhiều.'),
        ('little', ('few', 'many', 'several'), 'There is very ___ evidence to support that conclusion.', 'quantifier', '“Little” dùng với danh từ không đếm được.'),
        ('a number of', ('the number of', 'an amount of', 'much'), '___ employees have requested flexible hours.', 'quantifier', '“A number of” đi với danh từ đếm được số nhiều và động từ số nhiều.'),
        ('the number of', ('a number of', 'many of', 'several'), '___ complaints has fallen this month.', 'quantifier', '“The number of” là cụm chủ ngữ số ít.'),
        ('neither', ('both', 'all', 'every'), '___ of the two proposals meets all the requirements.', 'quantifier', '“Neither of” mang nghĩa không phương án nào trong hai phương án.'),
        ('each', ('many', 'few', 'much'), 'The reviewer examined ___ of the documents separately.', 'quantifier', '“Each of” nhấn mạnh từng thành viên riêng lẻ trong một nhóm.'),
        ('a great deal of', ('many', 'a few', 'several'), 'The investigation required ___ patience.', 'quantifier', '“A great deal of” dùng với danh từ không đếm được.'),
        ('fewer', ('less', 'little', 'much'), 'The updated form contains ___ questions than the old one.', 'quantifier', '“Fewer” dùng với danh từ đếm được số nhiều.'),
    ),
}

OTHER_SEEDS = (
    ('review', ('reviews', 'reviewed', 'to review'), 'You should ___ the instructions before starting.'),
    ('working', ('work', 'to work', 'worked'), 'The team enjoys ___ on practical problems.'),
    ('are', ('is', 'be', 'has'), 'There ___ several reasons to revise the proposal.'),
    ('to complete', ('complete', 'completing', 'completed'), 'The group expects ___ the work tomorrow.'),
    ('used to', ('use to', 'was use to', 'using to'), 'The office ___ keep paper records.'),
    ('be checked', ('check', 'checking', 'checked'), 'Every reference must ___ before publication.'),
    ('had better', ('would better', 'has better', 'better to'), 'You ___ save a copy before closing the file.'),
    ('reading', ('read', 'to reading', 'reads'), 'She suggested ___ the instructions again.'),
    ('are being', ('is being', 'were', 'have'), 'The final documents ___ reviewed right now.'),
    ('could have', ('could', 'can have', 'has'), 'They ___ avoided the delay by checking earlier.'),
    ('not only', ('neither', 'either', 'both'), 'The update is ___ faster but also more reliable.'),
    ('whether', ('if or', 'that whether', 'which'), 'We need to decide ___ the change is necessary.'),
)


def _four_options(answer, distractors):
    values = [answer, *distractors]
    result = []
    for value in values:
        if normalize_answer(value) not in {normalize_answer(item) for item in result}:
            result.append(value)
    fillers = ('does', 'did', 'being', 'to do')
    for filler in fillers:
        if len(result) >= 4:
            break
        if normalize_answer(filler) not in {normalize_answer(item) for item in result}:
            result.append(filler)
    return result[:4]


def _wrap_personalized_prompt(prompt, difficulty, index):
    wrapper = PERSONALIZED_WRAPPERS[difficulty][index % len(PERSONALIZED_WRAPPERS[difficulty])]
    if wrapper == '{}':
        return prompt
    lowered = prompt[0].lower() + prompt[1:] if prompt else prompt
    return wrapper.format(lowered)


def _personalized_seed_item(concept, exercise_type, difficulty, index, decorate=True):
    topic = CONCEPT_TOPICS.get(concept, concept if concept in TOPICS else 'other_grammar')
    if topic == 'verb_tense':
        subject, base, past, participle, third, gerund, complement = VERB_SEEDS[index % len(VERB_SEEDS)]
        if concept == 'present_perfect':
            answer, prompt, distractors = participle, f'{subject.capitalize()} has ___ {complement}.', (base, past, gerund)
            explanation = f'Sau “has” cần quá khứ phân từ; “{participle}” là V3 của “{base}”.'
        elif concept == 'present_continuous':
            answer, prompt, distractors = gerund, f'{subject.capitalize()} is ___ {complement} now.', (base, past, third)
            explanation = f'Hiện tại tiếp diễn dùng “is + V-ing”, nên chọn “{gerund}”.'
        elif concept == 'past_continuous':
            answer, prompt, distractors = gerund, f'{subject.capitalize()} was ___ {complement} when the call arrived.', (base, past, participle)
            explanation = f'Quá khứ tiếp diễn dùng “was + V-ing”, nên chọn “{gerund}”.'
        elif concept == 'future_forms':
            answer, prompt, distractors = base, f'{subject.capitalize()} will ___ {complement} tomorrow.', (past, third, gerund)
            explanation = f'Sau “will” dùng động từ nguyên mẫu “{base}”.'
        elif concept == 'present_simple':
            answer, prompt, distractors = third, f'{subject.capitalize()} usually ___ {complement}.', (base, past, gerund)
            explanation = f'Chủ ngữ số ít trong hiện tại đơn dùng “{third}”.'
        else:
            answer, prompt, distractors = past, f'Last week, {subject} ___ {complement}.', (base, third, gerund)
            explanation = f'“Last week” yêu cầu quá khứ đơn; dạng đúng là “{past}”.'
        hint = base
        seed_id = f'verb-{concept}-{index % len(VERB_SEEDS)}'
    elif topic == 'subject_verb_agreement':
        subject, answer, distractors, complement = SVA_SEEDS[index % len(SVA_SEEDS)]
        prompt, hint = f'{subject} ___ {complement}.', infinitive_hint(answer)
        explanation = f'Dạng “{answer}” hòa hợp với chủ ngữ “{subject}”.'
        seed_id = f'sva-{index % len(SVA_SEEDS)}'
    elif topic == 'articles':
        answer, noun, complement = ARTICLE_SEEDS[index % len(ARTICLE_SEEDS)]
        prompt, hint, distractors = f'___ {noun.capitalize()} {complement}.', 'article', ('a', 'an', 'the', 'no article')
        explanation = f'Dùng “{answer}” trước cụm danh từ “{noun}” trong ngữ cảnh này.'
        seed_id = f'article-{index % len(ARTICLE_SEEDS)}'
    elif topic == 'prepositions':
        answer, prompt = PREPOSITION_SEEDS[index % len(PREPOSITION_SEEDS)]
        hint, distractors = 'preposition', ('at', 'on', 'in', 'for', 'since', 'by')
        explanation = f'Giới từ phù hợp với cụm từ trong câu là “{answer}”.'
        seed_id = f'prep-{index % len(PREPOSITION_SEEDS)}'
    elif topic == 'pronouns':
        answer, prompt = PRONOUN_SEEDS[index % len(PRONOUN_SEEDS)]
        hint = (
            'object form of “who” after a preposition'
            if prompt.startswith('The researcher to ___') else 'pronoun role'
        )
        distractors = ('I', 'me', 'my', 'mine', 'they', 'them', 'their', 'themselves')
        explanation = f'Vị trí và vai trò trong câu yêu cầu đại từ “{answer}”.'
        seed_id = f'pronoun-{index % len(PRONOUN_SEEDS)}'
    elif topic == 'word_form':
        answer, distractors, prompt = WORD_FORM_SEEDS[index % len(WORD_FORM_SEEDS)]
        hint = re.sub(r'(ly|ness|ment|ive|al|ful)$', '', answer) or answer
        explanation = f'Cấu trúc câu yêu cầu đúng từ loại của “{answer}”.'
        seed_id = f'word-form-{index % len(WORD_FORM_SEEDS)}'
    elif topic == 'sentence_structure':
        answer, prompt = STRUCTURE_SEEDS[index % len(STRUCTURE_SEEDS)]
        hint, distractors = 'conjunction', ('and', 'but', 'because', 'although', 'therefore', 'unless')
        explanation = f'Quan hệ giữa các mệnh đề yêu cầu từ nối “{answer}”.'
        seed_id = f'structure-{index % len(STRUCTURE_SEEDS)}'
    elif topic in ADVANCED_GRAMMAR_SEEDS:
        seeds = ADVANCED_GRAMMAR_SEEDS[topic]
        answer, distractors, prompt, hint, explanation = seeds[index % len(seeds)]
        seed_id = f'{topic}-{index % len(seeds)}'
    else:
        answer, distractors, prompt = OTHER_SEEDS[index % len(OTHER_SEEDS)]
        hint = answer.split()[-1] if len(answer.split()) == 1 else answer
        explanation = f'Cấu trúc ngữ pháp trong câu yêu cầu “{answer}”.'
        seed_id = f'other-{index % len(OTHER_SEEDS)}'

    if decorate:
        prompt = _wrap_personalized_prompt(prompt, difficulty, index // 2)
    options = [] if exercise_type == 'fill_blank' else _four_options(answer, distractors)
    return {
        'exercise_type': exercise_type,
        'category': topic,
        'concept_code': concept,
        'difficulty': difficulty,
        'prompt': prompt,
        'hint': hint if exercise_type == 'fill_blank' else '',
        'options': options,
        'accepted_answers': [answer],
        'explanation': explanation,
        # Deliberately exclude the decorative wrapper. Changing only a prefix
        # must never turn the same grammar frame into a new exercise.
        'generation_key': fingerprint(concept, difficulty, exercise_type, seed_id),
    }


MISTAKE_VERB_DISTRACTORS = {
    'brought': ('bring', 'brings', 'bringing'),
    'had': ('have', 'has', 'having'),
    'forgotten': ('forget', 'forgot', 'forgetting'),
    'begun': ('begin', 'began', 'beginning'),
    'seen': ('see', 'saw', 'seeing'),
    'left': ('leave', 'leaves', 'leaving'),
    'finished': ('finish', 'finishes', 'finishing'),
    'gone': ('go', 'went', 'going'),
    'spent': ('spend', 'spends', 'spending'),
    'travel': ('travels', 'traveled', 'traveling'),
}


def _mistake_distractors(mistake, original, correction):
    concept = str(getattr(mistake, 'concept_code', '') or '')
    category = str(getattr(mistake, 'category', '') or '')
    pools = {
        'subject_verb_agreement': ('has', 'have', 'is', 'are', 'does', 'do', "hasn't", "haven't"),
        'prepositions': ('for', 'since', 'at', 'on', 'in', 'by', 'during'),
        'articles': ('a', 'an', 'the', 'no article'),
        'pronouns': ('I', 'me', 'my', 'mine', 'who', 'whom', 'whose'),
        'word_form': ('travel', 'travels', 'traveled', 'traveling'),
    }
    candidates = [original]
    candidates.extend(MISTAKE_VERB_DISTRACTORS.get(normalize_answer(correction), ()))
    candidates.extend(pools.get(concept, pools.get(category, ())))
    candidates.extend(('does', 'did', 'being', 'to do'))
    return tuple(
        value for value in candidates
        if normalize_answer(value) != normalize_answer(correction)
    )


def _mistake_specific_hint(mistake, blanked_context):
    concept = str(getattr(mistake, 'concept_code', '') or '')
    before_blank = blanked_context.split('___', 1)[0].casefold().rstrip()
    if re.search(r"\b(has|have|had|hasn't|haven't|hadn't)$", before_blank):
        return 'past participle after have, has, or had'
    if re.search(r"\b(did|didn't)$", before_blank):
        return 'base verb after did or did not'
    return {
        'subject_verb_agreement': 'match the auxiliary or verb with the true subject',
        'prepositions': 'distinguish a duration from a starting point or fixed time expression',
        'articles': 'check specificity and the opening sound of the following noun',
        'pronouns': 'identify whether the blank is a subject, object, possessive, or reflexive pronoun',
        'word_form': 'identify the part of speech required by the surrounding words',
        'sentence_structure': 'check how the two clauses are logically connected',
        'present_perfect': 'use have or has plus the past participle',
        'past_simple': 'check whether the surrounding auxiliary requires a base form or past participle',
        'verb_tense': 'use the time relationship between the clauses to choose the verb form',
        'other_grammar': 'use the grammar pattern immediately around the blank',
    }.get(concept, 'use the correction pattern from your Editor mistake')


def mistake_anchored_exercise(mistake, exercise_type, difficulty):
    context = str(getattr(mistake, 'sentence_context', '') or '').strip()
    original = str(getattr(mistake, 'original', '') or '').strip()
    correction = str(getattr(mistake, 'correction', '') or '').strip()
    if not context or not original or not correction:
        return None
    match = re.search(re.escape(original), context, flags=re.IGNORECASE)
    if not match:
        return None
    blanked = f'{context[:match.start()]}___{context[match.end():]}'
    if exercise_type == 'multiple_choice':
        prompt = f'Which form correctly replaces “{original}” in this sentence? {context}'
        options = _four_options(correction, _mistake_distractors(mistake, original, correction))
        hint = ''
    else:
        prompt = f'Correct the sentence from your Editor: {blanked}'
        options = []
        hint = _mistake_specific_hint(mistake, blanked)
    explanation = str(getattr(mistake, 'explanation', '') or '').strip()
    if not explanation:
        explanation = f'Trong câu này cần dùng “{correction}” thay cho “{original}”.'
    mistake_key = str(getattr(mistake, 'fingerprint', '') or fingerprint(
        getattr(mistake, 'concept_code', ''), context, original, correction,
    ))
    return {
        'exercise_type': exercise_type,
        'category': CONCEPT_TOPICS.get(
            getattr(mistake, 'concept_code', ''),
            normalize_topic(getattr(mistake, 'category', '')),
        ),
        'concept_code': getattr(mistake, 'concept_code', ''),
        'difficulty': difficulty,
        'prompt': prompt,
        'hint': hint,
        'options': options,
        'accepted_answers': [correction],
        'explanation': explanation,
        'generation_key': fingerprint('editor-mistake', mistake_key, difficulty, exercise_type),
        '_mistake': mistake,
        '_anchored': True,
    }


def personalized_fallback_exercises(mistake, slots, recent_prompts=None, recent_generation_keys=None):
    mistakes = list(mistake) if isinstance(mistake, (list, tuple)) else [mistake]
    mistakes = [item for item in mistakes if item is not None]
    if not mistakes:
        return []
    recent_generation_keys = set(recent_generation_keys or [])
    result = []
    for slot_index, slot in enumerate(slots):
        found = None
        for mistake_shift in range(len(mistakes)):
            source_mistake = mistakes[(slot_index + mistake_shift) % len(mistakes)]
            candidate = mistake_anchored_exercise(
                source_mistake,
                slot['exercise_type'],
                slot['difficulty'],
            )
            if not candidate or candidate['generation_key'] in recent_generation_keys:
                continue
            if is_recent_prompt(candidate['prompt'], [*(recent_prompts or []), *[item['prompt'] for item in result]]):
                continue
            found = candidate
            break
        for attempt in range(160):
            if found:
                break
            source_mistake = mistakes[(slot_index + attempt) % len(mistakes)]
            candidate = _personalized_seed_item(
                source_mistake.concept_code,
                slot['exercise_type'],
                slot['difficulty'],
                slot_index + attempt,
            )
            candidate['_mistake'] = source_mistake
            if candidate['generation_key'] in recent_generation_keys:
                continue
            if is_recent_prompt(candidate['prompt'], [*(recent_prompts or []), *[item['prompt'] for item in result]]):
                continue
            found = candidate
            break
        if found:
            result.append(found)
            recent_generation_keys.add(found['generation_key'])
    return result


GENERAL_GRAMMAR_CURRICULUM = {
    'easy': (
        'present_simple', 'articles', 'prepositions', 'subject_verb_agreement',
        'pronouns', 'comparisons', 'quantifiers', 'modals', 'past_simple',
        'gerunds_infinitives', 'word_form', 'sentence_structure',
    ),
    'medium': (
        'present_perfect', 'passive_voice', 'articles', 'conditionals',
        'prepositions', 'reported_speech', 'subject_verb_agreement', 'relative_clauses',
        'past_continuous', 'gerunds_infinitives', 'future_forms', 'modals',
        'word_form', 'comparisons', 'sentence_structure', 'quantifiers',
    ),
    'hard': (
        'conditionals', 'reported_speech', 'passive_voice', 'relative_clauses',
        'modals', 'gerunds_infinitives', 'sentence_structure', 'word_form',
        'present_perfect', 'past_continuous', 'comparisons', 'quantifiers',
        'subject_verb_agreement', 'articles', 'prepositions', 'pronouns',
    ),
}


def general_fallback_exercises(difficulty, count, required_types, recent_prompts=None):
    difficulty = difficulty if difficulty in GENERAL_GRAMMAR_CURRICULUM else 'easy'
    curriculum = list(GENERAL_GRAMMAR_CURRICULUM[difficulty])
    targets = dict(required_types)
    result = []
    requested_types = [
        exercise_type
        for exercise_type, amount in targets.items()
        for _ in range(amount)
    ][:count]
    random.shuffle(requested_types)
    curriculum_offset = random.randrange(len(curriculum))
    seed_offset = random.randrange(96)
    used_keys = set()

    # Prefer a different grammar concept for every slot. Only move to another
    # concept when all of its unused sentence frames conflict with recent work.
    for slot_index, exercise_type in enumerate(requested_types):
        found = None
        for concept_shift in range(len(curriculum)):
            concept = curriculum[(curriculum_offset + slot_index + concept_shift) % len(curriculum)]
            for seed_attempt in range(24):
                candidate = _personalized_seed_item(
                    concept,
                    exercise_type,
                    difficulty,
                    seed_offset + slot_index + seed_attempt,
                    decorate=False,
                )
                if candidate['generation_key'] in used_keys:
                    continue
                compared_prompts = [*(recent_prompts or []), *[item['prompt'] for item in result]]
                if is_recent_prompt(candidate['prompt'], compared_prompts, compare_templates=True):
                    continue
                found = candidate
                break
            if found:
                break
        if found:
            result.append(found)
            used_keys.add(found['generation_key'])
            targets[exercise_type] -= 1

    # Defensive compatibility fallback for an exceptionally saturated history.
    for topic in list(FALLBACK_BANK):
        if len(result) >= count:
            break
        candidates = dynamic_fallback_exercises(
            topic,
            count - len(result),
            targets,
            [*(recent_prompts or []), *[item['prompt'] for item in result]],
            compare_templates=True,
        )
        for item in candidates:
            if targets.get(item['exercise_type'], 0) <= 0:
                continue
            item['difficulty'] = difficulty
            item['generation_key'] = fingerprint(
                topic, difficulty, item['exercise_type'], prompt_template_signature(item['prompt'])
            )
            result.append(item)
            targets[item['exercise_type']] -= 1
            if len(result) >= count:
                break
    return result[:count]
