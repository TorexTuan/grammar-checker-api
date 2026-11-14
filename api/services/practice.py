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
required_types mix from the request. Do not copy personal names,
places, employers, contact details, or distinctive facts from the source sentence;
create analogous neutral examples instead.

Each item must have:
{"exercise_type":"multiple_choice|fill_blank","category":"allowed category code",
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


def validate_generated_exercise(item, requested_topic):
    if not isinstance(item, dict):
        return None
    exercise_type = item.get('exercise_type')
    if exercise_type not in TYPE_TARGETS:
        return None
    category = normalize_topic(item.get('category') or requested_topic)
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


def generate_with_gemini(mistakes, topic, count=SESSION_SIZE, mode='personalized', difficulty='medium', recent_prompts=None, required_types=None):
    if cache.get(BREAKER_OPEN_KEY):
        return [], None
    request_payload = {
        'count': count,
        'required_types': dict(required_types or TYPE_TARGETS),
        'difficulty': difficulty,
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
            for mistake in mistakes[:5]
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
                exercise = validate_generated_exercise(item, topic)
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


def prompt_similarity(left, right):
    def grams(value):
        words = re.findall(r"[a-z0-9']+", normalize_answer(value))
        return set(zip(words, words[1:])) if len(words) > 1 else set(words)
    first, second = grams(left), grams(right)
    return len(first & second) / max(1, len(first | second))


def is_recent_prompt(prompt, recent_prompts):
    return any(
        fingerprint(prompt) == fingerprint(recent) or prompt_similarity(prompt, recent) >= 0.8
        for recent in (recent_prompts or [])
    )


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
    contexts = (
        'In a classroom exercise', 'For a travel journal', 'In a short email',
        'During a team meeting', 'For a daily conversation', 'In a school project',
        'At a community event', 'For a weekend plan', 'In a work update',
        'During an English lesson', 'For a personal diary', 'In a friendly message',
        'At a study group', 'For a short story', 'In an office conversation',
        'During a family discussion', 'For a class assignment', 'In a news summary',
        'At a local workshop', 'For a progress report', 'In a phone conversation',
        'During a club activity', 'For an online post', 'In a simple narrative',
    )
    context = contexts[index % len(contexts)]
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


def dynamic_fallback_exercises(topic, count, required_types, recent_prompts=None):
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
            if is_recent_prompt(candidate['prompt'], recent_prompts) or is_recent_prompt(candidate['prompt'], [x['prompt'] for x in result]):
                continue
            result.append(candidate)
            targets[candidate['exercise_type']] -= 1
            if len(result) >= count:
                return result
    return result


def personalized_fallback_exercises(mistake, count, required_types, recent_prompts=None):
    # The fine-grained concept is more reliable than the broad provider
    # category (for example, a verb-tense issue may arrive as other_grammar).
    topic = CONCEPT_TOPICS.get(mistake.concept_code)
    if not topic:
        topic = mistake.category if mistake.category in FALLBACK_BANK else normalize_topic(mistake.concept_code)
    if topic not in FALLBACK_BANK:
        topic = 'other_grammar'
    exercises = dynamic_fallback_exercises(topic, count, required_types, recent_prompts)
    for item in exercises:
        item['category'] = topic
        item['concept_code'] = mistake.concept_code
    return exercises


def general_fallback_exercises(difficulty, count, required_types, recent_prompts=None):
    pools = {
        'easy': ['articles', 'prepositions', 'pronouns', 'subject_verb_agreement'],
        'medium': ['verb_tense', 'word_form', 'sentence_structure', 'other_grammar'],
        'hard': ['sentence_structure', 'word_form', 'verb_tense', 'other_grammar'],
    }
    topics = list(pools.get(difficulty, pools['easy']))
    random.shuffle(topics)
    targets = dict(required_types)
    result = []
    for topic in topics * 3:
        needed = sum(targets.values())
        if not needed:
            break
        candidates = dynamic_fallback_exercises(topic, needed, targets, [*(recent_prompts or []), *[x['prompt'] for x in result]])
        for item in candidates:
            if targets.get(item['exercise_type'], 0) <= 0:
                continue
            item['difficulty'] = difficulty
            result.append(item)
            targets[item['exercise_type']] -= 1
    return result[:count]
