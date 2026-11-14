import json
import os
import re

from google import genai
from google.genai import types


GEMINI_MODEL = 'gemini-3.8-flash'
GEMINI_FALLBACK_MODEL = 'gemini-3.5-flash-lite'
GEMINI_REQUEST_TIMEOUT_MS = 11000

IRREGULAR_VERB_FORMS = {
    'been': ('be', 'quá khứ phân từ (V3)'),
    'begun': ('begin', 'quá khứ phân từ (V3)'),
    'broken': ('break', 'quá khứ phân từ (V3)'),
    'brought': ('bring', 'quá khứ đơn (V2) hoặc quá khứ phân từ (V3)'),
    'chosen': ('choose', 'quá khứ phân từ (V3)'),
    'done': ('do', 'quá khứ phân từ (V3)'),
    'driven': ('drive', 'quá khứ phân từ (V3)'),
    'eaten': ('eat', 'quá khứ phân từ (V3)'),
    'forgot': ('forget', 'quá khứ đơn (V2)'),
    'forgotten': ('forget', 'quá khứ phân từ (V3)'),
    'gone': ('go', 'quá khứ phân từ (V3)'),
    'had': ('have', 'quá khứ đơn (V2) hoặc quá khứ phân từ (V3)'),
    'known': ('know', 'quá khứ phân từ (V3)'),
    'left': ('leave', 'quá khứ đơn (V2) hoặc quá khứ phân từ (V3)'),
    'made': ('make', 'quá khứ đơn (V2) hoặc quá khứ phân từ (V3)'),
    'seen': ('see', 'quá khứ phân từ (V3)'),
    'spoken': ('speak', 'quá khứ phân từ (V3)'),
    'taken': ('take', 'quá khứ phân từ (V3)'),
    'written': ('write', 'quá khứ phân từ (V3)'),
}

GENERIC_EXPLANATION_MARKERS = (
    'dạng động từ phải phù hợp với chủ ngữ và cấu trúc câu',
    'động từ phải hòa hợp với chủ ngữ về số',
    'động từ phải phù hợp với mốc thời gian',
    'phù hợp hơn với thời điểm của hành động trong câu',
    'cách sửa này giúp cấu trúc câu đúng ngữ pháp hơn',
    'v2 hoặc v3',
    'quá khứ đơn (v2) hoặc quá khứ phân từ (v3)',
)


def get_gemini_client():
    api_key = os.getenv('GEMINI_API_KEY', '').strip()
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=GEMINI_REQUEST_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    ) if api_key else None


SYSTEM_PROMPT = """
Bạn là giáo viên tiếng Anh. Với mỗi lỗi đã được hệ thống khác xác định, hãy giải thích
ngắn gọn bằng tiếng Việt vì sao cụm từ gốc chưa phù hợp và đề xuất sửa đã cho là đúng
trong ngữ cảnh câu. Không tự tìm thêm lỗi, không thay đổi mã định danh hoặc nội dung sửa.
Với lỗi về thì, chỉ ra dấu hiệu thời gian có trong sentence_context và gọi đúng tên thì
(ví dụ: quá khứ đơn). Với lỗi hòa hợp chủ ngữ - động từ, nêu rõ chủ ngữ nào là số ít
hay số nhiều và vì sao chọn dạng động từ đó. Nếu một đề xuất sửa gồm nhiều thay đổi,
giải thích từng thay đổi quan trọng. Tránh những câu chung chung như “động từ phải phù
hợp với mốc thời gian” hoặc “động từ phải hòa hợp với chủ ngữ”.
Trả về duy nhất JSON dạng [{"id":"...", "explanation":"..."}] theo đúng thứ tự đầu vào.
Nếu không có dấu hiệu rõ ràng trong ngữ cảnh, hãy giải thích chính xác dạng từ được đề
xuất nhưng không tự đặt ra một mốc thời gian hoặc chủ ngữ không xuất hiện trong câu.
Also return a structured concept_code for each item. Allowed values: past_simple,
past_continuous, present_simple, present_continuous, present_perfect, future_forms,
verb_tense, subject_verb_agreement, articles, prepositions, pronouns, word_form,
sentence_structure, other_grammar. Keep the existing id and explanation fields.
"""


def _sentence_context(text, offset):
    if not text or not isinstance(offset, int):
        return '', ''
    offset = max(0, min(offset, len(text)))
    start = max(text.rfind(mark, 0, offset) for mark in '.!?\n') + 1
    ends = [text.find(mark, offset) for mark in '.!?\n']
    end = min((position for position in ends if position >= 0), default=len(text))
    return text[start:end], text[start:offset]


def _past_time_cue(sentence):
    pattern = (
        r'\b(?:yesterday|the day before|previously|earlier|'
        r'last\s+(?:night|week|month|year|[A-Za-z]+day)|'
        r'\d+\s+(?:days?|weeks?|months?|years?)\s+ago|'
        r'in\s+(?:19|20)\d{2})\b'
    )
    match = re.search(pattern, sentence, flags=re.IGNORECASE)
    return match.group(0) if match else None


def _normalized_word(value):
    return re.sub(r"[^a-z'’-]", '', str(value or '').casefold()).replace('’', "'")


def _preceding_perfect_auxiliary(prefix):
    match = re.search(r"\b(has|have|had)(?:\s+not|\s+n['’]t)?\s*$", prefix, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _subject_before(prefix):
    value = re.sub(r'\s+', ' ', prefix).strip()
    if not value:
        return '', ''
    clause = re.split(r'[,;:!?]|\b(?:but|because|although|while)\b', value, flags=re.IGNORECASE)[-1].strip()
    compound = re.search(
        r"([A-Za-z][\w'’-]*(?:\s+[A-Za-z][\w'’-]*){0,2}\s+and\s+"
        r"[A-Za-z][\w'’-]*(?:\s+[A-Za-z][\w'’-]*){0,2})$",
        clause,
    )
    if compound:
        return compound.group(1), 'số nhiều'
    pronoun = re.search(r'\b(I|you|we|they|he|she|it)\s*$', clause, flags=re.IGNORECASE)
    if pronoun:
        display = pronoun.group(1)
        normalized = display.casefold()
        if normalized in {'he', 'she', 'it'}:
            return display, 'ngôi thứ ba số ít'
        if normalized in {'we', 'they'}:
            return display, 'số nhiều'
        if normalized == 'you':
            return display, 'chủ ngữ “you”'
        return display, 'chủ ngữ “I”'
    noun_phrase = re.search(
        r"\b((?:the|this|that|my|your|his|her|our|their)\s+[A-Za-z][\w'’-]*"
        r"(?:\s+(?:of\s+)?[A-Za-z][\w'’-]*){0,3})$",
        clause,
        flags=re.IGNORECASE,
    )
    if noun_phrase:
        display = noun_phrase.group(1)
        last_word = re.findall(r"[A-Za-z][\w'’-]*", display)[-1].casefold()
        number = 'số nhiều' if last_word.endswith('s') and not last_word.endswith('ss') else 'ngôi thứ ba số ít'
        return display, number
    proper_name = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})$', clause)
    if proper_name:
        return proper_name.group(1), 'ngôi thứ ba số ít'
    return '', ''


def _subject_explanation_lead(subject, subject_number):
    if not subject or not subject_number:
        return ''
    if subject_number.startswith('chủ ngữ'):
        return f'Với chủ ngữ “{subject}”, '
    return f'Chủ ngữ “{subject}” là {subject_number}, nên '


def _verb_form(value):
    normalized = _normalized_word(value)
    if normalized in IRREGULAR_VERB_FORMS:
        base, label = IRREGULAR_VERB_FORMS[normalized]
        return {'base': base, 'label': label}
    if normalized.endswith('ed') and len(normalized) > 3:
        return {'base': '', 'label': 'quá khứ đơn (V2) hoặc quá khứ phân từ (V3)'}
    return None


def _temporal_complement(issue, text=''):
    if not isinstance(text, str) or not text:
        return '', ''
    offset = issue.get('offset', 0)
    if not isinstance(offset, int):
        return '', ''
    original = str(issue.get('original') or '')
    suffix_start = max(0, min(offset + len(original), len(text)))
    suffix = re.split(r'[.!?\n]', text[suffix_start:], maxsplit=1)[0][:100]
    duration_pattern = (
        r'\b(?:a\s+long\s+time|ages|'
        r'(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|several|many|few|\d+)\s+'
        r'(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?))\b'
    )
    point_pattern = (
        r'\b(?:(?:19|20)\d{2}|'
        r'Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|'
        r'January|February|March|April|May|June|July|August|September|October|November|December|'
        r'yesterday|last\s+(?:night|week|month|year))\b'
    )
    duration = re.search(duration_pattern, suffix, flags=re.IGNORECASE)
    if duration:
        return duration.group(0), 'duration'
    point = re.search(point_pattern, suffix, flags=re.IGNORECASE)
    if point:
        return point.group(0), 'start_point'
    return '', ''


def _grammar_facts(issue, text=''):
    sentence, prefix = _sentence_context(text, issue.get('offset', 0))
    subject, subject_number = _subject_before(prefix)
    temporal_expression, temporal_kind = _temporal_complement(issue, text)
    return {
        'sentence_context': sentence.strip(),
        'time_cue': _past_time_cue(sentence) or '',
        'preceding_perfect_auxiliary': _preceding_perfect_auxiliary(prefix) or '',
        'subject': subject,
        'subject_number': subject_number,
        'original_form': _verb_form(issue.get('original')) or {},
        'replacement_form': _verb_form(issue.get('replacement')) or {},
        'temporal_expression': temporal_expression,
        'temporal_kind': temporal_kind,
    }


def _specific_explanation(explanation, issue, text=''):
    normalized = re.sub(r'\s+', ' ', str(explanation or '')).strip().casefold()
    if len(normalized) < 25 or any(marker in normalized for marker in GENERIC_EXPLANATION_MARKERS):
        return False
    replacement = str(issue.get('replacement') or '').strip().casefold().replace('’', "'")
    normalized_apostrophes = normalized.replace('’', "'")
    if replacement and replacement not in normalized_apostrophes:
        return False

    original_word = _normalized_word(issue.get('original'))
    replacement_word = _normalized_word(issue.get('replacement'))
    facts = _grammar_facts(issue, text)
    if {original_word, replacement_word} == {'has', 'have'}:
        subject = facts['subject'].casefold()
        if subject and subject not in normalized:
            return False
        if not any(marker in normalized for marker in ('số nhiều', 'số ít', 'ngôi thứ ba', 'i”, “you”, “we”, “they')):
            return False
    if facts['preceding_perfect_auxiliary'] and _verb_form(issue.get('replacement')):
        auxiliary = facts['preceding_perfect_auxiliary'].casefold()
        if auxiliary not in normalized or not any(marker in normalized for marker in ('v3', 'quá khứ phân từ')):
            return False
    if {original_word, replacement_word} == {'since', 'for'}:
        if 'khoảng thời gian' not in normalized or 'mốc' not in normalized:
            return False
        expression = facts['temporal_expression'].casefold()
        if expression and expression not in normalized:
            return False
    return True


def _fallback_explanation(issue, text=''):
    code = (issue.get('category') or '').lower()
    original = issue.get('original', '')
    replacement = issue.get('replacement', '')
    change = (
        f'Thay “{original}” bằng “{replacement}”. '
        if original and replacement else
        f'Bỏ “{original}”. ' if original else
        f'Thêm “{replacement}”. ' if replacement else ''
    )
    sentence, prefix = _sentence_context(text, issue.get('offset', 0))
    prefix = prefix.lower()
    if original.lower().startswith('me and ') and ' and i ' in replacement.lower():
        return (
            change + 'Cụm này làm chủ ngữ nên dùng “I” thay cho “me” và đặt người còn lại trước. '
            'Chủ ngữ gồm hai người nên động từ hiện tại đơn dùng dạng không thêm “s”.'
        )
    if 'spelling' in code or 'spell' in code:
        return change + 'Đây là cách viết đúng chính tả.'
    if 'punct' in code:
        return change + 'Dấu câu cần phù hợp với cấu trúc và ý nghĩa của câu.'
    subject = re.search(
        r'\b(we|they|you)\b(?:\s+(?:all|both|often|always|usually|sometimes|also|still|currently|just|now))*\s*$',
        prefix,
    )
    if original.lower() == 'is' and replacement.lower() == 'are' and subject:
        pronoun = subject.group(1)
        if pronoun == 'you':
            return change + 'Với chủ ngữ “you”, động từ “to be” ở hiện tại dùng “are”, không dùng “is”.'
        return change + f'Chủ ngữ “{pronoun}” là số nhiều, nên động từ “to be” ở hiện tại dùng “are”, không dùng “is”.'
    if 'tense' in code:
        cue = _past_time_cue(sentence)
        if cue:
            return change + f'Dấu hiệu “{cue}” cho biết hành động xảy ra trong quá khứ, nên động từ cần chia ở thì quá khứ đơn.'
        if original.lower() == 'gives' and replacement.lower() == 'gave':
            return change + '“Gave” là dạng quá khứ đơn của “give”; “gives” là dạng hiện tại đơn dùng với chủ ngữ số ít.'
        return change + 'Dạng động từ được đề xuất phù hợp hơn với thời điểm của hành động trong câu.'
    if 'sva' in code or 'agreement' in code:
        if ' and ' in prefix[-50:] and original.endswith('s') and not replacement.endswith('s'):
            return change + 'Chủ ngữ có hai thành phần nối bằng “and”, nên động từ ở hiện tại đơn dùng dạng không thêm “s”.'
        return change + 'Động từ phải hòa hợp với chủ ngữ về số.'
    if 'det:art' in code or 'article' in code:
        return change + 'Mạo từ phải phù hợp với danh từ theo sau.'
    if 'pron' in code:
        return change + 'Đại từ phải phù hợp với vai trò của nó trong câu.'
    if 'verb' in code:
        return change + 'Dạng động từ phải phù hợp với chủ ngữ và cấu trúc câu.'
    if 'grammar' in code:
        return change + 'Cách sửa này giúp cấu trúc câu đúng ngữ pháp hơn.'
    if 'fluency' in code or 'style' in code:
        return change + 'Cách diễn đạt này tự nhiên và rõ ràng hơn.'
    return change + 'Cách sửa này giúp câu tự nhiên và chính xác hơn.'


def _contextual_fallback_explanation(issue, text=''):
    code = str(issue.get('category') or '').casefold()
    original = str(issue.get('original') or '')
    replacement = str(issue.get('replacement') or '')
    change = (
        f'Thay “{original}” bằng “{replacement}”. '
        if original and replacement else
        f'Bỏ “{original}”. ' if original else
        f'Thêm “{replacement}”. ' if replacement else ''
    )
    sentence, prefix = _sentence_context(text, issue.get('offset', 0))
    original_word = _normalized_word(original)
    replacement_word = _normalized_word(replacement)
    auxiliary = _preceding_perfect_auxiliary(prefix)
    replacement_form = _verb_form(replacement)
    original_form = _verb_form(original)

    if {original_word, replacement_word} == {'has', 'have'}:
        subject, subject_number = _subject_before(prefix)
        subject_lead = _subject_explanation_lead(subject, subject_number)
        if replacement_word == 'have':
            subject_detail = (
                subject_lead if subject_lead else
                'Với “I”, “you”, “we”, “they” và chủ ngữ số nhiều, '
            )
            return (
                change + subject_detail + 'dùng trợ động từ “have”, không dùng “has”. '
                '“Has” chỉ dùng với “he”, “she”, “it” và chủ ngữ ngôi thứ ba số ít.'
            )
        subject_detail = (
            subject_lead if subject_lead else
            'Với “he”, “she”, “it” và chủ ngữ ngôi thứ ba số ít, '
        )
        return (
            change + subject_detail + 'dùng trợ động từ “has”, không dùng “have”. '
            '“Have” được dùng với “I”, “you”, “we”, “they” và chủ ngữ số nhiều.'
        )

    if auxiliary and replacement_form and 'phân từ' in replacement_form['label']:
        base = replacement_form.get('base') or original
        contrast = (
            f', còn “{original}” là dạng {original_form["label"]}'
            if original_form else ''
        )
        return (
            change + f'Sau “{auxiliary}”, động từ chính phải ở dạng quá khứ phân từ (V3). '
            f'“{replacement}” là V3 của “{base}”{contrast}.'
        )

    if {original_word, replacement_word} == {"haven't", "hasn't"}:
        subject, subject_number = _subject_before(prefix)
        subject_lead = _subject_explanation_lead(subject, subject_number)
        if replacement_word == "hasn't":
            subject_detail = (
                subject_lead if subject_lead else
                'Với “he”, “she”, “it” và chủ ngữ số ít, '
            )
            return (
                change + subject_detail + 'dùng “has not/hasn’t”. '
                '“Have not/haven’t” được dùng với “I”, “you”, “we” và chủ ngữ số nhiều.'
            )
        subject_detail = (
            subject_lead if subject_lead else
            'Với “I”, “you”, “we” và chủ ngữ số nhiều, '
        )
        return (
            change + subject_detail + 'dùng “have not/haven’t”. '
            '“Has not/hasn’t” chỉ dùng với “he”, “she”, “it” và chủ ngữ số ít.'
        )

    if {original_word, replacement_word} == {'since', 'for'}:
        expression, expression_kind = _temporal_complement(issue, text)
        if replacement_word == 'for':
            detail = (
                f'“{expression}” diễn tả một khoảng thời gian, nên '
                if expression and expression_kind == 'duration' else
                '“For” được dùng để chỉ một khoảng thời gian, nên '
            )
            return (
                change + detail + 'dùng “for”. '
                '“Since” dùng với một mốc bắt đầu cụ thể, chẳng hạn “since 2024” hoặc “since Monday”.'
            )
        detail = (
            f'“{expression}” là mốc bắt đầu của hành động, nên '
            if expression and expression_kind == 'start_point' else
            '“Since” được dùng để chỉ mốc bắt đầu của hành động, nên '
        )
        return (
            change + detail + 'dùng “since”. '
            '“For” dùng với một khoảng thời gian, chẳng hạn “for two years”.'
        )

    cue = _past_time_cue(sentence)
    if cue and replacement_form and (
        'quá khứ đơn' in replacement_form['label'] or replacement_word.endswith('ed')
    ):
        base = replacement_form.get('base') or original
        return (
            change + f'Dấu hiệu “{cue}” cho biết hành động đã xảy ra và kết thúc trong quá khứ, '
            f'nên dùng thì quá khứ đơn. “{replacement}” là dạng quá khứ của “{base}”.'
        )

    if replacement_form and ('tense' in code or 'verb' in code):
        base = replacement_form.get('base') or original
        original_detail = (
            f' Trong khi đó, “{original}” là dạng {original_form["label"]}.'
            if original_form else ''
        )
        return (
            change + f'“{replacement}” là dạng {replacement_form["label"]} của “{base}”.'
            f'{original_detail}'
        )

    return _fallback_explanation(issue, text)


def explain_issues_in_vietnamese(issues, text=''):
    if not issues:
        return issues, 0
    for issue in issues:
        issue['explanation'] = _contextual_fallback_explanation(issue, text)

    client = get_gemini_client()
    if not client:
        return issues, 0

    input_issues = [{
        'id': issue['id'],
        'category': issue.get('category', ''),
        'error_type': issue.get('error_type', ''),
        'original': issue.get('original', ''),
        'replacement': issue.get('replacement', ''),
        'sentence_context': _sentence_context(text, issue.get('offset', 0))[0],
        'grammar_facts': _grammar_facts(issue, text),
    } for issue in issues]

    for model in (GEMINI_MODEL, GEMINI_FALLBACK_MODEL):
        try:
            response = client.models.generate_content(
                model=model,
                contents=json.dumps(input_issues, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type='application/json',
                    temperature=0.1,
                    thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
                ),
            )
            result = json.loads((response.text or '').strip())
            if not isinstance(result, list):
                raise ValueError('Gemini returned an invalid explanation list.')
            explanations = {
                item.get('id'): item.get('explanation', '').strip()
                for item in result
                if isinstance(item, dict) and isinstance(item.get('explanation'), str)
            }
            concepts = {
                item.get('id'): item.get('concept_code', '').strip()
                for item in result
                if isinstance(item, dict) and isinstance(item.get('concept_code'), str)
            }
            explained = 0
            for issue in issues:
                explanation = explanations.get(issue['id'])
                if explanation and _specific_explanation(explanation, issue, text):
                    issue['explanation'] = explanation
                    explained += 1
                if concepts.get(issue['id']):
                    issue['concept_code'] = concepts[issue['id']]
            return issues, explained
        except Exception as exc:
            print(f'[Gemini Explanation Warning] {model}: {exc}')
            if model != GEMINI_MODEL or not any(code in str(exc) for code in ('503', '504')):
                break
    return issues, 0
