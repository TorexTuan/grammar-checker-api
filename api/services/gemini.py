import json
import os
import re

from google import genai
from google.genai import types


GEMINI_MODEL = 'gemini-3.8-flash'
GEMINI_FALLBACK_MODEL = 'gemini-3.5-flash-lite'
GEMINI_REQUEST_TIMEOUT_MS = 11000


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


def explain_issues_in_vietnamese(issues, text=''):
    if not issues:
        return issues, 0
    for issue in issues:
        issue['explanation'] = _fallback_explanation(issue, text)

    client = get_gemini_client()
    if not client:
        return issues, 0

    input_issues = [{
        'id': issue['id'],
        'category': issue.get('category', ''),
        'original': issue.get('original', ''),
        'replacement': issue.get('replacement', ''),
        'sentence_context': _sentence_context(text, issue.get('offset', 0))[0],
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
            explained = 0
            for issue in issues:
                explanation = explanations.get(issue['id'])
                if explanation:
                    issue['explanation'] = explanation
                    explained += 1
            return issues, explained
        except Exception as exc:
            print(f'[Gemini Explanation Warning] {model}: {exc}')
            if model != GEMINI_MODEL or not any(code in str(exc) for code in ('503', '504')):
                break
    return issues, 0
