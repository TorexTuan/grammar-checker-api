import json
import math

from google.genai import types

from .gemini import get_gemini_client


ASSESSMENT_MODEL = 'gemini-3.5-flash-lite'
CRITERIA = ('grammar', 'clarity', 'coherence', 'vocabulary')


class AssessmentError(RuntimeError):
    """Base error raised when a quality assessment cannot be produced."""


class AssessmentTimeout(AssessmentError):
    """Raised when Gemini does not finish an assessment in time."""


class InvalidAssessmentResponse(AssessmentError):
    """Raised when Gemini returns malformed or incomplete assessment data."""


SYSTEM_PROMPT = """
Bạn là giáo viên tiếng Anh đánh giá chất lượng viết tiếng Anh phổ thông.
Hãy đánh giá đúng nội dung được cung cấp, không sửa lại toàn bộ bài viết.
Mọi nhận xét phải bằng tiếng Việt, cụ thể, ngắn gọn và hữu ích.
Trả về duy nhất một JSON object có cấu trúc:
{
  "overall_score": 0,
  "criteria": {
    "grammar": {"score": 0, "feedback": ""},
    "clarity": {"score": 0, "feedback": ""},
    "coherence": {"score": 0, "feedback": ""},
    "vocabulary": {"score": 0, "feedback": ""}
  },
  "summary": "",
  "strengths": [""],
  "priorities": [""]
}
Mỗi điểm là số từ 0 đến 100. strengths phải có từ 1 đến 3 mục và priorities
phải có từ 1 đến 3 mục, sắp xếp theo mức độ quan trọng.
"""


def _is_timeout_error(exc):
    message = str(exc).lower()
    name = type(exc).__name__.lower()
    return (
        isinstance(exc, TimeoutError)
        or 'timeout' in name
        or 'timed out' in message
        or 'deadline_exceeded' in message
        or 'deadline expired' in message
        or '504' in message
    )


def _normalize_score(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InvalidAssessmentResponse(f'Invalid score for {field}.')
    return max(0, min(100, int(round(value))))


def _string_list(value, field, maximum):
    if not isinstance(value, list):
        raise InvalidAssessmentResponse(f'Invalid {field} list.')
    cleaned = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise InvalidAssessmentResponse(f'Invalid item in {field}.')
        cleaned.append(item.strip())
    if not cleaned:
        raise InvalidAssessmentResponse(f'{field} must not be empty.')
    return cleaned[:maximum]


def normalize_assessment(payload):
    if not isinstance(payload, dict):
        raise InvalidAssessmentResponse('Assessment must be a JSON object.')

    # Validate the provider field even though the API recomputes it below.
    _normalize_score(payload.get('overall_score'), 'overall')

    criteria_payload = payload.get('criteria')
    if not isinstance(criteria_payload, dict):
        raise InvalidAssessmentResponse('Assessment criteria are missing.')

    criteria = {}
    for name in CRITERIA:
        item = criteria_payload.get(name)
        if not isinstance(item, dict):
            raise InvalidAssessmentResponse(f'Missing criterion: {name}.')
        feedback = item.get('feedback')
        if not isinstance(feedback, str) or not feedback.strip():
            raise InvalidAssessmentResponse(f'Invalid feedback for {name}.')
        criteria[name] = {
            'score': _normalize_score(item.get('score'), name),
            'feedback': feedback.strip(),
        }

    summary = payload.get('summary')
    if not isinstance(summary, str) or not summary.strip():
        raise InvalidAssessmentResponse('Assessment summary is missing.')

    total = sum(item['score'] for item in criteria.values())
    return {
        'overall_score': int(total / len(CRITERIA) + 0.5),
        'criteria': criteria,
        'summary': summary.strip(),
        'strengths': _string_list(payload.get('strengths'), 'strengths', 3),
        'priorities': _string_list(payload.get('priorities'), 'priorities', 3),
    }


def assess_text_quality(text, language='en-US'):
    client = get_gemini_client()
    if not client:
        raise AssessmentError('GEMINI_API_KEY is not configured.')

    request_payload = {'language': language, 'text': text}
    try:
        response = client.models.generate_content(
            model=ASSESSMENT_MODEL,
            contents=json.dumps(request_payload, ensure_ascii=False),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type='application/json',
                temperature=0.1,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.LOW,
                ),
            ),
        )
    except Exception as exc:
        if _is_timeout_error(exc):
            raise AssessmentTimeout('Gemini assessment request timed out.') from exc
        raise AssessmentError('Gemini assessment request failed.') from exc

    try:
        payload = json.loads((response.text or '').strip())
    except (TypeError, ValueError, AttributeError) as exc:
        raise InvalidAssessmentResponse('Gemini returned invalid JSON.') from exc
    return normalize_assessment(payload)
