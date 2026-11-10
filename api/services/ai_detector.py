import math
import os

import requests

from .sapling import _provider_error_detail, _sapling_api_key


SAPLING_AI_DETECT_URL = os.getenv(
    'SAPLING_AI_DETECT_URL',
    'https://api.sapling.ai/api/v1/aidetect',
)
SAPLING_AI_DETECT_TIMEOUT = float(os.getenv('SAPLING_AI_DETECT_TIMEOUT', '20'))
_http_session = requests.Session()


class AiDetectorError(RuntimeError):
    pass


class AiDetectorTimeout(AiDetectorError):
    pass


class AiDetectorRateLimit(AiDetectorError):
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


class InvalidAiDetectorResponse(AiDetectorError):
    pass


def _probability(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InvalidAiDetectorResponse(f'Invalid {field}.')
    if not 0 <= value <= 1:
        raise InvalidAiDetectorResponse(f'{field} must be between 0 and 1.')
    return float(value)


def _optional_string(value, field):
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidAiDetectorResponse(f'Invalid {field}.')
    return value.strip() or None


def _language_code(value):
    # Sapling currently returns language metadata as an object, while older
    # detector versions returned the language code directly.
    if isinstance(value, dict):
        value = value.get('code')
    return _optional_string(value, 'language')


def normalize_ai_detection(payload, text):
    if not isinstance(payload, dict):
        raise InvalidAiDetectorResponse('AI detector response must be an object.')

    verdict = payload.get('verdict')
    if verdict not in {'human', 'mixed', 'ai'}:
        raise InvalidAiDetectorResponse('Invalid detector verdict.')

    low_confidence = payload.get('low_confidence', False)
    if not isinstance(low_confidence, bool):
        raise InvalidAiDetectorResponse('Invalid low confidence flag.')

    reasons = payload.get('low_confidence_reasons', [])
    if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
        raise InvalidAiDetectorResponse('Invalid low confidence reasons.')

    raw_segments = payload.get('segments', [])
    if not isinstance(raw_segments, list):
        raise InvalidAiDetectorResponse('Invalid detector segments.')
    segments = []
    for item in raw_segments:
        if not isinstance(item, dict):
            raise InvalidAiDetectorResponse('Invalid detector segment.')
        start = item.get('start')
        end = item.get('end')
        if (
            isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, int) or not isinstance(end, int)
            or start < 0 or end <= start or end > len(text)
        ):
            raise InvalidAiDetectorResponse('Invalid detector segment range.')
        segment_text = text[start:end]
        provider_text = item.get('text') or item.get('sentence')
        if provider_text is not None and provider_text != segment_text:
            raise InvalidAiDetectorResponse('Detector segment does not match the submitted text.')
        segments.append({
            'start': start,
            'end': end,
            'score': _probability(item.get('score'), 'segment score'),
            'text': segment_text,
        })

    return {
        'score': _probability(payload.get('score'), 'score'),
        'ai_fraction': _probability(payload.get('ai_fraction'), 'ai fraction'),
        'verdict': verdict,
        'low_confidence': low_confidence,
        'low_confidence_reasons': [item.strip() for item in reasons if item.strip()],
        'language': _language_code(payload.get('language')),
        'version': _optional_string(payload.get('version'), 'version'),
        'character_count': len(text),
        'segments': segments,
    }


def detect_ai_text(text):
    try:
        api_key = _sapling_api_key()
    except RuntimeError as exc:
        raise AiDetectorError(str(exc)) from exc

    try:
        response = _http_session.post(
            SAPLING_AI_DETECT_URL,
            json={
                'text': text,
                'lang': 'en',
                'sent_scores': True,
                'zero_retention': True,
            },
            headers={
                'Authorization': f'Bearer {api_key}',
                'Accept': 'application/json',
            },
            timeout=SAPLING_AI_DETECT_TIMEOUT,
        )
    except requests.Timeout as exc:
        raise AiDetectorTimeout('Sapling AI detector request timed out.') from exc
    except requests.RequestException as exc:
        raise AiDetectorError(f'Sapling AI detector request failed: {exc}') from exc

    if response.status_code == 429:
        raise AiDetectorRateLimit(
            _provider_error_detail(response) or 'Sapling AI detector rate limit reached.',
            response.headers.get('Retry-After'),
        )
    try:
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        raise AiDetectorTimeout('Sapling AI detector request timed out.') from exc
    except (requests.RequestException, ValueError) as exc:
        detail = _provider_error_detail(response)
        raise AiDetectorError(
            f'Sapling AI detector request failed: {detail or str(exc)}'
        ) from exc
    return normalize_ai_detection(payload, text)
