import os
import uuid

import requests


SAPLING_API_URL = os.getenv('SAPLING_API_URL', 'https://api.sapling.ai/api/v1/edits')
SAPLING_TIMEOUT = float(os.getenv('SAPLING_TIMEOUT', '15'))
MAX_CHUNK_LENGTH = 4000
_http_session = requests.Session()


def _map_language(language):
    normalized = (language or 'en-US').strip().lower().replace('_', '-')
    varieties = {
        'en-us': 'us-variety',
        'en-gb': 'gb-variety',
        'en-au': 'au-variety',
        'en-ca': 'ca-variety',
    }
    if normalized in varieties:
        return 'en', varieties[normalized]
    return normalized.split('-', 1)[0], None


def _split_text(text, max_length=MAX_CHUNK_LENGTH):
    """Split at paragraph or whitespace boundaries and retain absolute offsets."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_length, len(text))
        if end < len(text):
            boundary = max(text.rfind('\n\n', start, end), text.rfind(' ', start, end), text.rfind('\n', start, end))
            if boundary > start:
                end = boundary
        if end <= start:
            end = min(start + max_length, len(text))
        chunks.append((start, text[start:end]))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    return chunks


def _map_error_type(general_error_type):
    category = (general_error_type or '').lower()
    if category in {'grammar', 'spelling', 'punctuation'}:
        return 'grammar'
    return 'clarity'


def _check_chunk(text, language, variety, api_key):
    payload = {'text': text, 'lang': language, 'include_error_categories': True}
    if variety:
        payload['variety'] = variety
    try:
        response = _http_session.post(
            SAPLING_API_URL,
            json=payload,
            headers={'Authorization': f'Bearer {api_key}', 'Accept': 'application/json'},
            timeout=SAPLING_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except requests.Timeout as exc:
        raise RuntimeError('Sapling API request timed out.') from exc
    except requests.RequestException as exc:
        detail = ''
        if getattr(exc, 'response', None) is not None:
            try:
                detail = exc.response.json().get('msg', '')
            except (ValueError, AttributeError):
                detail = ''
        raise RuntimeError(f'Sapling API request failed: {detail or str(exc)}') from exc

    if not isinstance(data, dict) or not isinstance(data.get('edits', []), list):
        raise RuntimeError('Sapling returned an invalid response.')
    return data.get('edits', [])


def check_text_with_sapling(text, language='en-US'):
    if not text or not text.strip():
        return []
    api_key = os.getenv('SAPLING_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError('SAPLING_API_KEY is not configured.')

    sapling_language, variety = _map_language(language)
    issues = []
    for chunk_start, chunk in _split_text(text):
        for edit in _check_chunk(chunk, sapling_language, variety, api_key):
            try:
                offset = chunk_start + int(edit['sentence_start']) + int(edit['start'])
                end = chunk_start + int(edit['sentence_start']) + int(edit['end'])
                replacement = edit.get('replacement', '')
                if offset < 0 or end < offset or end > len(text):
                    continue
                original = text[offset:end]
                if not original and not replacement:
                    continue
            except (KeyError, TypeError, ValueError):
                continue

            general_type = edit.get('general_error_type', 'Other')
            error_type = edit.get('error_type', '')
            issues.append({
                'id': f"issue-{uuid.uuid4().hex[:8]}",
                'provider_issue_id': edit.get('id'),
                'type': _map_error_type(general_type),
                'category': f'{general_type.upper()} • {error_type}' if error_type else general_type.upper(),
                'original': original,
                'replacement': replacement,
                'replacements': [replacement] if replacement else [],
                'explanation': '',
                'offset': offset,
                'length': end - offset,
                'shortcut': 'Press [Tab]',
            })
    return issues
