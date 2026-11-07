import os
import uuid
from urllib.parse import quote

import requests


SAPLING_API_URL = os.getenv('SAPLING_API_URL', 'https://api.sapling.ai/api/v1/edits')
SAPLING_TIMEOUT = float(os.getenv('SAPLING_TIMEOUT', '15'))
SAPLING_AUTOCOMPLETE_URL = os.getenv('SAPLING_AUTOCOMPLETE_URL', 'https://api.sapling.ai/api/v1/complete')
SAPLING_AUTOCOMPLETE_TIMEOUT = float(os.getenv('SAPLING_AUTOCOMPLETE_TIMEOUT', '5'))
SAPLING_REPHRASE_URL = os.getenv('SAPLING_REPHRASE_URL', 'https://api.sapling.ai/api/v1/rephrase')
SAPLING_REPHRASE_TIMEOUT = float(os.getenv('SAPLING_REPHRASE_TIMEOUT', '12'))
MAX_CHUNK_LENGTH = 4000
MAX_AUTOCOMPLETE_QUERY_LENGTH = 200
MAX_REPHRASE_TEXT_LENGTH = 4000
_http_session = requests.Session()


class SaplingAutocompleteError(RuntimeError):
    pass


class SaplingAutocompleteTimeout(SaplingAutocompleteError):
    pass


class SaplingRephraseError(RuntimeError):
    pass


class SaplingRephraseTimeout(SaplingRephraseError):
    pass


class SaplingRephraseRateLimit(SaplingRephraseError):
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


class InvalidSaplingRephraseResponse(SaplingRephraseError):
    pass


def _sapling_api_key():
    api_key = os.getenv('SAPLING_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError('SAPLING_API_KEY is not configured.')
    return api_key


def _autocomplete_request(url, payload, retry_transient=False):
    try:
        api_key = _sapling_api_key()
    except RuntimeError as exc:
        raise SaplingAutocompleteError(str(exc)) from exc
    attempts = 2 if retry_transient else 1
    for attempt in range(attempts):
        try:
            response = _http_session.post(
                url,
                json=payload,
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Accept': 'application/json',
                },
                timeout=SAPLING_AUTOCOMPLETE_TIMEOUT,
            )
            if attempt == 0 and response.status_code in {500, 502, 503, 504}:
                continue
            response.raise_for_status()
            data = response.json()
        except requests.Timeout as exc:
            raise SaplingAutocompleteTimeout('Sapling autocomplete request timed out.') from exc
        except (requests.RequestException, ValueError) as exc:
            detail = ''
            if getattr(exc, 'response', None) is not None:
                try:
                    detail = exc.response.json().get('msg', '')
                except (ValueError, AttributeError):
                    detail = ''
            raise SaplingAutocompleteError(
                f'Sapling autocomplete request failed: {detail or str(exc)}'
            ) from exc
        # Sapling returns an empty JSON list when the model has no completion
        # for an otherwise valid query.
        if data == []:
            return None
        if not isinstance(data, dict):
            raise SaplingAutocompleteError('Sapling returned an invalid autocomplete response.')
        return data


def autocomplete_with_sapling(query, session_id=None):
    if not isinstance(query, str) or not query.strip():
        raise ValueError('Autocomplete query must be a non-empty string.')
    if len(query) > MAX_AUTOCOMPLETE_QUERY_LENGTH:
        raise ValueError(
            f'Autocomplete query must not exceed {MAX_AUTOCOMPLETE_QUERY_LENGTH} characters.'
        )

    payload = {'query': query}
    if session_id:
        payload['session_id'] = session_id
    data = _autocomplete_request(SAPLING_AUTOCOMPLETE_URL, payload, retry_transient=True)
    if data is None:
        return {'output': '', 'query': query, 'hash': ''}
    output = data.get('output')
    response_query = data.get('query')
    completion_id = data.get('hash')
    if not all(isinstance(value, str) for value in (output, response_query, completion_id)):
        raise SaplingAutocompleteError('Sapling returned an invalid autocomplete response.')
    return {'output': output[:1000], 'query': response_query, 'hash': completion_id}


def accept_sapling_completion(completion_id, query, completion, session_id=None):
    try:
        normalized_id = str(uuid.UUID(completion_id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError('Invalid completion ID.') from exc
    payload = {'context': {'query': query, 'completion': completion}}
    if session_id:
        payload['session_id'] = session_id
    url = SAPLING_AUTOCOMPLETE_URL.rstrip('/') + '/' + quote(normalized_id, safe='') + '/accept'
    data = _autocomplete_request(url, payload)
    if data.get('result') != 'success':
        raise SaplingAutocompleteError('Sapling did not accept the completion event.')


def _provider_error_detail(response):
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return payload.get('msg') or payload.get('error') or ''
    except ValueError:
        pass
    return ''


def rephrase_with_sapling(text, num_results=3):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Rephrase text must be a non-empty string.')
    if len(text) > MAX_REPHRASE_TEXT_LENGTH:
        raise ValueError(f'Rephrase text must not exceed {MAX_REPHRASE_TEXT_LENGTH} characters.')
    if isinstance(num_results, bool) or not isinstance(num_results, int) or not 1 <= num_results <= 3:
        raise ValueError('num_results must be an integer from 1 to 3.')
    try:
        api_key = _sapling_api_key()
    except RuntimeError as exc:
        raise SaplingRephraseError(str(exc)) from exc

    payload = {
        'text': text,
        'mapping': 'switch_tone',
        'tone_mapping': 'professional',
        'num_results': num_results,
    }
    for attempt in range(2):
        try:
            response = _http_session.post(
                SAPLING_REPHRASE_URL,
                json=payload,
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Accept': 'application/json',
                },
                timeout=SAPLING_REPHRASE_TIMEOUT,
            )
        except requests.Timeout as exc:
            raise SaplingRephraseTimeout('Sapling rephrase request timed out.') from exc
        except requests.RequestException as exc:
            raise SaplingRephraseError(f'Sapling rephrase request failed: {exc}') from exc

        if response.status_code == 429:
            raise SaplingRephraseRateLimit(
                _provider_error_detail(response) or 'Sapling rephrase rate limit reached.',
                response.headers.get('Retry-After'),
            )
        if attempt == 0 and response.status_code in {500, 502, 503, 504}:
            continue
        if not 200 <= response.status_code < 300:
            detail = _provider_error_detail(response)
            raise SaplingRephraseError(
                f'Sapling rephrase request failed: {detail or response.status_code}'
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise InvalidSaplingRephraseResponse('Sapling returned invalid rephrase JSON.') from exc
        break

    if not isinstance(data, dict) or not isinstance(data.get('results'), list):
        raise InvalidSaplingRephraseResponse('Sapling returned an invalid rephrase response.')

    normalized = []
    seen = set()
    for item in data['results']:
        if not isinstance(item, dict):
            raise InvalidSaplingRephraseResponse('Sapling returned an invalid rephrase result.')
        replacement = item.get('replacement')
        result_id = item.get('hash')
        if (
            not isinstance(replacement, str)
            or not replacement.strip()
            or not isinstance(result_id, str)
            or not result_id.strip()
        ):
            raise InvalidSaplingRephraseResponse('Sapling returned an invalid rephrase result.')
        replacement = replacement.strip()
        result_id = result_id.strip()
        duplicate_key = replacement.casefold()
        if duplicate_key in seen:
            continue
        seen.add(duplicate_key)
        normalized.append({'id': result_id, 'text': replacement})
        if len(normalized) == num_results:
            break
    return normalized


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
    api_key = _sapling_api_key()

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
