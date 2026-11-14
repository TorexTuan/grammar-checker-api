from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import UserRateThrottle

from .services.ai_detector import (
    AiDetectorError,
    AiDetectorRateLimit,
    AiDetectorTimeout,
    InvalidAiDetectorResponse,
    detect_ai_text,
)

from .services.assessment import (
    AssessmentError,
    AssessmentTimeout,
    InvalidAssessmentResponse,
    assess_text_quality,
)
from .services.gemini import explain_issues_in_vietnamese
from .services.practice import enrich_practice_issues, record_detected_mistakes
from .services.sapling import (
    MAX_AUTOCOMPLETE_QUERY_LENGTH,
    MAX_REPHRASE_TEXT_LENGTH,
    InvalidSaplingRephraseResponse,
    SaplingAutocompleteError,
    SaplingAutocompleteTimeout,
    SaplingRephraseError,
    SaplingRephraseRateLimit,
    SaplingRephraseTimeout,
    accept_sapling_completion,
    autocomplete_with_sapling,
    check_text_with_sapling,
    rephrase_with_sapling,
)


SAPLING_TIMEOUT = 30
GEMINI_TIMEOUT = 25
ASSESSMENT_TIMEOUT = 20
MIN_ASSESSMENT_WORDS = 20
MAX_ASSESSMENT_CHARACTERS = 10000
MAX_AUTOCOMPLETE_SESSION_LENGTH = 128
MAX_AUTOCOMPLETE_OUTPUT_LENGTH = 1000
MIN_AI_DETECT_CHARACTERS = 300
MAX_AI_DETECT_CHARACTERS = 10000


class AiDetectorUserThrottle(UserRateThrottle):
    rate = '10/hour'


@api_view(['GET'])
def health_check(request):
    return Response({
        'status': 'ok',
        'message': 'Django backend is running successfully.',
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@throttle_classes([AiDetectorUserThrottle])
def ai_detect(request):
    text = request.data.get('text')
    if not isinstance(text, str):
        return Response(
            {'success': False, 'error': 'text must be a string.'},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if len(text) < MIN_AI_DETECT_CHARACTERS:
        return Response(
            {'success': False, 'error': f'Text must contain at least {MIN_AI_DETECT_CHARACTERS} characters.'},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if len(text) > MAX_AI_DETECT_CHARACTERS:
        return Response(
            {'success': False, 'error': f'Text must not exceed {MAX_AI_DETECT_CHARACTERS:,} characters.'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        return Response({'success': True, **detect_ai_text(text)})
    except AiDetectorRateLimit as exc:
        response = Response(
            {'success': False, 'error': 'AI detector rate limit reached.'},
            status=status.HTTP_429_TOO_MANY_REQUESTS,
        )
        if exc.retry_after:
            response['Retry-After'] = exc.retry_after
        return response
    except AiDetectorTimeout:
        return Response(
            {'success': False, 'error': 'AI detection timed out.'},
            status=status.HTTP_504_GATEWAY_TIMEOUT,
        )
    except InvalidAiDetectorResponse as exc:
        print(f'[AI Detector Invalid Response]: {exc}')
        return Response(
            {'success': False, 'error': 'AI detector returned an invalid response.'},
            status=status.HTTP_502_BAD_GATEWAY,
        )
    except AiDetectorError as exc:
        print(f'[AI Detector Error]: {exc}')
        return Response(
            {'success': False, 'error': 'AI detector is temporarily unavailable.'},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


def _autocomplete_string(data, field, maximum):
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Invalid {field}. Expected a non-empty string.')
    if len(value) > maximum:
        raise ValueError(f'{field} must not exceed {maximum} characters.')
    return value


@api_view(['POST'])
def autocomplete(request):
    try:
        query = _autocomplete_string(request.data, 'query', MAX_AUTOCOMPLETE_QUERY_LENGTH)
        session_id = request.data.get('session_id')
        if session_id is not None:
            session_id = _autocomplete_string(
                request.data, 'session_id', MAX_AUTOCOMPLETE_SESSION_LENGTH
            )
        completion = autocomplete_with_sapling(query, session_id)
        return Response({'success': True, **completion})
    except ValueError as exc:
        return Response({'success': False, 'error': str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    except SaplingAutocompleteTimeout:
        return Response(
            {'success': False, 'error': 'Autocomplete request timed out.'},
            status=status.HTTP_504_GATEWAY_TIMEOUT,
        )
    except SaplingAutocompleteError as exc:
        print(f'[Sapling Autocomplete Error]: {exc}')
        return Response(
            {'success': False, 'error': 'Autocomplete is temporarily unavailable.'},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


@api_view(['POST'])
def accept_autocomplete(request):
    try:
        completion_id = _autocomplete_string(request.data, 'hash', 128)
        query = _autocomplete_string(request.data, 'query', MAX_AUTOCOMPLETE_QUERY_LENGTH)
        completion = _autocomplete_string(
            request.data, 'completion', MAX_AUTOCOMPLETE_OUTPUT_LENGTH
        )
        session_id = request.data.get('session_id')
        if session_id is not None:
            session_id = _autocomplete_string(
                request.data, 'session_id', MAX_AUTOCOMPLETE_SESSION_LENGTH
            )
        accept_sapling_completion(completion_id, query, completion, session_id)
        return Response({'success': True})
    except ValueError as exc:
        return Response({'success': False, 'error': str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    except SaplingAutocompleteTimeout:
        return Response(
            {'success': False, 'error': 'Autocomplete feedback timed out.'},
            status=status.HTTP_504_GATEWAY_TIMEOUT,
        )
    except SaplingAutocompleteError as exc:
        print(f'[Sapling Autocomplete Feedback Error]: {exc}')
        return Response(
            {'success': False, 'error': 'Unable to record autocomplete feedback.'},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


@api_view(['POST'])
def rephrase(request):
    text = request.data.get('text')
    style = request.data.get('style')
    num_results = request.data.get('num_results', 3)

    if not isinstance(text, str) or not text.strip():
        return Response(
            {'success': False, 'error': 'text must be a non-empty string.'},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if len(text) > MAX_REPHRASE_TEXT_LENGTH:
        return Response(
            {
                'success': False,
                'error': f'text must not exceed {MAX_REPHRASE_TEXT_LENGTH} characters.',
            },
            status=status.HTTP_400_BAD_REQUEST,
        )
    if style != 'professional':
        return Response(
            {'success': False, 'error': 'style must be professional.'},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if (
        isinstance(num_results, bool)
        or not isinstance(num_results, int)
        or not 1 <= num_results <= 3
    ):
        return Response(
            {'success': False, 'error': 'num_results must be an integer from 1 to 3.'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        results = rephrase_with_sapling(text, num_results)
        return Response({
            'success': True,
            'original': text,
            'results': results,
        })
    except SaplingRephraseRateLimit as exc:
        response = Response(
            {'success': False, 'error': 'Professional rewrite rate limit reached.'},
            status=status.HTTP_429_TOO_MANY_REQUESTS,
        )
        if exc.retry_after:
            response['Retry-After'] = exc.retry_after
        return response
    except SaplingRephraseTimeout:
        return Response(
            {'success': False, 'error': 'Professional rewrite request timed out.'},
            status=status.HTTP_504_GATEWAY_TIMEOUT,
        )
    except InvalidSaplingRephraseResponse as exc:
        print(f'[Sapling Rephrase Invalid Response]: {exc}')
        return Response(
            {'success': False, 'error': 'Professional rewrite returned an invalid response.'},
            status=status.HTTP_502_BAD_GATEWAY,
        )
    except ValueError as exc:
        return Response(
            {'success': False, 'error': str(exc)},
            status=status.HTTP_400_BAD_REQUEST,
        )
    except SaplingRephraseError as exc:
        print(f'[Sapling Rephrase Error]: {exc}')
        return Response(
            {'success': False, 'error': 'Professional rewrite is temporarily unavailable.'},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


@api_view(['POST'])
def assess_quality(request):
    text = request.data.get('text', '')
    language = request.data.get('language', 'en-US')
    if not isinstance(text, str):
        return Response({'success': False, 'error': 'Invalid text format. Expected a string.'}, status=status.HTTP_400_BAD_REQUEST)
    if not isinstance(language, str) or not language.strip():
        return Response({'success': False, 'error': 'Invalid language format.'}, status=status.HTTP_400_BAD_REQUEST)
    if len(text) > MAX_ASSESSMENT_CHARACTERS:
        return Response({'success': False, 'error': 'Text must not exceed 10,000 characters.'}, status=status.HTTP_400_BAD_REQUEST)
    if len(text.split()) < MIN_ASSESSMENT_WORDS:
        return Response({'success': False, 'error': 'Text must contain at least 20 words.'}, status=status.HTTP_400_BAD_REQUEST)

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(assess_text_quality, text, language.strip())
    try:
        assessment = future.result(timeout=ASSESSMENT_TIMEOUT)
        return Response({'success': True, **assessment})
    except FutureTimeoutError:
        future.cancel()
        return Response({'success': False, 'error': 'Quality assessment timed out.'}, status=status.HTTP_504_GATEWAY_TIMEOUT)
    except AssessmentTimeout:
        return Response({'success': False, 'error': 'Quality assessment timed out.'}, status=status.HTTP_504_GATEWAY_TIMEOUT)
    except InvalidAssessmentResponse:
        return Response({'success': False, 'error': 'Gemini returned an invalid assessment.'}, status=status.HTTP_502_BAD_GATEWAY)
    except AssessmentError:
        return Response({'success': False, 'error': 'Quality assessment is temporarily unavailable.'}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    finally:
        executor.shutdown(wait=False)


@api_view(['POST'])
def check_grammar(request):
    """Detect edits with Sapling and explain them in Vietnamese with Gemini."""
    text = request.data.get('text', '')
    language = request.data.get('language', 'en-US')
    capture_practice = request.data.get('capture_practice') is True
    excluded_fingerprints = request.data.get('excluded_practice_fingerprints', [])
    if not isinstance(excluded_fingerprints, list):
        excluded_fingerprints = []
    excluded_fingerprints = [str(value) for value in excluded_fingerprints[:500]]
    if not isinstance(text, str) or not text.strip():
        return Response(
            {'error': 'Invalid text format. Expected a non-empty string.'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        sapling_executor = ThreadPoolExecutor(max_workers=1)
        sapling_future = sapling_executor.submit(check_text_with_sapling, text, language)
        try:
            issues = sapling_future.result(timeout=SAPLING_TIMEOUT) or []
        except FutureTimeoutError:
            sapling_future.cancel()
            return Response(
                {'success': False, 'error': 'Sapling API request timed out.'},
                status=status.HTTP_504_GATEWAY_TIMEOUT,
            )
        finally:
            sapling_executor.shutdown(wait=False)

        if issues:
            gemini_executor = ThreadPoolExecutor(max_workers=1)
            gemini_future = gemini_executor.submit(explain_issues_in_vietnamese, issues, text)
            try:
                issues, explained_count = gemini_future.result(timeout=GEMINI_TIMEOUT)
            except FutureTimeoutError:
                gemini_future.cancel()
                explained_count = 0
            finally:
                gemini_executor.shutdown(wait=False)
        else:
            explained_count = 0

        enrich_practice_issues(issues, text)
        recorded_count = 0
        if capture_practice and request.user.is_authenticated:
            try:
                recorded_count = record_detected_mistakes(request.user, issues, excluded_fingerprints)
            except Exception as exc:
                print(f'[Practice Capture Warning]: {exc}')

        return Response({
            'success': True,
            'total_issues': len(issues),
            'sources': {
                'sapling': len(issues),
                'gemini_explained': explained_count,
                'practice_recorded': recorded_count,
            },
            'issues': issues,
        })
    except RuntimeError as exc:
        print(f'[Grammar Service Error]: {exc}')
        return Response(
            {
                'success': False,
                'error': 'Grammar service is temporarily unavailable. Please try again later.',
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    except Exception as exc:
        return Response(
            {'success': False, 'error': str(exc)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
