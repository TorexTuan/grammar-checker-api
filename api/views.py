from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .services.assessment import (
    AssessmentError,
    AssessmentTimeout,
    InvalidAssessmentResponse,
    assess_text_quality,
)
from .services.gemini import explain_issues_in_vietnamese
from .services.sapling import check_text_with_sapling


SAPLING_TIMEOUT = 30
GEMINI_TIMEOUT = 25
ASSESSMENT_TIMEOUT = 20
MIN_ASSESSMENT_WORDS = 20
MAX_ASSESSMENT_CHARACTERS = 10000


@api_view(['GET'])
def health_check(request):
    return Response({
        'status': 'ok',
        'message': 'Django backend is running successfully.',
    })


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

        return Response({
            'success': True,
            'total_issues': len(issues),
            'sources': {
                'sapling': len(issues),
                'gemini_explained': explained_count,
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
