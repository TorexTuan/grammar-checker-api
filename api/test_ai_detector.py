from unittest.mock import Mock, patch

import requests
from django.contrib.auth import get_user_model
from django.core.cache import cache
from rest_framework.test import APITestCase

from .services.ai_detector import (
    AiDetectorRateLimit,
    AiDetectorTimeout,
    InvalidAiDetectorResponse,
    detect_ai_text,
    normalize_ai_detection,
)


User = get_user_model()
TEXT = ('Human and machine writing can share many stylistic patterns. ' * 7)[:360]
DETECTION = {
    'score': 0.84,
    'ai_fraction': 0.62,
    'verdict': 'ai',
    'low_confidence': False,
    'low_confidence_reasons': [],
    'language': 'en',
    'version': 'test-version',
    'character_count': len(TEXT),
    'segments': [],
}


class AiDetectorServiceTests(APITestCase):
    def test_normalizes_segments_against_submitted_text(self):
        payload = {
            **DETECTION,
            'language': {'code': 'en', 'name': 'English', 'confidence': 0.99},
            'segments': [{
                'start': 0,
                'end': 5,
                'score': 0.91,
                'text': TEXT[:5],
            }],
        }
        result = normalize_ai_detection(payload, TEXT)
        self.assertEqual(result['segments'][0]['text'], TEXT[:5])
        self.assertEqual(result['character_count'], len(TEXT))
        self.assertEqual(result['language'], 'en')

    def test_rejects_invalid_scores_ranges_and_text(self):
        with self.assertRaises(InvalidAiDetectorResponse):
            normalize_ai_detection({**DETECTION, 'score': 1.2}, TEXT)
        with self.assertRaises(InvalidAiDetectorResponse):
            normalize_ai_detection({
                **DETECTION,
                'segments': [{'start': 0, 'end': len(TEXT) + 1, 'score': 0.8}],
            }, TEXT)
        with self.assertRaises(InvalidAiDetectorResponse):
            normalize_ai_detection({
                **DETECTION,
                'segments': [{'start': 0, 'end': 5, 'score': 0.8, 'text': 'wrong'}],
            }, TEXT)

    @patch.dict('os.environ', {'SAPLING_API_KEY': 'test-key'})
    @patch('api.services.ai_detector._http_session.post')
    def test_sends_private_zero_retention_request(self, post):
        response = Mock()
        response.status_code = 200
        response.json.return_value = DETECTION
        response.raise_for_status.return_value = None
        post.return_value = response

        result = detect_ai_text(TEXT)

        self.assertEqual(result['score'], 0.84)
        request = post.call_args
        self.assertEqual(request.kwargs['json'], {
            'text': TEXT,
            'lang': 'en',
            'sent_scores': True,
            'zero_retention': True,
        })
        self.assertEqual(request.kwargs['headers']['Authorization'], 'Bearer test-key')

    @patch.dict('os.environ', {'SAPLING_API_KEY': 'test-key'})
    @patch('api.services.ai_detector._http_session.post')
    def test_maps_provider_timeout_and_rate_limit(self, post):
        post.side_effect = requests.Timeout()
        with self.assertRaises(AiDetectorTimeout):
            detect_ai_text(TEXT)

        response = Mock()
        response.status_code = 429
        response.headers = {'Retry-After': '30'}
        response.json.return_value = {'msg': 'slow down'}
        post.side_effect = None
        post.return_value = response
        with self.assertRaises(AiDetectorRateLimit) as raised:
            detect_ai_text(TEXT)
        self.assertEqual(raised.exception.retry_after, '30')


class AiDetectorApiTests(APITestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(
            username='detector@example.com',
            email='detector@example.com',
            password='StrongPass!2026',
        )

    def tearDown(self):
        cache.clear()

    def test_requires_authentication(self):
        response = self.client.post('/api/ai-detect/', {'text': TEXT}, format='json')
        self.assertIn(response.status_code, (401, 403))

    @patch('api.views.detect_ai_text', return_value=DETECTION)
    def test_returns_normalized_result_for_authenticated_user(self, detect):
        self.client.force_authenticate(self.user)
        response = self.client.post('/api/ai-detect/', {'text': TEXT}, format='json')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data['success'])
        self.assertEqual(response.data['verdict'], 'ai')
        detect.assert_called_once_with(TEXT)

    @patch('api.views.detect_ai_text', return_value=DETECTION)
    def test_validates_character_limits_before_provider_call(self, detect):
        self.client.force_authenticate(self.user)
        short = self.client.post('/api/ai-detect/', {'text': 'a' * 299}, format='json')
        long = self.client.post('/api/ai-detect/', {'text': 'a' * 10001}, format='json')
        self.assertEqual(short.status_code, 400)
        self.assertEqual(long.status_code, 400)
        detect.assert_not_called()

    @patch('api.views.detect_ai_text', return_value=DETECTION)
    def test_throttles_after_ten_requests_per_hour(self, detect):
        self.client.force_authenticate(self.user)
        statuses = [
            self.client.post('/api/ai-detect/', {'text': TEXT}, format='json').status_code
            for _ in range(11)
        ]
        self.assertEqual(statuses[:10], [200] * 10)
        self.assertEqual(statuses[10], 429)
