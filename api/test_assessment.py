from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.test import SimpleTestCase
from rest_framework import status
from rest_framework.test import APITestCase

from .services.assessment import (
    AssessmentError,
    AssessmentTimeout,
    InvalidAssessmentResponse,
    assess_text_quality,
    normalize_assessment,
)


def valid_payload():
    return {
        'overall_score': 1,
        'criteria': {
            'grammar': {'score': 61, 'feedback': 'Ngữ pháp cần cải thiện.'},
            'clarity': {'score': 72, 'feedback': 'Ý tương đối rõ ràng.'},
            'coherence': {'score': 83, 'feedback': 'Các ý liên kết tốt.'},
            'vocabulary': {'score': 94, 'feedback': 'Từ vựng phù hợp.'},
        },
        'summary': 'Đoạn văn có nền tảng tốt.',
        'strengths': ['Có ý chính rõ ràng.'],
        'priorities': ['Sửa lỗi chia động từ.', 'Tăng liên kết giữa các câu.'],
    }


class AssessmentServiceTests(SimpleTestCase):
    def test_normalizes_scores_and_calculates_overall_score(self):
        payload = valid_payload()
        payload['criteria']['grammar']['score'] = -5
        payload['criteria']['vocabulary']['score'] = 120
        result = normalize_assessment(payload)
        self.assertEqual(result['criteria']['grammar']['score'], 0)
        self.assertEqual(result['criteria']['vocabulary']['score'], 100)
        self.assertEqual(result['overall_score'], 64)

    def test_rejects_incomplete_schema(self):
        payload = valid_payload()
        del payload['criteria']['coherence']
        with self.assertRaises(InvalidAssessmentResponse):
            normalize_assessment(payload)

    def test_rejects_invalid_provider_overall_score(self):
        payload = valid_payload()
        payload['overall_score'] = 'high'
        with self.assertRaises(InvalidAssessmentResponse):
            normalize_assessment(payload)

    @patch('api.services.assessment.get_gemini_client')
    def test_maps_timeout_without_fabricating_result(self, get_client):
        client = Mock()
        client.models.generate_content.side_effect = TimeoutError('timed out')
        get_client.return_value = client
        with self.assertRaises(AssessmentTimeout):
            assess_text_quality('word ' * 20)

    @patch('api.services.assessment.get_gemini_client')
    def test_rejects_invalid_json(self, get_client):
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(text='not json')
        get_client.return_value = client
        with self.assertRaises(InvalidAssessmentResponse):
            assess_text_quality('word ' * 20)


class AssessmentApiTests(APITestCase):
    def test_validates_text_bounds_and_types(self):
        cases = [
            ({'text': 123}, 'Invalid text format'),
            ({'text': 'too short'}, 'at least 20 words'),
            ({'text': 'word ' * 2001}, '10,000 characters'),
        ]
        for body, message in cases:
            with self.subTest(body=body):
                response = self.client.post('/api/assess/', body, format='json')
                self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
                self.assertIn(message, response.data['error'])

    @patch('api.views.assess_text_quality')
    def test_returns_mocked_assessment(self, assess):
        assess.return_value = normalize_assessment(valid_payload())
        response = self.client.post('/api/assess/', {
            'text': 'This is a sufficiently long sample paragraph with more than twenty words so the quality assessment endpoint can evaluate it safely today.',
            'language': 'en-US',
        }, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['success'])
        self.assertEqual(response.data['overall_score'], 78)
        self.assertEqual(set(response.data['criteria']), {
            'grammar', 'clarity', 'coherence', 'vocabulary',
        })

    @patch('api.views.assess_text_quality', side_effect=InvalidAssessmentResponse('bad'))
    def test_invalid_gemini_response_is_a_bad_gateway(self, _assess):
        response = self.client.post('/api/assess/', {'text': 'word ' * 20}, format='json')
        self.assertEqual(response.status_code, status.HTTP_502_BAD_GATEWAY)

    @patch('api.views.assess_text_quality', side_effect=AssessmentError('unavailable'))
    def test_service_failure_is_unavailable(self, _assess):
        response = self.client.post('/api/assess/', {'text': 'word ' * 20}, format='json')
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
