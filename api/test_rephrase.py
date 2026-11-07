import os
from unittest.mock import Mock, patch

import requests
from rest_framework import status
from rest_framework.test import APITestCase

from .services.sapling import (
    InvalidSaplingRephraseResponse,
    SaplingRephraseError,
    SaplingRephraseRateLimit,
    SaplingRephraseTimeout,
    rephrase_with_sapling,
)


def provider_response(status_code=200, payload=None, headers=None):
    response = Mock(status_code=status_code, headers=headers or {})
    response.json.return_value = payload
    return response


class RephraseServiceTests(APITestCase):
    @patch.dict(os.environ, {'SAPLING_API_KEY': 'test-key'})
    @patch('api.services.sapling._http_session.post')
    def test_uses_professional_tone_mapping(self, post):
        post.return_value = provider_response(payload={
            'results': [{'hash': 'result-1', 'replacement': 'Please review this document.'}],
        })

        results = rephrase_with_sapling('Look at this document.', 3)

        self.assertEqual(results, [
            {'id': 'result-1', 'text': 'Please review this document.'},
        ])
        _, kwargs = post.call_args
        self.assertEqual(kwargs['json'], {
            'text': 'Look at this document.',
            'mapping': 'switch_tone',
            'tone_mapping': 'professional',
            'num_results': 3,
        })
        self.assertEqual(kwargs['headers']['Authorization'], 'Bearer test-key')

    @patch.dict(os.environ, {'SAPLING_API_KEY': 'test-key'})
    @patch('api.services.sapling._http_session.post')
    def test_deduplicates_and_limits_results(self, post):
        post.return_value = provider_response(payload={'results': [
            {'hash': '1', 'replacement': ' First result '},
            {'hash': '2', 'replacement': 'first RESULT'},
            {'hash': '3', 'replacement': 'Second result'},
            {'hash': '4', 'replacement': 'Third result'},
            {'hash': '5', 'replacement': 'Fourth result'},
        ]})

        results = rephrase_with_sapling('Original text', 3)

        self.assertEqual([result['text'] for result in results], [
            'First result', 'Second result', 'Third result',
        ])

    @patch.dict(os.environ, {'SAPLING_API_KEY': 'test-key'})
    @patch('api.services.sapling._http_session.post')
    def test_retries_exactly_once_for_provider_5xx(self, post):
        post.side_effect = [
            provider_response(503, {'error': 'unavailable'}),
            provider_response(200, {'results': []}),
        ]

        self.assertEqual(rephrase_with_sapling('Original text'), [])
        self.assertEqual(post.call_count, 2)

    @patch.dict(os.environ, {'SAPLING_API_KEY': 'test-key'})
    @patch('api.services.sapling._http_session.post', side_effect=requests.Timeout('slow'))
    def test_does_not_retry_timeout(self, post):
        with self.assertRaises(SaplingRephraseTimeout):
            rephrase_with_sapling('Original text')
        self.assertEqual(post.call_count, 1)

    @patch.dict(os.environ, {'SAPLING_API_KEY': 'test-key'})
    @patch('api.services.sapling._http_session.post')
    def test_does_not_retry_rate_limit(self, post):
        post.return_value = provider_response(429, {'error': 'limited'}, {'Retry-After': '8'})

        with self.assertRaises(SaplingRephraseRateLimit) as raised:
            rephrase_with_sapling('Original text')

        self.assertEqual(raised.exception.retry_after, '8')
        self.assertEqual(post.call_count, 1)

    @patch.dict(os.environ, {'SAPLING_API_KEY': 'test-key'})
    @patch('api.services.sapling._http_session.post')
    def test_does_not_retry_payment_required(self, post):
        post.return_value = provider_response(402, {'error': 'credits depleted'})

        with self.assertRaises(SaplingRephraseError):
            rephrase_with_sapling('Original text')

        self.assertEqual(post.call_count, 1)

    @patch.dict(os.environ, {'SAPLING_API_KEY': 'test-key'})
    @patch('api.services.sapling._http_session.post')
    def test_rejects_malformed_response(self, post):
        post.return_value = provider_response(payload={'unexpected': []})

        with self.assertRaises(InvalidSaplingRephraseResponse):
            rephrase_with_sapling('Original text')


class RephraseApiTests(APITestCase):
    endpoint = '/api/rephrase/'

    @patch('api.views.rephrase_with_sapling')
    def test_returns_normalized_results(self, rephrase):
        rephrase.return_value = [{'id': 'result-1', 'text': 'A professional sentence.'}]
        payload = {'text': 'A casual sentence.', 'style': 'professional', 'num_results': 3}

        response = self.client.post(self.endpoint, payload, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data, {
            'success': True,
            'original': payload['text'],
            'results': rephrase.return_value,
        })
        rephrase.assert_called_once_with(payload['text'], 3)

    @patch('api.views.rephrase_with_sapling')
    def test_validates_text(self, rephrase):
        for text in ('', '   ', None, 123, 'x' * 4001):
            with self.subTest(text_type=type(text), text_length=len(text) if isinstance(text, str) else None):
                response = self.client.post(self.endpoint, {
                    'text': text,
                    'style': 'professional',
                    'num_results': 3,
                }, format='json')
                self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        rephrase.assert_not_called()

    @patch('api.views.rephrase_with_sapling')
    def test_validates_style_and_result_count(self, rephrase):
        invalid_payloads = [
            {'text': 'Text', 'style': 'casual', 'num_results': 3},
            {'text': 'Text', 'style': 'professional', 'num_results': True},
            {'text': 'Text', 'style': 'professional', 'num_results': 0},
            {'text': 'Text', 'style': 'professional', 'num_results': 4},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = self.client.post(self.endpoint, payload, format='json')
                self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        rephrase.assert_not_called()

    @patch('api.views.rephrase_with_sapling', side_effect=SaplingRephraseTimeout('slow'))
    def test_maps_timeout(self, _rephrase):
        response = self._post_valid()
        self.assertEqual(response.status_code, status.HTTP_504_GATEWAY_TIMEOUT)

    @patch(
        'api.views.rephrase_with_sapling',
        side_effect=SaplingRephraseRateLimit('limited', retry_after='12'),
    )
    def test_maps_rate_limit(self, _rephrase):
        response = self._post_valid()
        self.assertEqual(response.status_code, status.HTTP_429_TOO_MANY_REQUESTS)
        self.assertEqual(response['Retry-After'], '12')

    @patch(
        'api.views.rephrase_with_sapling',
        side_effect=InvalidSaplingRephraseResponse('bad shape'),
    )
    def test_maps_invalid_provider_response(self, _rephrase):
        response = self._post_valid()
        self.assertEqual(response.status_code, status.HTTP_502_BAD_GATEWAY)

    @patch('api.views.rephrase_with_sapling', side_effect=SaplingRephraseError('down'))
    def test_maps_other_provider_error(self, _rephrase):
        response = self._post_valid()
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)

    def _post_valid(self):
        return self.client.post(self.endpoint, {
            'text': 'Please rewrite this.',
            'style': 'professional',
            'num_results': 3,
        }, format='json')
