import os
from unittest.mock import Mock, patch

from rest_framework import status
from rest_framework.test import APITestCase

from .services.sapling import SaplingAutocompleteTimeout, autocomplete_with_sapling


class AutocompleteApiTests(APITestCase):
    @patch.dict(os.environ, {'SAPLING_API_KEY': 'test-key'})
    @patch('api.services.sapling._http_session.post')
    def test_retries_one_transient_provider_error(self, post):
        unavailable = Mock(status_code=502)
        success = Mock(status_code=200)
        success.json.return_value = {
            'output': ' there',
            'query': 'Hello',
            'hash': 'e3fffa5f-1f11-531e-8a8e-d9a758677e0c',
        }
        post.side_effect = [unavailable, success]

        result = autocomplete_with_sapling('Hello')

        self.assertEqual(result['output'], ' there')
        self.assertEqual(post.call_count, 2)

    @patch.dict(os.environ, {'SAPLING_API_KEY': 'test-key'})
    @patch('api.services.sapling._http_session.post')
    def test_empty_provider_list_means_no_suggestion(self, post):
        response = Mock(status_code=200)
        response.json.return_value = []
        post.return_value = response

        result = autocomplete_with_sapling('A complete sentence.')

        self.assertEqual(result, {
            'output': '',
            'query': 'A complete sentence.',
            'hash': '',
        })

    @patch('api.views.autocomplete_with_sapling')
    def test_returns_completion(self, complete):
        complete.return_value = {
            'output': ' you need anything else',
            'query': 'Let us know if',
            'hash': 'e3fffa5f-1f11-531e-8a8e-d9a758677e0c',
        }
        response = self.client.post('/api/autocomplete/', {
            'query': 'Let us know if',
            'session_id': 'document-123',
        }, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['success'])
        self.assertEqual(response.data['output'], ' you need anything else')
        complete.assert_called_once_with('Let us know if', 'document-123')

    @patch('api.views.autocomplete_with_sapling')
    def test_validates_query(self, complete):
        for query in ('', 'x' * 201, None):
            with self.subTest(query=query):
                response = self.client.post('/api/autocomplete/', {'query': query}, format='json')
                self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        complete.assert_not_called()

    @patch('api.views.autocomplete_with_sapling', side_effect=SaplingAutocompleteTimeout('slow'))
    def test_maps_timeout(self, _complete):
        response = self.client.post('/api/autocomplete/', {'query': 'Hello there'}, format='json')
        self.assertEqual(response.status_code, status.HTTP_504_GATEWAY_TIMEOUT)

    def test_missing_api_key_is_unavailable(self):
        with patch.dict(os.environ, {'SAPLING_API_KEY': ''}):
            response = self.client.post(
                '/api/autocomplete/', {'query': 'Hello there'}, format='json'
            )
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)

    @patch('api.views.accept_sapling_completion')
    def test_records_accepted_completion(self, accept):
        payload = {
            'hash': 'e3fffa5f-1f11-531e-8a8e-d9a758677e0c',
            'query': 'Let us know if',
            'completion': ' you need anything else',
            'session_id': 'document-123',
        }
        response = self.client.post('/api/autocomplete/accept/', payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        accept.assert_called_once_with(
            payload['hash'], payload['query'], payload['completion'], payload['session_id']
        )
