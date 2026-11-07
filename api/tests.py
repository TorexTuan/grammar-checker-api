from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient, APITestCase

from .models import Document


User = get_user_model()
DOC = {'type': 'doc', 'content': [{'type': 'paragraph', 'content': [{'type': 'text', 'text': 'Hello'}]}]}
COMMENT = {
    'id': 'comment-1',
    'from': 1,
    'to': 6,
    'quote': 'Hello',
    'note': 'Review this sentence.',
    'createdAt': '2026-11-03T00:00:00.000Z',
    'unlinked': False,
}


class AuthApiTests(APITestCase):
    def setUp(self):
        self.client = APIClient(enforce_csrf_checks=True)

    def csrf(self):
        response = self.client.get(reverse('auth-csrf'))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        token = response.json()['csrfToken']
        self.client.credentials(HTTP_X_CSRFTOKEN=token)

    def test_register_login_me_and_logout(self):
        self.csrf()
        response = self.client.post(reverse('auth-register'), {
            'email': 'Writer@Example.com',
            'password': 'StrongPass!2026',
        }, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.json()['user']['email'], 'writer@example.com')
        self.assertIsNotNone(self.client.get(reverse('auth-me')).json()['user'])

        self.csrf()
        response = self.client.post(reverse('auth-logout'), {}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsNone(self.client.get(reverse('auth-me')).json()['user'])

        self.csrf()
        response = self.client.post(reverse('auth-login'), {
            'email': 'WRITER@example.com',
            'password': 'StrongPass!2026',
        }, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_register_requires_csrf_and_unique_email(self):
        response = self.client.post(reverse('auth-register'), {
            'email': 'writer@example.com',
            'password': 'StrongPass!2026',
        }, format='json')
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

        self.csrf()
        first = self.client.post(reverse('auth-register'), {
            'email': 'writer@example.com',
            'password': 'StrongPass!2026',
        }, format='json')
        self.assertEqual(first.status_code, status.HTTP_201_CREATED)
        self.csrf()
        duplicate = self.client.post(reverse('auth-register'), {
            'email': 'WRITER@example.com',
            'password': 'StrongPass!2026',
        }, format='json')
        self.assertEqual(duplicate.status_code, status.HTTP_400_BAD_REQUEST)


class DocumentApiTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user('one@example.com', 'one@example.com', 'StrongPass!2026')
        self.other = User.objects.create_user('two@example.com', 'two@example.com', 'StrongPass!2026')
        self.client.force_authenticate(self.user)

    def test_document_crud_and_version_conflict(self):
        response = self.client.post('/api/documents/', {
            'title': 'My document', 'content': DOC, 'comments': [COMMENT],
        }, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        document_id = response.data['id']
        self.assertEqual(response.data['version'], 1)
        self.assertEqual(response.data['comments'], [COMMENT])

        retrieved = self.client.get(f'/api/documents/{document_id}/')
        self.assertEqual(retrieved.status_code, status.HTTP_200_OK)
        self.assertEqual(retrieved.data['comments'], [COMMENT])

        listing = self.client.get('/api/documents/')
        self.assertEqual(listing.status_code, status.HTTP_200_OK)
        self.assertEqual(len(listing.data), 1)
        self.assertNotIn('content', listing.data[0])

        updated = self.client.patch(f'/api/documents/{document_id}/', {
            'title': 'Updated', 'version': 1,
        }, format='json')
        self.assertEqual(updated.status_code, status.HTTP_200_OK)
        self.assertEqual(updated.data['version'], 2)

        conflict = self.client.patch(f'/api/documents/{document_id}/', {
            'title': 'Stale write', 'version': 1,
        }, format='json')
        self.assertEqual(conflict.status_code, status.HTTP_409_CONFLICT)
        self.assertEqual(conflict.data['code'], 'version_conflict')

        deleted = self.client.delete(f'/api/documents/{document_id}/')
        self.assertEqual(deleted.status_code, status.HTTP_204_NO_CONTENT)

    def test_documents_are_private_and_payload_is_validated(self):
        private = Document.objects.create(owner=self.other, title='Private', content=DOC)
        self.assertEqual(self.client.get(f'/api/documents/{private.id}/').status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(self.client.delete(f'/api/documents/{private.id}/').status_code, status.HTTP_404_NOT_FOUND)

        invalid = self.client.post('/api/documents/', {
            'title': 'Invalid', 'content': {'type': 'wrong'}, 'comments': {},
        }, format='json')
        self.assertEqual(invalid.status_code, status.HTTP_400_BAD_REQUEST)

    def test_documents_require_authentication(self):
        self.client.force_authenticate(user=None)
        self.assertIn(self.client.get('/api/documents/').status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_session_document_write_requires_csrf(self):
        client = APIClient(enforce_csrf_checks=True)
        client.force_login(self.user)
        payload = {'title': 'Protected', 'content': DOC, 'comments': []}
        self.assertEqual(client.post('/api/documents/', payload, format='json').status_code, status.HTTP_403_FORBIDDEN)

        csrf_response = client.get(reverse('auth-csrf'))
        client.credentials(HTTP_X_CSRFTOKEN=csrf_response.json()['csrfToken'])
        self.assertEqual(client.post('/api/documents/', payload, format='json').status_code, status.HTTP_201_CREATED)
