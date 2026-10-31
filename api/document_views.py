from django.core.exceptions import ValidationError
from django.db import transaction
from rest_framework import status, viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import Document
from .serializers import DocumentListSerializer, DocumentSerializer


class DocumentViewSet(viewsets.ModelViewSet):
    permission_classes = [IsAuthenticated]
    http_method_names = ['get', 'post', 'patch', 'delete', 'head', 'options']

    def get_queryset(self):
        if not self.request.user.is_authenticated:
            return Document.objects.none()
        return Document.objects.filter(owner=self.request.user)

    def get_serializer_class(self):
        return DocumentListSerializer if self.action == 'list' else DocumentSerializer

    def partial_update(self, request, *args, **kwargs):
        expected_version = request.data.get('version')
        if not isinstance(expected_version, int) or isinstance(expected_version, bool):
            return Response(
                {'version': ['A numeric version is required.']},
                status=status.HTTP_400_BAD_REQUEST,
            )

        with transaction.atomic():
            try:
                document = self.get_queryset().select_for_update().filter(pk=kwargs['pk']).first()
            except (ValueError, ValidationError):
                document = None
            if document is None:
                return Response(status=status.HTTP_404_NOT_FOUND)
            if document.version != expected_version:
                return Response(
                    {
                        'error': 'This document has changed since it was opened.',
                        'code': 'version_conflict',
                        'document': DocumentSerializer(document, context={'request': request}).data,
                    },
                    status=status.HTTP_409_CONFLICT,
                )

            serializer = DocumentSerializer(
                document,
                data=request.data,
                partial=True,
                context={'request': request},
            )
            serializer.is_valid(raise_exception=True)
            serializer.save()
            return Response(serializer.data)
