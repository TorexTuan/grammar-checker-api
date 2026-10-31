import uuid

from django.conf import settings
from django.db import models


class Document(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='grammar_documents',
    )
    title = models.CharField(max_length=120, default='Untitled document')
    content = models.JSONField(default=dict)
    comments = models.JSONField(default=list)
    version = models.PositiveIntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['owner', '-updated_at'], name='api_documen_owner_i_7b4e36_idx'),
        ]

    def __str__(self):
        return f'{self.title} ({self.owner_id})'
