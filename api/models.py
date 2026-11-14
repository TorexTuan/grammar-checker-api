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


class PracticeMistake(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='practice_mistakes',
    )
    category = models.CharField(max_length=48)
    concept_code = models.CharField(max_length=64, default='other_grammar')
    concept_label = models.CharField(max_length=120, default='Other grammar')
    source = models.CharField(max_length=24, default='legacy')
    sentence_context = models.CharField(max_length=500)
    original = models.CharField(max_length=300, blank=True)
    correction = models.CharField(max_length=300, blank=True)
    explanation = models.TextField(blank=True)
    fingerprint = models.CharField(max_length=64)
    occurrence_count = models.PositiveIntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)
    last_seen_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-last_seen_at']
        constraints = [
            models.UniqueConstraint(
                fields=['owner', 'fingerprint'],
                name='unique_owner_practice_mistake',
            ),
        ]
        indexes = [models.Index(fields=['owner', 'category', '-last_seen_at'], name='practice_mistake_lookup')]


class PracticeExercise(models.Model):
    TYPE_MULTIPLE_CHOICE = 'multiple_choice'
    TYPE_FILL_BLANK = 'fill_blank'
    TYPE_CHOICES = (
        (TYPE_MULTIPLE_CHOICE, 'Multiple choice'),
        (TYPE_FILL_BLANK, 'Fill in the blank'),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='practice_exercises',
    )
    mistake = models.ForeignKey(
        PracticeMistake,
        on_delete=models.SET_NULL,
        related_name='exercises',
        null=True,
        blank=True,
    )
    exercise_type = models.CharField(max_length=24, choices=TYPE_CHOICES)
    mode = models.CharField(max_length=24, default='legacy')
    difficulty = models.CharField(max_length=16, default='medium')
    category = models.CharField(max_length=48)
    prompt = models.CharField(max_length=800)
    hint = models.CharField(max_length=120, blank=True)
    options = models.JSONField(default=list)
    accepted_answers = models.JSONField(default=list)
    explanation = models.TextField()
    source = models.CharField(max_length=32)
    fingerprint = models.CharField(max_length=64)
    used_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['used_count', '-created_at']
        indexes = [models.Index(fields=['owner', 'category', 'exercise_type', 'used_count'], name='practice_exercise_lookup')]


class PracticeSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='practice_sessions',
    )
    topic = models.CharField(max_length=48)
    mode = models.CharField(max_length=24, default='legacy')
    difficulty = models.CharField(max_length=16, default='medium')
    focus_concepts = models.JSONField(default=list)
    total_questions = models.PositiveSmallIntegerField(default=5)
    source_summary = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']


class PracticeSessionItem(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        PracticeSession,
        on_delete=models.CASCADE,
        related_name='items',
    )
    exercise = models.ForeignKey(
        PracticeExercise,
        on_delete=models.PROTECT,
        related_name='session_items',
    )
    order = models.PositiveSmallIntegerField()
    user_answer = models.CharField(max_length=500, blank=True)
    is_correct = models.BooleanField(null=True, blank=True)
    answered_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['order']
        constraints = [
            models.UniqueConstraint(fields=['session', 'order'], name='unique_practice_session_order'),
            models.UniqueConstraint(fields=['session', 'exercise'], name='unique_practice_session_exercise'),
        ]


class PracticeProfile(models.Model):
    LEVEL_CHOICES = (('easy', 'Easy'), ('medium', 'Medium'), ('hard', 'Hard'))

    owner = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='practice_profile',
    )
    general_level = models.CharField(max_length=16, choices=LEVEL_CHOICES, default='easy')
    level_answered = models.PositiveSmallIntegerField(default=0)
    level_correct = models.PositiveSmallIntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)
