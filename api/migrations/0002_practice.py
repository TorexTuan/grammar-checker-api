import django.db.models.deletion
import uuid

from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('api', '0001_document'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='PracticeMistake',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('category', models.CharField(max_length=48)),
                ('sentence_context', models.CharField(max_length=500)),
                ('original', models.CharField(blank=True, max_length=300)),
                ('correction', models.CharField(blank=True, max_length=300)),
                ('explanation', models.TextField(blank=True)),
                ('fingerprint', models.CharField(max_length=64)),
                ('occurrence_count', models.PositiveIntegerField(default=1)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('last_seen_at', models.DateTimeField(auto_now=True)),
                ('document', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='practice_mistakes', to='api.document')),
                ('owner', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='practice_mistakes', to=settings.AUTH_USER_MODEL)),
            ],
            options={'ordering': ['-last_seen_at']},
        ),
        migrations.CreateModel(
            name='PracticeExercise',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('exercise_type', models.CharField(choices=[('multiple_choice', 'Multiple choice'), ('fill_blank', 'Fill in the blank')], max_length=24)),
                ('category', models.CharField(max_length=48)),
                ('prompt', models.CharField(max_length=800)),
                ('options', models.JSONField(default=list)),
                ('accepted_answers', models.JSONField(default=list)),
                ('explanation', models.TextField()),
                ('source', models.CharField(max_length=32)),
                ('fingerprint', models.CharField(max_length=64)),
                ('used_count', models.PositiveIntegerField(default=0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('mistake', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='exercises', to='api.practicemistake')),
                ('owner', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='practice_exercises', to=settings.AUTH_USER_MODEL)),
            ],
            options={'ordering': ['used_count', '-created_at']},
        ),
        migrations.CreateModel(
            name='PracticeSession',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('topic', models.CharField(max_length=48)),
                ('difficulty', models.CharField(default='medium', max_length=16)),
                ('total_questions', models.PositiveSmallIntegerField(default=5)),
                ('source_summary', models.JSONField(default=dict)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('completed_at', models.DateTimeField(blank=True, null=True)),
                ('owner', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='practice_sessions', to=settings.AUTH_USER_MODEL)),
            ],
            options={'ordering': ['-created_at']},
        ),
        migrations.CreateModel(
            name='PracticeSessionItem',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('order', models.PositiveSmallIntegerField()),
                ('user_answer', models.CharField(blank=True, max_length=500)),
                ('is_correct', models.BooleanField(blank=True, null=True)),
                ('answered_at', models.DateTimeField(blank=True, null=True)),
                ('exercise', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='session_items', to='api.practiceexercise')),
                ('session', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='items', to='api.practicesession')),
            ],
            options={'ordering': ['order']},
        ),
        migrations.AddConstraint(
            model_name='practicemistake',
            constraint=models.UniqueConstraint(fields=('owner', 'fingerprint'), name='unique_owner_practice_mistake'),
        ),
        migrations.AddIndex(
            model_name='practicemistake',
            index=models.Index(fields=['owner', 'category', '-last_seen_at'], name='practice_mistake_lookup'),
        ),
        migrations.AddIndex(
            model_name='practiceexercise',
            index=models.Index(fields=['owner', 'category', 'exercise_type', 'used_count'], name='practice_exercise_lookup'),
        ),
        migrations.AddConstraint(
            model_name='practicesessionitem',
            constraint=models.UniqueConstraint(fields=('session', 'order'), name='unique_practice_session_order'),
        ),
        migrations.AddConstraint(
            model_name='practicesessionitem',
            constraint=models.UniqueConstraint(fields=('session', 'exercise'), name='unique_practice_session_exercise'),
        ),
    ]
