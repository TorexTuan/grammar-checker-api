import uuid

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = [migrations.swappable_dependency(settings.AUTH_USER_MODEL)]

    operations = [
        migrations.CreateModel(
            name='Document',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('title', models.CharField(default='Untitled document', max_length=120)),
                ('content', models.JSONField(default=dict)),
                ('comments', models.JSONField(default=list)),
                ('version', models.PositiveIntegerField(default=1)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('owner', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='grammar_documents', to=settings.AUTH_USER_MODEL)),
            ],
            options={'ordering': ['-updated_at']},
        ),
        migrations.AddIndex(
            model_name='document',
            index=models.Index(fields=['owner', '-updated_at'], name='api_documen_owner_i_7b4e36_idx'),
        ),
    ]
