from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ('api', '0003_practiceexercise_hint'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AddField('practicemistake', 'concept_code', models.CharField(default='other_grammar', max_length=64)),
        migrations.AddField('practicemistake', 'concept_label', models.CharField(default='Other grammar', max_length=120)),
        migrations.AddField('practicemistake', 'source', models.CharField(default='legacy', max_length=24)),
        migrations.RemoveField('practicemistake', 'document'),
        migrations.AddField('practiceexercise', 'mode', models.CharField(default='legacy', max_length=24)),
        migrations.AddField('practiceexercise', 'difficulty', models.CharField(default='medium', max_length=16)),
        migrations.AddField('practicesession', 'mode', models.CharField(default='legacy', max_length=24)),
        migrations.AddField('practicesession', 'focus_concepts', models.JSONField(default=list)),
        migrations.CreateModel(
            name='PracticeProfile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('general_level', models.CharField(choices=[('easy', 'Easy'), ('medium', 'Medium'), ('hard', 'Hard')], default='easy', max_length=16)),
                ('level_answered', models.PositiveSmallIntegerField(default=0)),
                ('level_correct', models.PositiveSmallIntegerField(default=0)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('owner', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='practice_profile', to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
