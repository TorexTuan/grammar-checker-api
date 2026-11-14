from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('api', '0004_practice_two_modes'),
    ]

    operations = [
        migrations.AddField(
            model_name='practiceexercise',
            name='generation_key',
            field=models.CharField(blank=True, db_index=True, default='', max_length=128),
        ),
    ]
