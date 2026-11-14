from django.db import migrations, models


HINTS = {
    ('verb_tense', 'visited'): 'visit',
    ('verb_tense', 'read'): 'read',
    ('subject_verb_agreement', 'barks'): 'bark',
    ('subject_verb_agreement', 'are'): 'be',
    ('articles', 'the'): 'article',
    ('articles', 'a'): 'article',
    ('prepositions', 'in'): 'preposition',
    ('prepositions', 'at'): 'preposition',
    ('pronouns', 'me'): 'pronoun',
    ('pronouns', 'themselves'): 'pronoun',
    ('word_form', 'quickly'): 'quick',
    ('word_form', 'impressive'): 'impress',
    ('sentence_structure', 'but'): 'conjunction',
    ('sentence_structure', 'so that'): 'conjunction',
    ('spelling_punctuation', 's'): 'necessary',
    ('spelling_punctuation', '?'): 'punctuation',
    ('other_grammar', 'drink'): 'drink',
    ('other_grammar', 'are'): 'be',
}


def populate_hints(apps, schema_editor):
    PracticeExercise = apps.get_model('api', 'PracticeExercise')
    for exercise in PracticeExercise.objects.filter(exercise_type='fill_blank'):
        answers = exercise.accepted_answers if isinstance(exercise.accepted_answers, list) else []
        answer = str(answers[0]).strip().casefold() if answers else ''
        exercise.hint = HINTS.get((exercise.category, answer), 'base word')
        exercise.save(update_fields=['hint'])


class Migration(migrations.Migration):
    dependencies = [('api', '0002_practice')]

    operations = [
        migrations.AddField(
            model_name='practiceexercise',
            name='hint',
            field=models.CharField(blank=True, max_length=120),
        ),
        migrations.RunPython(populate_hints, migrations.RunPython.noop),
    ]
