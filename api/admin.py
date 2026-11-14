from django.contrib import admin

from .models import Document, PracticeExercise, PracticeMistake, PracticeProfile, PracticeSession, PracticeSessionItem


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'owner', 'version', 'updated_at')
    search_fields = ('title', 'owner__email')
    list_filter = ('created_at', 'updated_at')


admin.site.register(PracticeMistake)
admin.site.register(PracticeExercise)
admin.site.register(PracticeSession)
admin.site.register(PracticeSessionItem)
admin.site.register(PracticeProfile)

# Register your models here.
