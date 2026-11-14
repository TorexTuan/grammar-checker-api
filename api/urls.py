from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .auth_views import csrf_token, login_view, logout_view, me, register
from .document_views import DocumentViewSet
from .practice_views import (
    answer_practice_session,
    create_practice_session,
    practice_overview,
    practice_progress,
    record_mistake,
    retrieve_practice_session,
)
from .views import (
    accept_autocomplete,
    ai_detect,
    assess_quality,
    autocomplete,
    check_grammar,
    health_check,
    rephrase,
)

router = DefaultRouter()
router.register('documents', DocumentViewSet, basename='document')

urlpatterns = [
    path('health/', health_check, name='health-check'),
    path('ai-detect/', ai_detect, name='ai-detect'),
    path('check/', check_grammar, name='check-grammar'),
    path('assess/', assess_quality, name='assess-quality'),
    path('autocomplete/', autocomplete, name='autocomplete'),
    path('autocomplete/accept/', accept_autocomplete, name='accept-autocomplete'),
    path('rephrase/', rephrase, name='rephrase'),
    path('auth/csrf/', csrf_token, name='auth-csrf'),
    path('auth/register/', register, name='auth-register'),
    path('auth/login/', login_view, name='auth-login'),
    path('auth/logout/', logout_view, name='auth-logout'),
    path('auth/me/', me, name='auth-me'),
    path('practice/mistakes/', record_mistake, name='practice-mistakes'),
    path('practice/overview/', practice_overview, name='practice-overview'),
    path('practice/progress/', practice_progress, name='practice-progress'),
    path('practice/sessions/', create_practice_session, name='practice-sessions'),
    path('practice/sessions/<uuid:session_id>/', retrieve_practice_session, name='practice-session-detail'),
    path('practice/sessions/<uuid:session_id>/answer/', answer_practice_session, name='practice-session-answer'),
    path('', include(router.urls)),
]
