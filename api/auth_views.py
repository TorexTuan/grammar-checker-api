import json

from django.contrib.auth import authenticate, get_user_model, login, logout
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.views.decorators.csrf import csrf_protect, ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST


User = get_user_model()


def _json_body(request):
    try:
        data = json.loads(request.body or '{}')
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _user_payload(user):
    return {'id': user.pk, 'email': user.email}


def _normalize_email(value):
    return User.objects.normalize_email(value.strip()).lower()


@require_GET
@ensure_csrf_cookie
def csrf_token(request):
    return JsonResponse({'csrfToken': get_token(request)})


@require_POST
@csrf_protect
def register(request):
    data = _json_body(request)
    if data is None:
        return JsonResponse({'error': 'Invalid JSON body.'}, status=400)

    email = _normalize_email(data.get('email', '')) if isinstance(data.get('email'), str) else ''
    password = data.get('password', '')
    errors = {}
    try:
        validate_email(email)
        if len(email) > User._meta.get_field('username').max_length:
            raise ValidationError('Email address is too long.')
    except ValidationError as exc:
        errors['email'] = list(exc.messages)
    if email and User.objects.filter(username__iexact=email).exists():
        errors['email'] = ['An account with this email already exists.']
    if not isinstance(password, str):
        errors['password'] = ['Enter a valid password.']
    else:
        try:
            validate_password(password)
        except ValidationError as exc:
            errors['password'] = list(exc.messages)
    if errors:
        return JsonResponse({'errors': errors}, status=400)

    user = User.objects.create_user(username=email, email=email, password=password)
    login(request, user)
    return JsonResponse({'user': _user_payload(user)}, status=201)


@require_POST
@csrf_protect
def login_view(request):
    data = _json_body(request)
    if data is None:
        return JsonResponse({'error': 'Invalid JSON body.'}, status=400)
    email = _normalize_email(data.get('email', '')) if isinstance(data.get('email'), str) else ''
    password = data.get('password', '')
    user = authenticate(request, username=email, password=password)
    if user is None:
        return JsonResponse({'error': 'Invalid email or password.'}, status=400)
    if not user.is_active:
        return JsonResponse({'error': 'This account is disabled.'}, status=403)
    login(request, user)
    return JsonResponse({'user': _user_payload(user)})


@require_POST
@csrf_protect
def logout_view(request):
    logout(request)
    return JsonResponse({}, status=200)


@require_GET
def me(request):
    return JsonResponse({'user': _user_payload(request.user) if request.user.is_authenticated else None})


def csrf_failure(request, reason=''):
    return JsonResponse({'error': 'CSRF verification failed. Refresh the page and try again.'}, status=403)
