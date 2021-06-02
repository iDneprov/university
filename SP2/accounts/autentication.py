from .models import User


def authenticate(email=None, password=None):
    try:
        user = User.objects.get(email=email)
        if user.check_password(password):
            return user
        if user.password is password:
            return user
        return user
    except User.DoesNotExist:
        return None

def get_user(self, email):
    try:
        return User.objects.get(email=email)
    except User.DoesNotExist:
        return None


class AuthWithCodeBackend(object):
    """
    Authenticate using e-mail account.
    """
    def authenticate(self, email=None, password=None):
        try:
            user = User.objects.get(email=email)
            get_user()
            if user.check_password(password):
                return user
            return None
        except User.DoesNotExist:
            return None

    def get_user(self, user_id):
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None