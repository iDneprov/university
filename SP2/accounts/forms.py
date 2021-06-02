from django import forms
from django.contrib.auth.forms import ReadOnlyPasswordHashField
from django.forms import EmailField
from django.contrib.auth import login

from .models import UserManager
from .autentication import *
import pyotp


class RegisterForm(forms.ModelForm):
    email = forms.CharField(label='Email', widget=forms.EmailInput)
    password = forms.CharField(label='Пароль', widget=forms.PasswordInput)
    password2 = forms.CharField(label='Подтвердите пароль', widget=forms.PasswordInput)
    #auth_code = forms.CharField(max_length=20)

    email.widget.attrs.update({'class':'form-control'})
    password.widget.attrs.update({'class': 'form-control'})
    password2.widget.attrs.update({'class': 'form-control'})

    #auth_code = pyotp.random_base32()

    class Meta:
        model = User
        fields = ('email', 'password','password2')
        #widgets = {'auth_code': forms.HiddenInput()}

    def clean_email(self):
        email = self.cleaned_data.get('email')
        qs = User.objects.filter(email=email)
        if qs.exists():
            raise forms.ValidationError("Пользователь с этой почтой уже зарегистрирован!")
        return email

    def clean_password2(self):
        password1 = self.cleaned_data.get("password")
        password2 = self.cleaned_data.get("password2")
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("Пароли не совпадают!")
        return password2

    '''def save(self):
        user = super().save(commit=True)
        user = self.model(
            email=self.normalize_email(email),
            auth_code=pyotp.random_base32(),
        )
        user.set_password(password)
        user.save(using=self._db)
        return user'''


    '''def save(self, commit=True):
        user = super().save(commit=commit)
        if commit:
            auth_user = authenticate(self, email=self.cleaned_data['email'], password=self.cleaned_data['password'])
            login(self, auth_user)

        return user'''


class AuthenticationForm(forms.Form):
    """
    Base class for authenticating users. Extend this to get a form that accepts
    username/password logins.
    """

    class Meta:
        model = User
        fields = ('email',)

    email = forms.CharField(label='Email', widget=forms.EmailInput)
    password = forms.CharField(label='Пароль', widget=forms.PasswordInput)
    auth_code = forms.CharField(label='Временный код', widget=forms.NumberInput)

    email.widget.attrs.update({'class': 'form-control'})
    password.widget.attrs.update({'class': 'form-control'})
    auth_code.widget.attrs.update({'class': 'form-control'})

    error_messages = {
        'invalid_login': "Введите корректный email и пароль",
        'inactive': "Аккаунт не активирован.",
        'code': "Временный код не подходит"
    }

    '''def get_user(self, user_id):
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None'''
    '''def clean_email(self):
        email = self.cleaned_data.get('email')
        qs = User.objects.filter(email=email)
        if not qs.exists():
            raise forms.ValidationError("Пользователя с данной почтой не существует!")
        return email'''

    def clean_password(self):
        email = self.cleaned_data.get('email')
        password = self.cleaned_data.get('password')

        if email is not None and password:
            self.user_cache = authenticate(email=email, password=password)
            p = self.user_cache.password
            if self.user_cache.password != password:
                raise forms.ValidationError("Введены неверные email и пароль!")
            else:
                self.confirm_login_allowed(self.user_cache)

        return self.cleaned_data

    def clean_auth_code(self):
        email = self.cleaned_data.get('email')
        auth_code = self.cleaned_data.get('auth_code')
        if User.objects.filter(email=email):
            user = User.objects.get(email=email)
            totp = pyotp.TOTP(user.auth_code)
            if not totp.verify(auth_code) and len(user.auth_code) is 16:
               raise forms.ValidationError("Временный код не подходит!")
            return auth_code

    def confirm_login_allowed(self, user):
        """
        Controls whether the given User may log in. This is a policy setting,
        independent of end-user authentication. This default behavior is to
        allow login by active users, and reject login by inactive users.

        If the given user cannot log in, this method should raise a
        ``forms.ValidationError``.

        If the given user may log in, this method should return None.
        """

        if not user.active:
            raise forms.ValidationError(
                self.error_messages['inactive'],
                code='inactive',
            )

    def get_invalid_login_error(self):
        return forms.ValidationError(
            self.error_messages['invalid_login'],
            code='invalid_login',
            # params={'email': self.email},
        )

    def get_user(self):
        try:
            return User.objects.get(email=self.cleaned_data.get('email'))
        except User.DoesNotExist:
            return None


class UserAdminCreationForm(forms.ModelForm):
    """A form for creating new users. Includes all the required
    fields, plus a repeated password."""
    password1 = forms.CharField(label='Пароль', widget=forms.PasswordInput)
    password2 = forms.CharField(label='Подтверждение пароля', widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ('email',)

    def confirm_login_allowed(self, user):
        if not user.is_active:
            raise forms.ValidationError(
                "This account is inactive.",
                code='inactive',
            )


class UserAdminChangeForm(forms.ModelForm):
    """A form for updating users. Includes all the fields on
    the user, but replaces the password field with admin's
    password hash display field.
    """
    password = ReadOnlyPasswordHashField()

    class Meta:
        model = User
        fields = ('email', 'password', 'active', 'admin')

    def clean_password(self):
        # Regardless of what the user provides, return the initial value.
        # This is done here, rather than on the field, because the
        # field does not have access to the initial value
        return self.initial["password"]
