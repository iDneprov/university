from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.views.generic.base import View
from django.contrib.auth import login
from .forms import User, RegisterForm, AuthenticationForm
from .autentication import authenticate, get_user
from django.views.generic.edit import FormView
from .models import UserManager
from django.contrib.auth.forms import UserCreationForm
import qrcode
import pyotp
from PIL import Image
from django.contrib.auth import login
from accounts.autentication import authenticate
from p1.settings import STATIC_ROOT
from django.contrib.auth import logout

def logout_view(request):
    # if not request.user.is_anonymous:
    logout(request)
    HttpResponseRedirect('/')



def auth_code(request):
    user = request.user

    if user.id == None:
        return HttpResponseRedirect('/error/')

    if user.auth_code is '':
        user.auth_code=pyotp.random_base32()
        user.save()


    # Create qr code instance
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )

    # The data that you want to store
    data = "otpauth://totp/" + user.email + "?secret=" + user.auth_code + "&issuer=Practice"

    # Add data
    qr.add_data(data)
    qr.make(fit=True)

    # Create an image from the QR Code instance
    img = qr.make_image()
    img.save(STATIC_ROOT+ "/accounts/" + user.auth_code + ".jpg")
    return render(request, 'accounts/auth-code.html', context={'email': user.email, 'auth_code': user.auth_code,
                                                               'qrcode': "accounts/" +  user.auth_code + ".jpg"})


class RegisterFormView(FormView):
    form_class = RegisterForm#({'auth_code': pyotp.random_base32()})
    #form_class.fields['auth_code'].initial = pyotp.random_base32()

    # Ссылка, на которую будет перенаправляться пользователь в случае успешной регистрации.
    # В данном случае указана ссылка на страницу входа для зарегистрированных пользователей.
    success_url = "/auth-code/"

    # Шаблон, который будет использоваться при отображении представления.
    template_name = "accounts/register.html"

    '''def __init__(self, *args, **kwargs):
        super(RegisterFormView, self).__init__(*args, **kwargs)
        self.fields['auth_code'].initial = pyotp.random_base32()'''

    def form_valid(self, form):
        '''userManager = UserManager()
        email = form.cleaned_data.get("email")
        password = form.cleaned_data.get("password")
        userManager.create_user(self, email=email, password=password)'''
        # Создаём пользователя, если данные в форму были введены корректно.
        #self.fields['auth_code'].initial = pyotp.random_base32()
        form.save() # тут добавить генерацию ключа
        email, password = form.cleaned_data.get('email'), form.cleaned_data.get('password')
        user = authenticate(email=email, password=password)
        login(self.request, user)
        # Вызываем метод базового класса
        return super(RegisterFormView, self).form_valid(form)


class LoginFormView(FormView):
    form_class = AuthenticationForm

    # Аналогично регистрации, только используем шаблон аутентификации.
    template_name = "accounts/login.html"

    # В случае успеха перенаправим на главную.
    success_url = "/users/"

    def form_valid(self, form):
        valid = super(LoginFormView, self).form_valid(form)
        email, password, auth_code = form.cleaned_data.get('email'), form.cleaned_data.get('password'), form.cleaned_data.get('auth_code')
        user_info = User.objects.get(email=email)
        totp = pyotp.TOTP(user_info.auth_code)
        if totp.verify(auth_code):
            user = authenticate(email=email, password=password)
            request = self.request
            login(self.request, user)
        else:
            raise ValueError('Временный код не совпадает!')
        return valid
