from django.db import models
from django.contrib.auth.models import (
    BaseUserManager, AbstractBaseUser
)
import pyotp


class UserManager(BaseUserManager):
    def create_user(self, email=None, password=None):
        """
        Creates and saves a User with the given email and password.
        """
        if not email:
            raise ValueError('Введите email')

        if not password:
            raise ValueError('Введите пароль')

        user = self.model(
            email=self.normalize_email(email),
            auth_code=pyotp.random_base32(),
        )

        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_staffuser(self, email, password):
        """
        Creates and saves a staff user with the given email and password.
        """
        user = self.create_user(
            email,
            password=password,
            auth_code=pyotp.random_base32(),
        )
        user.staff = True
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password):
        """
        Creates and saves a superuser with the given email and password.
        """
        user = self.create_user(
            email,
            password=password,
            #auth_code=pyotp.random_base32(),
        )
        user.staff = True
        user.admin = True
        user.save(using=self._db)
        return user


class User(AbstractBaseUser):
    email = models.EmailField(
        verbose_name='email',
        max_length=255,
        unique=True,
    )
    active = models.BooleanField(default=True)
    staff = models.BooleanField(default=False)  # a admin user; non super-user
    admin = models.BooleanField(default=False)  # a superuser
    auth_code = models.CharField(max_length=20)
    # user_id = models.Index()
    # notice the absence of a "Password field", that's built in.

    USERNAME_FIELD = 'email'
    # REQUIRED_FIELDS = []
    # Email & Password are required by default.

    objects = UserManager()

    def get_full_name(self):
        # The user is identified by their email address
        return self.email

    def get_short_name(self):
        # The user is identified by their email address
        return self.email

    def __str__(self):  # __unicode__ on Python 2
        return self.email

    def get_auth_code(self):
        # The user is identified by their email address
        return self.auth_code

    def has_perm(self, perm, obj=None):
        "Есть ли у пользователя специальное разрешение?"
        return True

    def has_module_perms(self, app_label):
        "Есть ли у пользователя разрешение видеть `app_label`?"
        return True

    @property
    def is_staff(self):
        "Является ли пользователь сотрудником?"
        return self.staff

    @property
    def is_admin(self):
        "Является ли пользователь администратором?"
        return self.admin

    @property
    def is_active(self):
        """Активен ли пользователь?"""
        return self.active
