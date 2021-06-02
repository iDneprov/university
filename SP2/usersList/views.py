from django.shortcuts import render
from accounts.models import User


# Create your views here.
def usersPrint(request):
    users = User.objects.all()
    active_user = request.user
    return render(request, 'usersList/index.html', context={'users': users, 'active_user': active_user})