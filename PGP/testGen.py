from random import random, randint

size = 1024
maxInt = 2**25
print(size)

for i in range(size):
    print(randint(0, maxInt) + random(), end=' ')
print()
