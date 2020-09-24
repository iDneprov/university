from random import random, randint, getrandbits

size = 1024
maxInt = 2**25
print(size)

for i in range(size):
    print((1 if getrandbits(1) else -1) * (randint(0, maxInt) + random()), end=' ')
print()
