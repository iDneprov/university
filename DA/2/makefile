CC = gcc
FLAGS = -pedantic -Wall -Werror -Wno-sign-compare -Wno-long-long -lm -O2
#-pedantic -Wall -std=c99 -Werror -Wno-sign-compare -lm

all: da2

da2: TPatricia.o main.o
	$(CC) $(FLAGS) $^ -o $@

clean:
	rm -f *.o da2