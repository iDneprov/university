CC = gcc
FLAGS = -pedantic -Wall -Werror -Wno-sign-compare -Wno-long-long -lm -O2

all: da4

da4: agalgorithm.o common.o ptread.o wordabc.o TPatricia.o main.o
	$(CC) $(FLAGS) $^ -o $@

clean:
	rm -f *.o da4