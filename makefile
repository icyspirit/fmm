
CC = mpicxx

all:
	$(CC) -std=c++17 -Wall -Wextra -Wno-missing-braces -fopenmp -O3 -ffast-math -march=native main.cpp

clean:
	rm -f a.out
