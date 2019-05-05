CC = nvcc

# compiler flags:
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
#  -std=c++11 enable the compiler and library support for the ISO C++ 2011 standard
#CFLAGS  = -Wall -g
# $(CFLAGS)

default: main

main:
	$(CC) main.cu -o output

clean: 
	$(RM) output *.o