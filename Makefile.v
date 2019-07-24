# This is a makefile.

PROG = openmP
OBJ = main.o ArgReader.o DataSet.o Initializer.o Matrix.o OptArgs.o Predict.o PSGD.o ResourceManager.o SGD.o Test.o Util.o Matrix1.o

#flags. -I /home/vibhatha/tools/blass/build/include/ -L/home/vibhatha/tools/blass/build/lib -lopenblas
CC = mpic++ -std=c++11
LFLAG = -larmadillo -lcblas -ftree-vectorize -ftree-vectorizer-verbose=2 -msse -ffast-math
CFLAG = -c
OFLAG = -o
GFLAG = -g

all: $(PROG)

$(PROG) : $(OBJ)
	$(CC) $(GFLAG) $(OFLAG) $(PROG) $(OBJ) $(LFLAG)
%.o : %.cpp
	$(CC) -c $(CFLAG)  $< -o $@

clean:
	rm -f *.o


#-I /home/vibhatha/tools/blass/build/include/ -L /home/vibhatha/tools/blass/build/lib/