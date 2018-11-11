# This is a makefile.

PROG = openmP
OBJ = main.o ArgReader.o DataSet.o Initializer.o Matrix.o OptArgs.o Predict.o PSGD.o ResourceManager.o SGD.o Test.o Util.o

#flags.
CC = mpic++ -std=c++11
LFLAG =
CFLAG = -c
OFLAG = -o

all: $(PROG)

$(PROG) : $(OBJ)
	$(CC) $(OFLAG) $(PROG) $(OBJ) $(LFLAG)
%.o : %.cpp
	$(CC) -c $(CFLAG) $< -o $@

clean:
	rm -f *.o
