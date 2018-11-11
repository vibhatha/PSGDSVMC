clean:
	rm -rf exe/*

main:
	g++ --std=c++11 tutorial/main.cpp -o exe/main.out

runmain:
	./exe/main.out

hello:
	g++ --std=c++11 tutorial/main.cpp -o exe/main.out

runhello:
	./exe/main.out;

namespaces:
	g++ --std=c++11 tutorial/namespacesexample.cpp -o exe/namespaces.out

runnamespaces:
	./exe/namespaces.out

test:
	g++ --std=c++11 tutorial/test.cpp -o exe/test.out

runtest:
	./exe/test.out

cbint:
	g++ --std=c++11 -c tutorial/BinaryTree.cpp -o exe/binarytree.o

cnode:
	g++ --std=c++11 -c tutorial/Node.cpp -o exe/node.o

cedge:
	g++ --std=c++11 -c tutorial/Edge.cpp -o exe/edge.o

ctree:
	g++ --std=c++11 -c tutorial/Tree.cpp -o exe/tree.o

cmain:
	g++ --std=c++11 exe/*.o -o exe/main.out tutorial/main.cpp

crand:
	g++ --std=c++11 -c tutorial/GenerateRandomPoints.cpp -o exe/generaterandompoints.o

genclass:
	make cnode;make cedge;make ctree;make cbint;make crand;make cmain;

genrrt:
	g++ --std=c++11 rrt/main.cpp -o exe/rrt.exe

runrrt:
	./exe/rrt.exe
