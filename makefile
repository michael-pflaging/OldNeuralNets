CC = g++ -g
EXE = net

$(EXE): neural_net.o net_tester.o main.o
	$(CC) neural_net.o net_tester.o main.o -o $(EXE)

neural_net.o: neural_net.cpp
	$(CC) -c neural_net.cpp

net_tester.o: net_tester.cpp
	$(CC) -c net_tester.cpp

main.o: main.cpp
	$(CC) -c main.cpp 

clean: 
	rm -f *.o $(EXE)

leak_check: neural_net.o net_tester.o main.o
	$(CC) neural_net.o net_tester.o main.o -o $(EXE)
	valgrind --leak-check=full ./$(EXE)