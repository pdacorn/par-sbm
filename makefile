clean:
	rm -f ./parsbm

parsbm: clean
	mpicc  -std=c1x -o ./parsbm parsbm.c -lm


run:
	mpiexec -n 4 ./parsbm     > ./makeresult 2>&1 


