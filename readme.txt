-----------------------------------------------------------------------------

Parallel Community Detection using Stochastic Block Models (Par-SBM)
Version 0.1

This program is to identify non-overlapping communities for undirected networks. It provides good solutions for the input graphs that fit the stochastic block models (SBMs). Comparing with non-model based algorithms, it can provide a meaningful alternative solution for general input graphs. Comparing with other SBM-based algorithms, it runs faster on one processor and is more scalable on multi-processor systems.  

This program has been tested on Red Hat Enterprise Linux 6.3 with mpi-openmpi version 1.4.3 and gcc version 4.6.0. If you have found a bug or have problems, you may contact the authors in the citation below. 

Citing Par-SBM: Chengbin Peng, Zhihua Zhang, Ka-Chun Wong, Xiangliang Zhang, David Keyes, "A scalable community detection algorithm for large graphs using stochastic block models", Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI), Buenos Aires, Argentina, 2015.

-----------------------------------------------------------------------------

How to Use the Code

1. Prepare the data
The graph data should be in the edge-list format. In this format, the data file contains two columns of integers, and each row contains two integers representing nodes on the two ends of each edge. The node IDs should start from zero to N-1, where N is the total number of nodes, and N should not be larger than the maximum value for an "int". Edges are considered to be undirected automatically. 

2. Prepare conf.txt
Text file conf.txt is a configuration file, defining several variables, one at a line. Lines starting with "#" will be ignored by the algorithm. In the configuration file, there are two mandatory variables: "netPath" is the path of the input network; and "outputPath" is the folder to store the output result. Two other variables are determined by our algorithm by default, but the user can specify them as well: "Kori" is the number of communities in the final result; "Kinit" is the number of communities in the algorithm initialization.

3. Compile the source code and run
The user may compile and run using the following commands:
make parsbm
make run

4. Get the output files
There are several output files. In the program folder, "makeresult" is a report for the running, containing iteration numbers, data loading and running times, etc. "resultZ.dat" reports the community label for each node, and integers in the first column contains the node IDs. "resultB.dat" reports the edge connectivity probability for each community, and the first column contains the community IDs.  

5. Additional parameters
The parameters relevant to our papers are listed in the code section "/* parameters for the algorithm */". All the other parameters are beyond the scope of our paper, and we make them easily tunable only for the ease of users to play with the code.

-----------------------------------------------------------------------------


