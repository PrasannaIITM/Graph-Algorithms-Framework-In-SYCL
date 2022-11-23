# Graph Algorithms Framework in SYCL

A large number of problems in the real world can be modelled as graph algorithms, for example, social networks, road connections, molecular interactions, planetary forces, etc. The execution time required for these algorithms is considerable for large graphs. They can be parallelised and sped up using GPUs. Once an algorithm is implemented for a specific backend, e.g. Nvidia GPUs in CUDA, it is not possible to run the same implementation of the algorithm on other GPUs like AMD or Intel. This is a problem as the same algorithm needs to be implemented and optimised in different languages depending on the available hardware to get the best performance.

* To enable the platform portability of graph algorithms by implementing them in SYCL.
* To optimise the SYCL code and compare the performance on large graphs with existing implementations.
* To create a framework that enables implementing graph algorithms quickly, hiding the complexities of SYCL syntax.
* To implement the following graph algorithms: single source shortest path, betweenness centrality, pagerank, minimum spanning tree, and triangle count using the developed framework.

[DPC++](https://github.com/intel/llvm), Intel's implementation of SYCL has been used in this project. 


## Installation Instructions

1. Clone the repo
2. Install SYCL locally: [Steps](https://intel.github.io/llvm-docs/GetStartedGuide.html#install-low-level-runtime) or Use [devcloud](<https://devcloud.intel.com/oneapi/>).  
3. Follow the relevant steps to compile the programs.
4. You can include the file `graph.h` in your custom applications to use the framework.

## Coverting graphs to CSR format
Use the `csr.py` script to convert a raw graph, represented as edge list, into CSR format.

Use the following command:
`python3 csr.py {graph-name}`

to convert a raw graph stored at the location `raw_graphs/{graph-name}.txt` into the CSR format. The output files(V, I, E, W, RI, RE) are stored inside the folder `csr_graphs/{graph-name}`.

## Single source shortest path

Implemented using two methods:
1. [Topology driven](https://github.com/PrasannaIITM/Graph-Algorithms-Framework-In-SYCL/blob/main/single_source_shortest_path/sssp_vtop.cpp):
2. [Data driven](https://github.com/PrasannaIITM/Graph-Algorithms-Framework-In-SYCL/blob/main/single_source_shortest_path/sssp_vdp.cpp)

After compiling the required file, use the following command to run the program:
   `./[executable] [graph-name] [source-vertex-id] [num-parallel-workers]`

The results and timing information are stored in `single_source_shortest_path/output` folder in the files `{graph-name}_sssp_{method}_results_{num-parallel-workers}.txt"` and `{graph-name}_sssp_{method}_time_{num-parallel-workers}.txt"` respectively.

## Betweenness centrality

## Pagerank

## Minimum Spanning Tree

## Triangle count

## Poster

![](https://github.com/PrasannaIITM/Graph-Algorithms-Framework-In-SYCL/blob/main/poster.png?raw=true)
