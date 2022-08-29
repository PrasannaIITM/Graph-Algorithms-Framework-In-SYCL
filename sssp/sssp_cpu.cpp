#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <climits>
#include "utils.h"
#define DEBUG 0

int main()
{

    std::chrono::steady_clock::time_point tic_0 = std::chrono::steady_clock::now();
    std::string name = "USAud";
    std::cout << "Sequential implementation on CPU"
              << "\n ";

    std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
    std::vector<int> V, I, E, W;
    load_from_file("input/" + name + "/V", V);
    load_from_file("input/" + name + "/I", I);
    load_from_file("input/" + name + "/E", E);
    load_from_file("input/" + name + "/W", W);
    std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();

    std::cout << "Time to load data from files: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

    if (DEBUG)
    {
        std::cout << "Node: ";
        print_vector(V);
        std::cout << "Offset: ";
        print_vector(I);
        std::cout << "Edge: ";
        print_vector(E);
        std::cout << "Weight: ";
        print_vector(W);
    }

    int N = V.size();
    std::vector<int> D(N, INT_MAX);

    D[0] = 0;

    tic = std::chrono::steady_clock::now();
    int early_stop = 0;
    for (int round = 1; round < N; round++)
    {
        std::cout << "Round num: " << round << std::endl;
        if (early_stop == 1)
        {
            break;
        }
        early_stop = 1;
        for (int i = 0; i < I.size() - 1; i++)
        {
            for (int j = I[i]; j < I[i + 1]; j++)
            {
                int u = V[i];
                int v = V[E[j]];
                int w = W[j];
                int du = D[i];
                int dv = D[E[j]];
                if (du == INT_MAX)
                {
                    continue;
                }
                if (du + w < dv)
                {
                    D[E[j]] = du + w;
                    early_stop = 0;
                }
            }
        }
    }
    toc = std::chrono::steady_clock::now();
    std::cout << "Time to run SSSP: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

    tic = std::chrono::steady_clock::now();
    std::ofstream myfile;
    myfile.open("output/" + name + "/sssp_cpu_result.txt");

    for (int i = 0; i < N; i++)
    {
        myfile << i << " " << D[i] << std::endl;
    }
    myfile.close();
    toc = std::chrono::steady_clock::now();
    std::cout << "Time to write data to file: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

    std::chrono::steady_clock::time_point toc_0 = std::chrono::steady_clock::now();
    std::cout << "Total time taken: " << std::chrono::duration_cast<std::chrono::microseconds>(toc_0 - tic_0).count() << "[µs]" << std::endl;

    return 0;
}
