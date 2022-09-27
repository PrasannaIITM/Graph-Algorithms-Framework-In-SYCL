#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>
#define DEBUG 0
#define NUM_THREADS 1024

using namespace sycl;

int main()
{

    std::chrono::steady_clock::time_point tic_0 = std::chrono::steady_clock::now();
    std::string name = "wikiud";
    std::cout << "Processing " << name << std::endl;
    // The default device selector will select the most performant device.
    default_selector d_selector;
    queue Q(d_selector);
    std::cout << "Selected device: " << Q.get_device().get_info<info::device::name>() << "\n";

    std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
    std::vector<int> V, I, E, RE, RI;
    load_from_file("input/" + name + "/V", V);
    load_from_file("input/" + name + "/I", I);
    load_from_file("input/" + name + "/E", E);

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
    }
    int *dev_V = malloc_device<int>(V.size(), Q);
    int *dev_I = malloc_device<int>(I.size(), Q);
    int *dev_E = malloc_device<int>(E.size(), Q);

    Q.submit([&](handler &h)
             { h.memcpy(dev_V, &V[0], V.size() * sizeof(int)); });

    Q.submit([&](handler &h)
             { h.memcpy(dev_I, &I[0], I.size() * sizeof(int)); });

    Q.submit([&](handler &h)
             { h.memcpy(dev_E, &E[0], E.size() * sizeof(int)); });

    Q.wait();

    int N = V.size();
    int stride = NUM_THREADS;

    tic = std::chrono::steady_clock::now();
    std::cout << "Starting triangle count..." << std::endl;
    int *triangle_count = malloc_shared<int>(1, Q);
    *triangle_count = 0;

    Q.submit([&](handler &h)
             { h.parallel_for(
                   NUM_THREADS, [=](id<1> i)
                   {
                            for (; i < N; i += stride)
                            {
                                for (int edge1 = dev_I[i]; edge1 < dev_I[i + 1]; edge1++)
                                {
                                    int u = dev_E[edge1];
                                    if(u < i){
                                        for (int edge2 = dev_I[i]; edge2 < dev_I[i + 1]; edge2++){
                                            int w = dev_E[edge2];
                                            int nbrs_connected = 0;
                                            if(w > i){
                                                
                                                int start_edge =  dev_I[u]; 
                                                int end_edge =  dev_I[u + 1] - 1; 

                                                if(dev_E[start_edge] == w){
                                                    nbrs_connected = 1;
                                                }
                                                else if(dev_E[end_edge] == 1){
                                                    nbrs_connected = 1;
                                                }
                                                else
                                                {    while(start_edge <= end_edge){
                                                        int mid = start_edge + (end_edge - start_edge)/2;
                                                        if(dev_E[mid] == w){
                                                            nbrs_connected = 1;
                                                            break;
                                                        }
                                                        if(w < dev_E[mid]){
                                                            end_edge = mid - 1;
                                                        }
                                                        if (w > dev_E[mid])
                                                        {
                                                            start_edge = mid + 1;
                                                        }
                                                    }
                                                }
                            
                                                }
                                                if(nbrs_connected){
                                                    atomic_ref<int, memory_order::seq_cst, memory_scope::device, access::address_space::global_space> atomic_data(*triangle_count);
                                                    atomic_data++;
                                                }
                                            }
                                        }
                                    }
                                } }); })
        .wait();

    toc = std::chrono::steady_clock::now();
    std::cout << "Time to run triangle count: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

    tic = std::chrono::steady_clock::now();
    std::ofstream myfile;

    std::string NUM_THREADS_STR = std::to_string(NUM_THREADS);
    myfile.open("output/" + name + "/tc_v2_result_" + NUM_THREADS_STR + ".txt");

    myfile << "Number of triangles in graph =  " << *triangle_count << std::endl;

    myfile.close();
    toc = std::chrono::steady_clock::now();
    std::cout << "Time to write data to file: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

    std::chrono::steady_clock::time_point toc_0 = std::chrono::steady_clock::now();
    std::cout << "Total time taken: " << std::chrono::duration_cast<std::chrono::microseconds>(toc_0 - tic_0).count() << "[µs]" << std::endl;

    return 0;
}
