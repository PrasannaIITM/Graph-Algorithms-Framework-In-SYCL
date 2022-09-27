#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>
#define DEBUG 0
#define NUM_THREADS 4

using namespace sycl;

int main()
{

    std::chrono::steady_clock::time_point tic_0 = std::chrono::steady_clock::now();
    std::string name = "simplebcud";
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
    int N = V.size();
    int stride = NUM_THREADS;

    std::vector<float> bc(N);

    int *dev_V = malloc_device<int>(V.size(), Q);
    int *dev_I = malloc_device<int>(I.size(), Q);
    int *dev_E = malloc_device<int>(E.size(), Q);
    float *dev_bc = malloc_shared<float>(bc.size(), Q);
    float *dev_delta = malloc_shared<float>(N, Q);
    int *dev_rev_stack = malloc_shared<int>(N, Q);
    int *dev_d = malloc_shared<int>(N, Q);
    int *dev_sigma = malloc_shared<int>(N, Q);
    int *dev_end_point = malloc_shared<int>(N, Q);

    Q.submit([&](handler &h)
             { h.memcpy(dev_V, &V[0], V.size() * sizeof(int)); });

    Q.submit([&](handler &h)
             { h.memcpy(dev_I, &I[0], I.size() * sizeof(int)); });

    Q.submit([&](handler &h)
             { h.memcpy(dev_E, &E[0], E.size() * sizeof(int)); });

    Q.wait();

    tic = std::chrono::steady_clock::now();
    std::cout << "Starting betweenness centrality calculation..." << std::endl;

    int *position = malloc_shared<int>(1, Q);
    int *s = malloc_shared<int>(1, Q);
    *s = 0;
    int *finish_limit_position = malloc_shared<int>(1, Q);
    int *done = malloc_shared<int>(1, Q);
    int *current_depth = malloc_shared<int>(1, Q);

    while (*s < N)
    {
        std::cout << "source = " << *s << std::endl;
        Q.submit([&](handler &h)
                 { h.parallel_for(
                       NUM_THREADS, [=](id<1> i)
                       {
                           for (; i < N; i += stride)
                           {
                               if (i == *s)
                               {
                                   dev_d[i] = 0;
                                   dev_sigma[i] = 1;
                               }
                               else
                               {
                                   dev_d[i] = INT_MAX;
                                   dev_sigma[i] = 0;
                               }
                               dev_delta[i] = 0;
                           } }); })
            .wait();

        *current_depth = 0;
        *done = 0;
        *position = 0;
        *finish_limit_position = 1;
        dev_end_point[0] = 0;

        while (!*done)
        {
            *done = 1;
            std::cout << "current depth" << *current_depth << std::endl;
            Q.submit([&](handler &h)
                     { 
                        stream out(1024, 256, h);
                        h.parallel_for(
                           NUM_THREADS, [=](id<1> i)
                           {
                           for (; i < N; i += stride)
                           {
                               if(dev_d[i] == *current_depth){
                                atomic_ref<int, memory_order::seq_cst, memory_scope::device, access::address_space::global_space> atomic_data(*position);
                                int t = atomic_data++;
                                out<<"position  "<<t<<"  "<<*position<<endl;
                                dev_rev_stack[t] = i;
                                for(int r = dev_I[i]; r < dev_I[i + 1]; r++){
                                    int w = dev_E[r];
                                    if(dev_d[w] == INT_MAX){
                                        dev_d[w] = dev_d[i] + 1;
                                        *done = 0;
                                    }

                                    if(dev_d[w] == (dev_d[i] + 1)){
                                        atomic_ref<int, memory_order::seq_cst, memory_scope::device, access::address_space::global_space> atomic_data(dev_sigma[w]);
                                        atomic_data += dev_sigma[i];
                                    } 
                                }
                               }
                           } }); })
                .wait();
            *current_depth += 1;
            dev_end_point[*finish_limit_position] = *position;
            ++*finish_limit_position;
        }

        for (int itr1 = *finish_limit_position - 1; itr1 >= 0; itr1--)
        {
            Q.submit([&](handler &h)
                     { h.parallel_for(
                           NUM_THREADS, [=](id<1> i)
                           {
                                    for(int itr2 = dev_end_point[itr1] + i; itr2 < dev_end_point[itr1 + 1]; itr2 += stride){
                                        for(int itr3 = dev_I[dev_rev_stack[itr2]]; itr3 < dev_I[dev_rev_stack[itr2] + 1]; itr3++){
                                            int consider = dev_E[itr3];
                                            if(dev_d[consider] == dev_d[dev_rev_stack[itr2]] + 1){
                                                dev_delta[dev_rev_stack[itr2]] += (((float)dev_sigma[dev_rev_stack[itr2]] / dev_sigma[consider]) * ((float)1 + dev_delta[consider]));

                                            }
                                        }

                                        if(dev_rev_stack[itr2] != *s){
                                            dev_bc[dev_rev_stack[itr2]] += dev_delta[dev_rev_stack[itr2]];
                                        }
                                    } }); })
                .wait();
        }
        // for (int itr1 = N - 1; itr1 >= 0; --itr1)
        // {
        //     for (int itr2 = I[dev_rev_stack[itr1]]; itr2 < I[dev_rev_stack[itr1] + 1]; ++itr2)
        //     {
        //         int consider = E[itr2];
        //         if (dev_d[consider] == dev_d[dev_rev_stack[itr1]] - 1)
        //         {
        //             dev_delta[consider] += (((float)dev_sigma[consider] / dev_sigma[dev_rev_stack[itr1]]) * ((float)1 + dev_delta[dev_rev_stack[itr1]]));
        //         }
        //     }
        //     if (dev_rev_stack[itr1] != *s)
        //     {
        //         dev_bc[dev_rev_stack[itr1]] += dev_delta[dev_rev_stack[itr1]];
        //     }
        // }
        *s += 1;
    }

    toc = std::chrono::steady_clock::now();
    std::cout << "Time to run betweenness centrality: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

    Q.submit([&](handler &h)
             { h.memcpy(&bc[0], dev_bc, N * sizeof(float)); })
        .wait();

    std::ofstream myfile;
    std::string NUM_THREADS_STR = std::to_string(NUM_THREADS);
    myfile.open("output/" + name + "/bc_v1_result_" + NUM_THREADS_STR + ".txt");

    for (int i = 0; i < N; i++)
    {
        myfile << i << " " << int(bc[i] / 2.0) << std::endl;
    }
    myfile.close();
    toc = std::chrono::steady_clock::now();
    std::cout << "Time to write data to file: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

    std::chrono::steady_clock::time_point toc_0 = std::chrono::steady_clock::now();
    std::cout << "Total time taken: " << std::chrono::duration_cast<std::chrono::microseconds>(toc_0 - tic_0).count() << "[µs]" << std::endl;
    return 0;
}
