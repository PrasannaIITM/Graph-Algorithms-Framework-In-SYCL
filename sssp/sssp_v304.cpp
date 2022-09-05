#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>
#define DEBUG 0
#define NUM_THREADS

using namespace sycl;

int main()
{

    std::chrono::steady_clock::time_point tic_0 = std::chrono::steady_clock::now();
    std::string name = "simple";
    // The default device selector will select the most performant device.
    default_selector d_selector;
    queue Q(d_selector);
    std::cout << "Selected device: " << Q.get_device().get_info<info::device::name>() << "\n";

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
    int *dev_V = malloc_device<int>(V.size(), Q);
    int *dev_I = malloc_device<int>(I.size(), Q);
    int *dev_E = malloc_device<int>(E.size(), Q);
    int *dev_W = malloc_device<int>(W.size(), Q);
    Q.submit([&](handler &h)
             { h.memcpy(dev_V, &V[0], V.size() * sizeof(int)); });

    Q.submit([&](handler &h)
             { h.memcpy(dev_I, &I[0], I.size() * sizeof(int)); });

    Q.submit([&](handler &h)
             { h.memcpy(dev_E, &E[0], E.size() * sizeof(int)); });

    Q.submit([&](handler &h)
             { h.memcpy(dev_W, &W[0], W.size() * sizeof(int)); });

    Q.wait();

    int N = V.size();
    int stride = NUM_THREADS;
    std::vector<int> dist(N, INT_MAX);

    int *dev_flag = malloc_device<int>(N, Q);
    int *dev_dist = malloc_device<int>(N, Q);
    int *dev_dist_i = malloc_device<int>(N, Q);

    int source = 21297764;
    Q.submit([&](handler &h)
             { h.parallel_for(NUM_THREADS, [=](id<1> i)
                              {
                        int curr_node = i;
                        for(; i < N; i+= stride){
                            dev_flag[i] = 0;
                            dev_dist[i] = INT_MAX;
                            dev_dist_i[i] = INT_MAX;
                        }

                        if(curr_node == source){
                            dev_flag[source] = 1;
                            dev_dist[source] = 0;
                            dev_dist_i[source] = 0;
                        } }); });
    Q.wait();

    {

        tic = std::chrono::steady_clock::now();
        int *early_stop = malloc_shared<int>(1, Q);
        int *seen = malloc_shared<int>(1, Q);
        for (int round = 1; round < N; round++)
        {
            std::cout << "Round num: " << round << std::endl;
            if (*early_stop == 1)
            {
                break;
            }
            *early_stop = 1;
            *seen = 0;

            Q.submit([&](handler &h)
                     { 
                        stream out(1024, 256, h);
                        h.parallel_for(
                            NUM_THREADS, [=](id<1> tid)
                            {
                                // out<<"BEHE"<<*seen<<endl;
                               for (int i = tid; i < N; i += stride)
                               {
                                   if (dev_flag[i])
                                   {
                                       dev_flag[i] = 0;
                                       for (int j = dev_I[i]; j < dev_I[i + 1]; j++)
                                       {
                                           int w = dev_W[j];
                                           int du = dev_dist[i];
                                           int dv = dev_dist[dev_E[j]];
                                           int new_dist = du + w;

                                           if (du == INT_MAX)
                                           {
                                               continue;
                                           }
                                           atomic_ref<int, memory_order::relaxed, memory_scope::device, access::address_space::ext_intel_global_device_space> atomic_data(dev_dist_i[dev_E[j]]);
                                           atomic_data.fetch_min(new_dist);
                                       }
                                   }
                               }
                               atomic_ref<int, memory_order::relaxed, memory_scope::system, access::address_space::global_space> atomic_data(*seen);
                               atomic_data += 1;
                            //    out<<"HEHE"<<*seen<<endl;
                               while (*seen < NUM_THREADS)
                                   ;

                               for (int i = tid; i < N; i += stride)
                               {
                                   if (dev_dist[i] > dev_dist_i[i])
                                   {
                                       dev_dist[i] = dev_dist_i[i];
                                       dev_flag[i] = 1;
                                       *early_stop = 0;
                                   }
                                   dev_dist_i[i] = dev_dist[i];
                                    } }); })
                .wait();
        }
        toc = std::chrono::steady_clock::now();
        std::cout << "Time to run SSSP: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;
    }
    tic = std::chrono::steady_clock::now();

    int num_covered = 0;
    std::ofstream myfile;

    Q.submit([&](handler &h)
             { h.memcpy(&dist[0], dev_dist, N * sizeof(int)); })
        .wait();

    std::string NUM_THREADS_STR = std::to_string(NUM_THREADS);
    myfile.open("output/" + name + "/sssp_v304_result_" + NUM_THREADS_STR + ".txt");

    for (int i = 0; i < N; i++)
    {
        if (dist[i] != INT_MAX)
        {
            num_covered += 1;
        }
        myfile << i << " " << dist[i] << std::endl;
    }
    myfile.close();
    toc = std::chrono::steady_clock::now();
    std::cout << "Time to write data to file: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

    std::chrono::steady_clock::time_point toc_0 = std::chrono::steady_clock::now();
    std::cout << "Total time taken: " << std::chrono::duration_cast<std::chrono::microseconds>(toc_0 - tic_0).count() << "[µs]" << std::endl;

    std::cout << "Percentage coverage = " << 100 * (1.0 * num_covered) / N << std::endl;
    return 0;
}
