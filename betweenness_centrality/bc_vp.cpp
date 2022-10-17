#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>
#define DEBUG 0

using namespace sycl;

int main(int argc, char **argv)
{
    std::chrono::steady_clock::time_point tic_0 = std::chrono::steady_clock::now();
    std::ofstream logfile;

    std::string name = argv[1];
    int NUM_THREADS = atoi(argv[2]);
    int sSize = atoi(argv[3]);
    std::string str_sSize = std::to_string(sSize);
    std::vector<int> sourceSet;

    std::ifstream input("betweenness_centrality/sources/" + name + "/" + str_sSize);
    int num;
    while ((input >> num))
    {
        sourceSet.push_back(num);
    }
    input.close();

    std::string NUM_THREADS_STR = std::to_string(NUM_THREADS);

    logfile.open("betweenness_centrality/output/" + name + "_" + str_sSize + "_bc_vp_time_" + NUM_THREADS_STR + ".txt");

    logfile << "Processing " << name << std::endl;
    default_selector d_selector;
    queue Q(d_selector);
    logfile << "Selected device: " << Q.get_device().get_info<info::device::name>() << std::endl;
    logfile << "Number of parallel work items: " << NUM_THREADS << std::endl;

    std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
    std::vector<int> V, I, E, RE, RI;
    load_from_file("csr_graphs/" + name + "/V", V);
    load_from_file("csr_graphs/" + name + "/I", I);
    load_from_file("csr_graphs/" + name + "/E", E);

    std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
    logfile << "Time to load data from files: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

    int N = V.size();
    int stride = NUM_THREADS;

    std::vector<float> bc(N);

    int *dev_V = malloc_device<int>(V.size(), Q);
    int *dev_I = malloc_device<int>(I.size(), Q);
    int *dev_E = malloc_device<int>(E.size(), Q);
    float *dev_bc = malloc_shared<float>(bc.size(), Q);
    float *dev_delta = malloc_shared<float>(N, Q);
    int *S = malloc_shared<int>(N, Q);
    int *dev_d = malloc_shared<int>(N, Q);
    int *dev_sigma = malloc_shared<int>(N, Q);
    int *ends = malloc_shared<int>(N, Q);

    Q.submit([&](handler &h)
             { h.memcpy(dev_V, &V[0], V.size() * sizeof(int)); });

    Q.submit([&](handler &h)
             { h.memcpy(dev_I, &I[0], I.size() * sizeof(int)); });

    Q.submit([&](handler &h)
             { h.memcpy(dev_E, &E[0], E.size() * sizeof(int)); });

    Q.wait();

    Q.submit([&](handler &h)
             { h.parallel_for(
                   NUM_THREADS, [=](id<1> v)
                   {
                           for (; v < N; v += stride)
                           {
                               dev_bc[v] = 0;
                           } }); })
        .wait();

    tic = std::chrono::steady_clock::now();
    logfile << "Starting betweenness centrality calculation..." << std::endl;

    int *position = malloc_shared<int>(1, Q);
    int *s = malloc_shared<int>(1, Q);
    *s = 0;
    int *finish_limit_position = malloc_shared<int>(1, Q);
    int *done = malloc_shared<int>(1, Q);
    int *current_depth = malloc_shared<int>(1, Q);

    for (auto x : sourceSet)
    {
        *s = x;
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
        ends[0] = 0;

        while (!*done)
        {
            *done = 1;
            Q.submit([&](handler &h)
                     { h.parallel_for(
                           NUM_THREADS, [=](id<1> i)
                           {
                           for (; i < N; i += stride)
                           {
                               if(dev_d[i] == *current_depth){
                                atomic_ref<int, memory_order::relaxed, memory_scope::system, access::address_space::global_space> atomic_data(*position);
                                int t = atomic_data++;
                                S[t] = i;
                                for(int r = dev_I[i]; r < dev_I[i + 1]; r++){
                                    int w = dev_E[r];
                                    if(dev_d[w] == INT_MAX){
                                        dev_d[w] = dev_d[i] + 1;
                                        *done = 0;
                                    }

                                    if(dev_d[w] == (dev_d[i] + 1)){
                                        atomic_ref<int, memory_order::relaxed, memory_scope::system, access::address_space::global_space> atomic_data(dev_sigma[w]);
                                        atomic_data += dev_sigma[i];
                                    } 
                                }
                               }
                           } }); })
                .wait();
            *current_depth += 1;
            ends[*finish_limit_position] = *position;
            ++*finish_limit_position;
        }
        for (int itr1 = *finish_limit_position - 1; itr1 >= 0; itr1--)
        {
            Q.submit([&](handler &h)
                     { h.parallel_for(
                           NUM_THREADS, [=](id<1> i)
                           {
                                    for(int itr2 = ends[itr1] + i; itr2 < ends[itr1 + 1]; itr2 += stride){
                                        for(int itr3 = dev_I[S[itr2]]; itr3 < dev_I[S[itr2] + 1]; itr3++){
                                            int consider = dev_E[itr3];
                                            if(dev_d[consider] == dev_d[S[itr2]] + 1){
                                                dev_delta[S[itr2]] += (((float)dev_sigma[S[itr2]] / dev_sigma[consider]) * ((float)1 + dev_delta[consider]));

                                            }
                                        }

                                        if(S[itr2] != *s){
                                            dev_bc[S[itr2]] += dev_delta[S[itr2]];
                                        }
                                    } }); })
                .wait();
        }
        // serial implementation
        // for (int itr1 = N - 1; itr1 >= 0; --itr1)
        // {
        //     for (int itr2 = I[S[itr1]]; itr2 < I[S[itr1] + 1]; ++itr2)
        //     {
        //         int consider = E[itr2];
        //         if (dev_d[consider] == dev_d[S[itr1]] - 1)
        //         {
        //             dev_delta[consider] += (((float)dev_sigma[consider] / dev_sigma[S[itr1]]) * ((float)1 + dev_delta[S[itr1]]));
        //         }
        //     }
        //     if (S[itr1] != *s)
        //     {
        //         dev_bc[S[itr1]] += dev_delta[S[itr1]];
        //     }
        // }
    }

    toc = std::chrono::steady_clock::now();
    logfile << "Time to run betweenness centrality: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

    Q.submit([&](handler &h)
             { h.memcpy(&bc[0], dev_bc, N * sizeof(float)); })
        .wait();

    tic = std::chrono::steady_clock::now();
    std::ofstream resultfile;

    resultfile.open("betweenness_centrality/output/" + name + "_" + str_sSize + "_bc_vp_result_" + NUM_THREADS_STR + ".txt");

    for (int i = 0; i < N; i++)
    {
        resultfile << i << " " << bc[i] / 2.0 << std::endl;
    }
    resultfile.close();
    toc = std::chrono::steady_clock::now();
    logfile << "Time to write data to file: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

    std::chrono::steady_clock::time_point toc_0 = std::chrono::steady_clock::now();
    logfile << "Total time taken: " << std::chrono::duration_cast<std::chrono::microseconds>(toc_0 - tic_0).count() << "[µs]" << std::endl;
    return 0;
}
