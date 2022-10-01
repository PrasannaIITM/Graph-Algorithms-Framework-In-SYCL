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
    int *dev_d = malloc_shared<int>(N, Q);
    int *dev_sigma = malloc_shared<int>(N, Q);
    float *dev_bc = malloc_shared<float>(bc.size(), Q);
    float *dev_delta = malloc_shared<float>(N, Q);

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
    std::cout << "Starting betweenness centrality calculation..." << std::endl;

    int *s = malloc_shared<int>(1, Q);
    *s = 0;

    int *q_curr = malloc_shared<int>(N, Q);
    int *q_curr_len = malloc_shared<int>(1, Q);

    int *q_next = malloc_shared<int>(N, Q);
    int *q_next_len = malloc_shared<int>(1, Q);

    int *S = malloc_shared<int>(N, Q);
    int *S_len = malloc_shared<int>(1, Q);

    int *ends = malloc_shared<int>(N, Q);
    int *ends_len = malloc_shared<int>(1, Q);
    int *depth = malloc_shared<int>(1, Q);

    while (*s < N)
    {
        Q.submit([&](handler &h)
                 { h.parallel_for(
                       NUM_THREADS, [=](id<1> v)
                       {
                           for (; v < N; v += stride)
                           {
                               if (v == *s)
                               {
                                   dev_d[v] = 0;
                                   dev_sigma[v] = 1;
                               }
                               else
                               {
                                   dev_d[v] = INT_MAX;
                                   dev_sigma[v] = 0;
                               }
                               dev_delta[v] = 0;
                           } }); })
            .wait();

        q_curr[0] = *s;
        *q_curr_len = 1;
        *q_next_len = 0;
        S[0] = *s;
        *S_len = 1;
        ends[0] = 0;
        ends[1] = 1;
        *ends_len = 2;
        *depth = 0;

        while (1)
        {
            Q.submit([&](handler &h)
                     { h.parallel_for(
                           *q_curr_len, [=](id<1> i)
                           {
                               int v = q_curr[i];
                               for (int r = dev_I[v]; r < dev_I[v + 1]; r++){
                                   int w = dev_E[r];

                                    atomic_ref<int, memory_order::seq_cst, memory_scope::device, access::address_space::global_space> atomic_data(dev_d[w]);
                                    int old = INT_MAX;
                                    old = atomic_data.compare_exchange_strong(old, dev_d[v] + 1);
                                    if(old){
                                        atomic_ref<int, memory_order::seq_cst, memory_scope::device, access::address_space::global_space> atomic_data(*q_next_len);
                                        int t = atomic_data++;
                                        q_next[t] = w;
                                    }
                                   

                                   if(dev_d[w] == dev_d[v] + 1){
                                       atomic_ref<int, memory_order::seq_cst, memory_scope::device, access::address_space::global_space> atomic_data(dev_sigma[w]);
                                       atomic_data += dev_sigma[v];
                                   }

                               } }); })
                .wait();

            if (*q_next_len == 0)
            {
                break;
            }
            else
            {
                Q.submit([&](handler &h)
                         { h.parallel_for(
                               *q_next_len, [=](id<1> i)
                               {
                                   q_curr[i] = q_next[i];
                                   S[i + *S_len] = q_next[i]; }); })
                    .wait();

                ends[*ends_len] = ends[*ends_len - 1] + *q_next_len;
                *ends_len += 1;
                int new_curr_len = *q_next_len;
                *q_curr_len = new_curr_len;
                *S_len += new_curr_len;
                *q_next_len = 0;
                *depth += 1;
            }
        }
        while (*depth >= 0)
        {
            Q.submit([&](handler &h)
                     { h.parallel_for(
                           ends[*depth + 1] - ends[*depth], [=](id<1> i)
                           {
                               int tid = i + ends[*depth];
                               int w = S[tid];
                               int dsw = 0;
                               for (int r = dev_I[w]; r < dev_I[w + 1]; r++)
                               {
                                   int v = dev_E[r];
                                   if (dev_d[v] == dev_d[w] + 1)
                                   {
                                       atomic_ref<float, memory_order::seq_cst, memory_scope::device, access::address_space::global_space> atomic_data(dev_delta[w]);
                                       atomic_data += (((float)dev_sigma[w] / dev_sigma[v]) * ((float)1 + dev_delta[v]));
                                   }
                               }
                               if (w != *s)
                               {
                                   dev_bc[w] += dev_delta[w];
                               } }); })
                .wait();

            *depth -= 1;
        }
        *s += 1;
    }

    std::cout << std::endl;
    toc = std::chrono::steady_clock::now();
    std::cout << "Time to run betweenness centrality: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

    Q.submit([&](handler &h)
             { h.memcpy(&bc[0], dev_bc, N * sizeof(float)); })
        .wait();

    std::ofstream myfile;
    std::string NUM_THREADS_STR = std::to_string(NUM_THREADS);
    myfile.open("output/" + name + "/bc_v2_result_" + NUM_THREADS_STR + ".txt");

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
