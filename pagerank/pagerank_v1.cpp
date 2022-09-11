#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>
#define DEBUG 0
#define NUM_THREADS 4

using namespace sycl;

int main()
{

    std::chrono::steady_clock::time_point tic_0 = std::chrono::steady_clock::now();
    std::string name = "simple";
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
    load_from_file("input/" + name + "/RE", RE);
    load_from_file("input/" + name + "/RI", RI);
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
    int *dev_RE = malloc_device<int>(RE.size(), Q);
    int *dev_RI = malloc_device<int>(RI.size(), Q);

    float beta = 0.001;
    int maxIter = 100;
    float delta = 0.85;

    Q.submit([&](handler &h)
             { h.memcpy(dev_V, &V[0], V.size() * sizeof(int)); });

    Q.submit([&](handler &h)
             { h.memcpy(dev_I, &I[0], I.size() * sizeof(int)); });

    Q.submit([&](handler &h)
             { h.memcpy(dev_E, &E[0], E.size() * sizeof(int)); });

    Q.submit([&](handler &h)
             { h.memcpy(dev_RE, &RE[0], RE.size() * sizeof(int)); });

    Q.submit([&](handler &h)
             { h.memcpy(dev_RI, &RI[0], RI.size() * sizeof(int)); });

    Q.wait();

    int N = V.size();
    int stride = NUM_THREADS;
    std::vector<float> pagerank(N);
    float *dev_pagerank = malloc_device<float>(N, Q);
    float *dev_pagerank_i = malloc_device<float>(N, Q);

    Q.submit([&](handler &h)
             { h.parallel_for(NUM_THREADS, [=](id<1> i)
                              {
                                  for (; i < N; i += stride)
                                  {
                                      dev_pagerank[i] = 1/N;
                                      dev_pagerank_i[i] = 1/N;

                                  } }); });
    Q.wait();

    {

        tic = std::chrono::steady_clock::now();
        std::cout << "Starting Pagerank..." << std::endl;
        float *diff = malloc_shared<float>(1, Q);
        int iterCount = 0;
        do
        {
            Q.submit([&](handler &h)
                     { h.parallel_for(
                           NUM_THREADS, [=](id<1> i)
                           {
                               for (; i < N; i += stride)
                               {
                                   float sum = 0;
                                   for (int edge = dev_RI[i]; edge < dev_RI[i + 1]; edge++){
                                       int nbr = dev_RE[edge];
                                       sum = sum + dev_pagerank[nbr] / (dev_I[nbr + 1] - dev_I[nbr]);
                                   }
                                   float val = (1 - delta) / N + delta * sum;
                                   atomic_ref<float, memory_order::seq_cst, memory_scope::device, access::address_space::global_space> atomic_data(*diff);
                                   atomic_data += (float)val - dev_pagerank[i];
                                   dev_pagerank_i[i] = val;
                               } }); })
                .wait();

            Q.submit([&](handler &h)
                     { h.parallel_for(
                           NUM_THREADS, [=](id<1> i)
                           {
                                for (; i < N; i += stride)
                                {
                                    dev_pagerank[i] = dev_pagerank_i[i];
                                } }); })
                .wait();
            iterCount += 1;
        } while ((*diff > beta) && (iterCount < maxIter));

        toc = std::chrono::steady_clock::now();
        std::cout << "Time to run Pagerank: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;
    }
    tic = std::chrono::steady_clock::now();
    std::ofstream myfile;
    int num_covered = 0;

    Q.submit([&](handler &h)
             { h.memcpy(&pagerank[0], dev_pagerank, N * sizeof(float)); })
        .wait();

    std::string NUM_THREADS_STR = std::to_string(NUM_THREADS);
    myfile.open("output/" + name + "/pagerank_v1_result_" + NUM_THREADS_STR + ".txt");

    for (int i = 0; i < N; i++)
    {
        myfile << i << " " << pagerank[i] << std::endl;
    }
    myfile.close();
    toc = std::chrono::steady_clock::now();
    std::cout << "Time to write data to file: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

    std::chrono::steady_clock::time_point toc_0 = std::chrono::steady_clock::now();
    std::cout << "Total time taken: " << std::chrono::duration_cast<std::chrono::microseconds>(toc_0 - tic_0).count() << "[µs]" << std::endl;

    return 0;
}
