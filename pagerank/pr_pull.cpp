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
    std::string NUM_THREADS_STR = std::to_string(NUM_THREADS);

    logfile.open("pagerank/output/" + name + "_pr_pull_time_" + NUM_THREADS_STR + ".txt");

    logfile << "Processing " << name << std::endl;

    default_selector d_selector;
    queue Q(d_selector);
    logfile << "Selected device: " << Q.get_device().get_info<info::device::name>() << std::endl;
    logfile << "Number of parallel work items: " << NUM_THREADS << std::endl;

    std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
    std::vector<int> V, I, E, RE, RI;
    load_from_file("csr_graphs/" + name + "/V", V);
    load_from_file("csr_graphs/" + name + "/I", I);
    load_from_file("csr_graphs/" + name + "/RE", RE);
    load_from_file("csr_graphs/" + name + "/RI", RI);
    std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
    logfile << "Time to load data from files: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

    int *dev_V = malloc_device<int>(V.size(), Q);
    int *dev_I = malloc_device<int>(I.size(), Q);
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

    tic = std::chrono::steady_clock::now();
    logfile << "Starting Pagerank..." << std::endl;
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
                                   //pull
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
    logfile << "Time to run Pagerank: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

    tic = std::chrono::steady_clock::now();
    std::ofstream resultfile;
    int num_covered = 0;

    Q.submit([&](handler &h)
             { h.memcpy(&pagerank[0], dev_pagerank, N * sizeof(float)); })
        .wait();

    resultfile.open("pagerank/output/" + name + "_pr_pull_result_" + NUM_THREADS_STR + ".txt");

    for (int i = 0; i < N; i++)
    {
        resultfile << i << " " << pagerank[i] << std::endl;
    }
    resultfile.close();
    toc = std::chrono::steady_clock::now();
    logfile << "Time to write data to file: " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;

    std::chrono::steady_clock::time_point toc_0 = std::chrono::steady_clock::now();
    logfile << "Total time taken: " << std::chrono::duration_cast<std::chrono::microseconds>(toc_0 - tic_0).count() << "[µs]" << std::endl;

    return 0;
}
