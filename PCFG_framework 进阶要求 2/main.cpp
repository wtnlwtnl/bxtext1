#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda.h>
using namespace std;
using namespace chrono;

// 声明关闭工作线程的函数
extern void shutdownWorkerThreads();

// 编译指令如下
// nvcc -O3 -o pcfg_gpu main.cpp train.cpp guessing.cpp md5.cpp cuda_generate.cu -std=c++11 -Xcompiler -fopenmp

int main(int argc, char* argv[])
{
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    
    // 获取命令行参数，用于设置多PT处理数量，默认为1
    int pt_batch_size = 1;
    if (argc > 1) {
        pt_batch_size = atoi(argv[1]);
        if (pt_batch_size <= 0) pt_batch_size = 1;
    }
    cout << "使用PT批处理大小: " << pt_batch_size << endl;
    
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("./input/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "here" << endl;
    int curr_num = 0;
    auto start = system_clock::now();
    // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    int history = 0;
    // std::ofstream a("./output/results.txt");
    
    // 提前预热GPU
    cudaFree(0);
    
    while (!q.priority.empty())
    {
        // 使用批量处理多个PT
        q.PopNextMultiple(pt_batch_size);
        q.total_guesses = q.guesses.size();
        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            // 在此处更改实验生成的猜测上限
            int generate_n=10000000;
            if (history + q.total_guesses > 10000000)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << " seconds"<< endl;
                cout << "Hash time:" << time_hash << " seconds"<<endl;
                cout << "Train time:" << time_train <<" seconds"<<endl;
                break;
            }
        }
        // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
        // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
        if (curr_num > 1000000)
        {
            auto start_hash = system_clock::now();
            bit32 state[4];
            for (string pw : q.guesses)
            {
                // TODO：对于SIMD实验，将这里替换成你的SIMD MD5函数
                MD5Hash(pw, state);
            }

            // 在这里对哈希所需的总时长进行计算
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            // 记录已经生成的口令总数
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }
    
    // 关闭工作线程
    shutdownWorkerThreads();
    
    return 0;
}