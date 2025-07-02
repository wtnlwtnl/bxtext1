#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sstream>
#include <algorithm>
using namespace std;
using namespace chrono;

// 声明关闭工作线程的函数
extern void shutdownWorkerThreads();

// 实现MD5函数，用于计算密码哈希
string md5(string s) {
    bit32 state[4]; // 创建存储结果的数组
    MD5Hash(s, state); // 调用真正的MD5实现
    
    // 将结果转换为十六进制字符串
    stringstream ss;
    for (int i = 0; i < 4; i++) {
        ss << hex << setfill('0') << setw(8) << state[i];
    }
    return ss.str();
}

int main(int argc, char* argv[])
{
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    
    // 获取命令行参数，用于设置多PT处理数量，默认为8
    int pt_batch_size = 8;
    if (argc > 1) {
        pt_batch_size = atoi(argv[1]);
        if (pt_batch_size <= 0) pt_batch_size = 8;
    }
    cout << "使用PT批处理大小: " << pt_batch_size << endl;
    
    // 预热GPU
    cudaFree(0);
    
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("./input/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "初始化完成，开始密码生成" << endl;
    int curr_num = 0;
    auto start = system_clock::now();
    // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    int history = 0;
    
    while (!q.priority.empty())
    {
        // 批量处理多个PT
        q.PopNextMultiple(pt_batch_size);
        
        if (q.guesses.size() - curr_num >= 100000)
        {
            cout << "已生成猜测: " << history + q.guesses.size() << ", 队列中剩余PT: " << q.priority.size() << endl;
            curr_num = q.guesses.size();

            // 在此处更改实验生成的猜测上限
            if (history + q.guesses.size() > 10000000)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "生成猜测时间:" << time_guess - time_hash << " seconds"<< endl;
                cout << "哈希计算时间:" << time_hash << " seconds"<<endl;
                cout << "训练模型时间:" << time_train <<" seconds"<<endl;
                cout << "每秒生成猜测: " << (history + q.guesses.size()) / (time_guess - time_hash) << endl;
                break;
            }
        }
        
        // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
        // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
        if (q.guesses.size() > 1000000)
        {
            auto start_hash = system_clock::now();
            bit32 state[4];
            for (string pw : q.guesses)
            {
                // MD5哈希计算
                MD5Hash(pw, state);
            }

            // 在这里对哈希所需的总时长进行计算
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            // 记录已经生成的口令总数
            history += q.guesses.size();
            curr_num = 0;
            q.guesses.clear();
        }
    }
    
    // 关闭工作线程
    shutdownWorkerThreads();
    
    return 0;
}