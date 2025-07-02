#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <cuda_runtime.h>
#include <cuda.h>
#include "cuda_generate.h" // 添加GPU头文件
using namespace std;
using namespace chrono;

// 声明关闭工作线程的函数
extern void shutdownWorkerThreads();

// 编译指令：nvcc -O2 correctness.cpp train.cpp guessing.cpp md5.cpp cuda_generate.cu -o cor -std=c++11 -Xcompiler -fopenmp

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
    
    cout << "初始化GPU资源..." << endl;
    // 初始化GPU资源
    initGPUResources();
    
    // 提前预热GPU，与main.cpp保持一致
    cudaFree(0);
    
    auto start_train = system_clock::now();
    
    cout << "开始训练模型..." << endl;
    // 使用与main.cpp相同的相对路径
    q.m.train("./input/Rockyou-singleLined-full.txt");
    q.m.order();
    
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
    
    cout << "训练完成，耗时: " << time_train << " 秒" << endl;
    
    // 加载测试数据
    cout << "加载测试数据..." << endl;
    unordered_set<std::string> test_set;
    ifstream test_data("./input/Rockyou-singleLined-full.txt");
    
    if (!test_data.is_open()) {
        cerr << "错误：无法打开测试数据文件！请检查文件路径" << endl;
        freeGPUResources();
        return 1;
    }
    
    int test_count = 0;
    string pw;
    while(test_data >> pw)
    {   
        test_count += 1;
        test_set.insert(pw);
        if (test_count >= 1000000)
        {
            break;
        }
    }
    cout << "测试集加载完成，共 " << test_set.size() << " 个密码" << endl;
    
    int total_cracked = 0;   // 总计破解数

    cout << "初始化优先队列..." << endl;
    q.init();
    cout << "优先队列初始化完成，大小: " << q.priority.size() << endl;
    cout << "here" << endl;
    
    int curr_num = 0;
    
    cout << "开始主循环..." << endl;
    auto start = system_clock::now();
    
    // 记录已生成的猜测总数
    int history = 0;
    int loop_count = 0;
    
    // 使用与main.cpp相同的循环结构
    while (!q.priority.empty())
    {
        loop_count++;
        
        // 使用GPU版本的批量处理多个PT
        q.PopNextMultiple(pt_batch_size);
        
        // 重要：与main.cpp保持一致的更新total_guesses方式
        q.total_guesses = q.guesses.size();
        
        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;
        }
        
        // 检查是否应该退出 - 与main.cpp保持一致
        if (history + q.total_guesses > 10000000) {
            cout << "达到最大猜测数限制，退出循环" << endl;
            auto end = system_clock::now();
            auto duration = duration_cast<microseconds>(end - start);
            time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
            
            cout << "Guess time:" << time_guess - time_hash << " seconds" << endl;
            cout << "Hash time:" << time_hash << " seconds" << endl;
            cout << "Train time:" << time_train << " seconds" << endl;
            cout << "Cracked:" << total_cracked << endl;
            break;
        }
        
        // 处理哈希计算 - 与main.cpp保持一致的阈值1000000
        if (curr_num > 1000000)
        {
            cout << "处理当前批次猜测，计算哈希..." << endl;
            auto start_hash = system_clock::now();
            
            bit32 state[4];
            int iteration_cracked = 0;  // 仅用于当前迭代
            
            for (const string& pw : q.guesses)
            {
                if (test_set.find(pw) != test_set.end()) {
                    iteration_cracked += 1;
                }
                MD5Hash(pw, state);
            }
            
            // 累加到总数
            total_cracked += iteration_cracked;
            
            // 计算哈希时间
            auto end_hash = system_clock::now();
            auto hash_duration = duration_cast<microseconds>(end_hash - start_hash);
            double current_hash_time = double(hash_duration.count()) * microseconds::period::num / microseconds::period::den;
            time_hash += current_hash_time;
            
            cout << "本次迭代处理 " << q.guesses.size() << " 个猜测，破解 " << iteration_cracked << " 个密码" << endl;
            cout << "当前总破解数: " << total_cracked << endl;
            
            // 与main.cpp保持一致的历史记录更新
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }
    
    // 处理最后剩余的猜测
    if (q.guesses.size() > 0) {
        cout << "处理最后剩余的 " << q.guesses.size() << " 个猜测..." << endl;
        auto start_hash = system_clock::now();
        
        bit32 state[4];
        int iteration_cracked = 0;
        
        for (const string& pw : q.guesses)
        {
            if (test_set.find(pw) != test_set.end()) {
                iteration_cracked += 1;
            }
            MD5Hash(pw, state);
        }
        
        total_cracked += iteration_cracked;
        
        auto end_hash = system_clock::now();
        auto hash_duration = duration_cast<microseconds>(end_hash - start_hash);
        double current_hash_time = double(hash_duration.count()) * microseconds::period::num / microseconds::period::den;
        time_hash += current_hash_time;
        
        cout << "最后处理破解了 " << iteration_cracked << " 个密码" << endl;
    }
    
    // 最终结果
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    
    cout << "最终结果:" << endl;
    cout << "循环次数: " << loop_count << endl;
    cout << "使用PT批处理大小: " << pt_batch_size << endl;
    cout << "Guess time:" << time_guess - time_hash << " seconds" << endl;
    cout << "Hash time:" << time_hash << " seconds" << endl;
    cout << "Train time:" << time_train << " seconds" << endl;
    cout << "Cracked:" << total_cracked << endl;
    cout << "Total guesses:" << history + q.total_guesses << endl;
    
    cout << "释放GPU资源..." << endl;
    // 关闭工作线程
    shutdownWorkerThreads();
    // 释放GPU资源
    freeGPUResources();
    
    return 0;
}