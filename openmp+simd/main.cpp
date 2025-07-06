
// #include "PCFG.h"
// #include <chrono>
// #include <fstream>
// #include "md5.h"
// #include <iomanip>
// #include <iostream>
// using namespace std;
// using namespace chrono;

// // 编译指令如下
// // g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// // g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// // g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2
// // 使用NEON优化时：
// // g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2 -mfpu=neon

// /**
//  * 串行处理MD5哈希计算
//  * @param guesses 需要计算哈希的字符串列表
//  * @param time_hash 累计哈希计算时间
//  */
// void process_serial_md5(const vector<string>& guesses, double& time_hash) {
//     auto start_hash = system_clock::now();

//     bit32 state[4];
//     for (const string& pw : guesses) {
//         MD5Hash(pw, state);
//         // 这里可以添加处理哈希结果的代码，如输出等
//     }

//     auto end_hash = system_clock::now();
//     auto duration = duration_cast<microseconds>(end_hash - start_hash);
//     time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
// }

// /**
//  * 并行处理MD5哈希计算，使用NEON SIMD指令
//  * @param guesses 需要计算哈希的字符串列表
//  * @param time_hash 累计哈希计算时间
//  */
// void process_parallel_md5(const vector<string>& guesses, double& time_hash) {
//     auto start_hash = system_clock::now();

//     const size_t totalGuesses = guesses.size();
//     size_t processed = 0;

//     while (processed < totalGuesses) {
//         // 每次处理4个字符串（NEON寄存器的宽度）
//         size_t batchSize = min(size_t(4), totalGuesses - processed);
//         vector<string> batch;
//         vector<bit32*> results;

//         // 准备当前批次的输入
//         for (size_t i = 0; i < batchSize; i++) {
//             batch.push_back(guesses[processed + i]);
//         }

//         // 使用NEON并行处理
//         MD5Hash_NEON(batch, results);

//         // 这里可以添加处理哈希结果的代码，如输出等

//         // 释放内存并更新处理计数
//         for (auto result : results) {
//             delete[] result;
//         }
//         processed += batchSize;
//     }

//     auto end_hash = system_clock::now();
//     auto duration = duration_cast<microseconds>(end_hash - start_hash);
//     time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
// }

// int main(int argc, char* argv[])
// {
//     double time_hash_serial = 0;   // 用于串行MD5哈希的时间
//     double time_hash_parallel = 0; // 用于并行MD5哈希的时间
//     double time_guess = 0;         // 猜测的总时长
//     double time_train = 0;         // 模型训练的总时长
    
//     cout << "同时运行串行和并行MD5实现..." << endl;

//     PriorityQueue q;
//     auto start_train = system_clock::now();
//     q.m.train("/guessdata/Rockyou-singleLined-full.txt");
//     q.m.order();
//     auto end_train = system_clock::now();
//     auto duration_train = duration_cast<microseconds>(end_train - start_train);
//     time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

//     q.init();
//     cout << "here" << endl;
//     int curr_num = 0;
//     auto start = system_clock::now();
//     // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
//     int history = 0;
//     // 为了同时运行两种实现，我们需要保存所有生成的猜测
//     vector<vector<string>> all_guesses;
    
//     while (!q.priority.empty())
//     {
//         q.PopNext();
//         q.total_guesses = q.guesses.size();
//         if (q.total_guesses - curr_num >= 100000)
//         {
//             cout << "Guesses generated: " << history + q.total_guesses << endl;
//             curr_num = q.total_guesses;

//             // 在此处更改实验生成的猜测上限
//             int generate_n = 10000000;
//             if (history + q.total_guesses > generate_n)
//             {
//                 // 最后一次处理剩余的猜测
//                 all_guesses.push_back(q.guesses);
//                 break;
//             }
//         }
        
//         // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
//         // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
//         if (curr_num > 1000000)
//         {
//             // 保存当前批次的猜测，供稍后同时处理
//             all_guesses.push_back(q.guesses);
            
//             // 记录已经生成的口令总数
//             history += curr_num;
//             curr_num = 0;
//             q.guesses.clear();
//         }
//     }

//     auto end_guess = system_clock::now();
//     auto duration_guess = duration_cast<microseconds>(end_guess - start);
//     time_guess = double(duration_guess.count()) * microseconds::period::num / microseconds::period::den;

//     // 现在分别运行串行和并行处理
//     cout << "处理所有猜测..." << endl;

//     // 串行处理所有猜测
//     for (const auto& guesses : all_guesses) {
//         process_serial_md5(guesses, time_hash_serial);
//     }
    
//     // 并行处理所有猜测
//     for (const auto& guesses : all_guesses) {
//         process_parallel_md5(guesses, time_hash_parallel);
//     }

//     // 输出性能结果 - 修正这一行
//     cout << "Guess time: " << time_guess << " seconds" << endl;
//     cout << "Serial hash time: " << time_hash_serial << " seconds" << endl;
//     cout << "Parallel hash time: " << time_hash_parallel << " seconds" << endl;
//     cout << "Total time (generation + both hash methods): " 
//          << time_guess + time_hash_serial + time_hash_parallel << " seconds" << endl;
//     cout << "Train time: " << time_train << " seconds" << endl;
    
//     // 计算加速比
//     double speedup = time_hash_serial / time_hash_parallel;
//     cout << "Speedup (Serial/Parallel): " << speedup << "x" << endl;

//     return 0;
// }



#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <iostream>
#include <unordered_set>
using namespace std;
using namespace chrono;

/**
 * 串行处理MD5哈希计算
 * @param guesses 需要计算哈希的字符串列表
 * @param time_hash 累计哈希计算时间
 */
void process_serial_md5(const vector<string>& guesses, double& time_hash) {
    auto start_hash = system_clock::now();

    bit32 state[4];
    for (const string& pw : guesses) {
        MD5Hash(pw, state);
    }

    auto end_hash = system_clock::now();
    auto duration = duration_cast<microseconds>(end_hash - start_hash);
    time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
}

/**
 * 并行处理MD5哈希计算，使用NEON SIMD指令
 * @param guesses 需要计算哈希的字符串列表
 * @param time_hash 累计哈希计算时间
 */
void process_parallel_md5(const vector<string>& guesses, double& time_hash) {
    auto start_hash = system_clock::now();

    const size_t totalGuesses = guesses.size();
    size_t processed = 0;

    while (processed < totalGuesses) {
        // 每次处理4个字符串（NEON寄存器的宽度）
        size_t batchSize = min(size_t(4), totalGuesses - processed);
        vector<string> batch;
        vector<bit32*> results;

        // 准备当前批次的输入
        for (size_t i = 0; i < batchSize; i++) {
            batch.push_back(guesses[processed + i]);
        }

        // 使用NEON并行处理
        MD5Hash_NEON(batch, results);

        // 释放内存并更新处理计数
        for (auto result : results) {
            delete[] result;
        }
        processed += batchSize;
    }

    auto end_hash = system_clock::now();
    auto duration = duration_cast<microseconds>(end_hash - start_hash);
    time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
}

int main(int argc, char* argv[])
{
    double time_hash_serial = 0;
    double time_hash_parallel = 0;
    double time_guess = 0;
    double time_vector_ops = 0; // 新增：记录向量操作时间
    double time_train = 0;
    
    cout << "同时运行串行和并行MD5实现..." << endl;

    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "here" << endl;
    int curr_num = 0;
    auto start = system_clock::now();
    int history = 0;
    vector<vector<string>> all_guesses;
    
    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();
        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            int generate_n = 10000000;
            if (history + q.total_guesses > generate_n)
            {
                // 计时向量操作
                auto start_vec = system_clock::now();
                all_guesses.push_back(q.guesses);
                auto end_vec = system_clock::now();
                auto duration_vec = duration_cast<microseconds>(end_vec - start_vec);
                time_vector_ops += double(duration_vec.count()) * microseconds::period::num / microseconds::period::den;
                break;
            }
        }
        
        if (curr_num > 1000000)
        {
            // 计时向量操作
            auto start_vec = system_clock::now();
            all_guesses.push_back(q.guesses);
            auto end_vec = system_clock::now();
            auto duration_vec = duration_cast<microseconds>(end_vec - start_vec);
            time_vector_ops += double(duration_vec.count()) * microseconds::period::num / microseconds::period::den;
            
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }

    auto end_guess = system_clock::now();
    auto duration_guess = duration_cast<microseconds>(end_guess - start);
    time_guess = double(duration_guess.count()) * microseconds::period::num / microseconds::period::den;

    // 处理猜测...
    cout << "处理所有猜测..." << endl;
    for (const auto& guesses : all_guesses) {
        process_serial_md5(guesses, time_hash_serial);
    }
    for (const auto& guesses : all_guesses) {
        process_parallel_md5(guesses, time_hash_parallel);
    }

    // 修改输出，以与correctness_guess.cpp保持一致
    cout << "Pure guess time: " << time_guess - time_vector_ops << " seconds" << endl;
    //cout << "Vector operations time: " << time_vector_ops << " seconds" << endl;
    //cout << "Total guess time (with vector ops): " << time_guess << " seconds" << endl;
    cout << "Serial hash time: " << time_hash_serial << " seconds" << endl;
    cout << "Parallel hash time: " << time_hash_parallel << " seconds" << endl;
    cout << "Total time: " << time_guess + time_hash_serial + time_hash_parallel << " seconds" << endl;
    cout << "Train time: " << time_train << " seconds" << endl;
    
    double speedup = time_hash_serial / time_hash_parallel;
    cout << "Speedup (Serial/Parallel): " << speedup << "x" << endl;

    return 0;
}