#ifndef CUDA_GENERATE_H
#define CUDA_GENERATE_H

#include <vector>
#include <string>
#include <mutex>

// 初始化GPU资源
void initGPUResources();

// 释放GPU资源
void freeGPUResources();

// 将segment值缓存到GPU
void cacheSegmentValues(int type, int length, const std::vector<std::string>& values);

// 在GPU上生成单segment的猜测
void generateSingleSegmentGPU(int segmentType, int segmentLength, int valueCount, 
                             std::vector<std::string>& guesses, int& total_guesses);

// 在GPU上生成多segment的猜测
void generateMultiSegmentGPU(const std::string& prefix, int lastSegmentType, 
                            int lastSegmentLength, int valueCount,
                            std::vector<std::string>& guesses, int& total_guesses);

// 线程安全版本的函数，用于多线程环境
namespace ThreadSafe {
    // 获取线程锁，确保GPU资源访问安全
    extern std::mutex gpu_mutex;
    
    // 线程安全版本的生成函数
    void generateSingleSegmentGPU(int segmentType, int segmentLength, int valueCount, 
                                 std::vector<std::string>& guesses, int& total_guesses);
    
    void generateMultiSegmentGPU(const std::string& prefix, int lastSegmentType, 
                                int lastSegmentLength, int valueCount,
                                std::vector<std::string>& guesses, int& total_guesses);
}

#endif // CUDA_GENERATE_H