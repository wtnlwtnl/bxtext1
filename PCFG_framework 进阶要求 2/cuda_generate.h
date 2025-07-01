#ifndef CUDA_GENERATE_H
#define CUDA_GENERATE_H

#include <vector>
#include <string>
#include <mutex>

// GPU资源初始化和释放函数
void initGPUResources();
void freeGPUResources();

// 单段密码生成函数
void generateSingleSegmentGPU(int segmentType, int segmentLength, int valueCount, 
                             std::vector<std::string>& guesses, int& total_guesses);

// 多段密码生成函数
void generateMultiSegmentGPU(const std::string& prefix, int lastSegmentType, 
                            int lastSegmentLength, int valueCount,
                            std::vector<std::string>& guesses, int& total_guesses);

// 预缓存segment值到GPU
void cacheSegmentValues(int type, int length, const std::vector<std::string>& values);

// 线程安全的函数声明
namespace ThreadSafe {
    extern std::mutex gpu_mutex;
    
    void generateSingleSegmentGPU(int segmentType, int segmentLength, int valueCount, 
                                 std::vector<std::string>& guesses, int& total_guesses);
    
    void generateMultiSegmentGPU(const std::string& prefix, int lastSegmentType, 
                                int lastSegmentLength, int valueCount,
                                std::vector<std::string>& guesses, int& total_guesses);
}

#endif // CUDA_GENERATE_H