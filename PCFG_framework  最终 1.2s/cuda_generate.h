#ifndef CUDA_GENERATE_H
#define CUDA_GENERATE_H

#include <vector>
#include <string>

// 初始化GPU资源
void initGPUResources();

// 释放GPU资源
void freeGPUResources();

// 预缓存segment值到GPU
void cacheSegmentValues(int segmentType, int segmentLength, const std::vector<std::string>& values);

// 使用预缓存的数据生成单段猜测
void generateSingleSegmentGPU(int segmentType, int segmentLength, int maxIndices,
                             std::vector<std::string>& guesses, int& total_guesses);

// 使用预缓存的数据生成多段猜测
void generateMultiSegmentGPU(const std::string& prefix, int segmentType, int segmentLength, 
                            int maxIndices, std::vector<std::string>& guesses, int& total_guesses);

#endif // CUDA_GENERATE_H