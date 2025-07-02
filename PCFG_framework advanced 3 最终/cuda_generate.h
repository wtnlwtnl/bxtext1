#ifndef CUDA_GENERATE_H
#define CUDA_GENERATE_H

#include <vector>
#include <string>

// 函数声明：生成单段PT的所有可能值
int generateSingleSegmentGPU(int segmentType, int segmentLength, int valueCount, 
                            std::vector<std::string>& results, int& generatedCount, 
                            size_t offset = 0);

// 函数声明：生成多段PT的所有可能值
int generateMultiSegmentGPU(const std::string& prefix, int lastSegmentType, 
                           int lastSegmentLength, int valueCount, 
                           std::vector<std::string>& results, int& generatedCount,
                           size_t offset = 0);

// 缓存段值，优化访问效率
void cacheSegmentValues(int type, int length, const std::vector<std::string>& values);

#endif // CUDA_GENERATE_H