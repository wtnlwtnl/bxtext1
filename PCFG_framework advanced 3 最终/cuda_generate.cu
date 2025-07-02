#include "cuda_generate.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

// 用于段值缓存的结构
struct SegmentCache {
    char* d_values;    // 设备上的值数组
    int* d_offsets;    // 每个值的偏移
    int* d_lengths;    // 每个值的长度
    int count;         // 值的数量
    int maxLength;     // 最长值的长度
    size_t memSize;    // 占用的GPU内存大小
};

// 全局缓存与管理
std::unordered_map<int, SegmentCache> segmentCaches;
std::mutex cacheMutex;
bool cacheInitialized = false;
size_t totalCacheMemory = 0;
const size_t MAX_CACHE_MEMORY = 512 * 1024 * 1024; // 512MB 缓存上限

// CUDA流与缓冲区
cudaStream_t cudaStreams[2]; // 多个CUDA流支持并行执行
char* d_resultBuffer[2]; // 双缓冲结果缓冲区
size_t resultBufferSize[2] = {0, 0};
int currentBuffer = 0; // 当前使用的缓冲区索引

// 生成缓存键
int getCacheKey(int type, int length) {
    return (type << 16) | length;
}

// 初始化CUDA资源
void initCudaResources() {
    if (!cacheInitialized) {
        for (int i = 0; i < 2; i++) {
            cudaStreamCreate(&cudaStreams[i]);
            d_resultBuffer[i] = nullptr;
        }
        cacheInitialized = true;
    }
}

// 检查CUDA错误
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// 确保结果缓冲区足够大
void ensureResultBufferSize(size_t requiredSize, int bufferIndex) {
    if (resultBufferSize[bufferIndex] < requiredSize) {
        if (d_resultBuffer[bufferIndex] != nullptr) {
            cudaFree(d_resultBuffer[bufferIndex]);
        }
        
        // 分配至少两倍所需大小，减少频繁重新分配
        size_t newSize = max(requiredSize * 2, size_t(1024 * 1024)); // 至少1MB
        cudaMalloc(&d_resultBuffer[bufferIndex], newSize);
        resultBufferSize[bufferIndex] = newSize;
    }
}

// 优化的单段密码生成内核 - 使用共享内存提高性能
__global__ void optimizedSingleSegmentKernel(char* results, const char* values, 
                                           const int* offsets, const int* lengths, 
                                           int numValues, int maxResultLength) {
    extern __shared__ char sharedMem[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numValues) return;
    
    // 计算结果偏移
    char* result = results + idx * maxResultLength;
    
    // 获取当前值的偏移和长度
    int offset = offsets[idx];
    int length = lengths[idx];
    
    // 先将数据加载到共享内存 - 使用线程协作
    int sharedOffset = 0;
    for (int i = threadIdx.x; i < length; i += blockDim.x) {
        if (offset + i < offset + length) {
            sharedMem[sharedOffset + i] = values[offset + i];
        }
    }
    __syncthreads();
    
    // 从共享内存复制到结果
    for (int i = 0; i < length; i++) {
        result[i] = sharedMem[sharedOffset + i];
    }
    
    // 添加字符串结束符
    result[length] = '\0';
}

// 优化的多段密码生成内核 - 批量处理提高效率
__global__ void multiSegmentKernel(char* results, const char* values, const int* offsets, 
                                  const int* lengths, int numValues, const char* prefix, 
                                  int prefixLength, int maxResultLength) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numValues) return;
    
    // 计算结果偏移
    char* result = results + idx * maxResultLength;
    
    // 复制前缀到结果 - 线程协作
    for (int i = threadIdx.x % prefixLength; i < prefixLength; i += blockDim.x) {
        if (i < prefixLength) {
            result[i] = prefix[i];
        }
    }
    
    // 获取当前值的偏移和长度
    int offset = offsets[idx];
    int length = lengths[idx];
    
    // 复制值到结果
    for (int i = 0; i < length; i++) {
        result[prefixLength + i] = values[offset + i];
    }
    
    // 添加字符串结束符
    result[prefixLength + length] = '\0';
}

// 缓存段值 - 增加智能缓存管理
void cacheSegmentValues(int type, int length, const std::vector<std::string>& values) {
    if (values.empty()) return;
    
    // 初始化CUDA资源
    if (!cacheInitialized) {
        initCudaResources();
    }
    
    std::lock_guard<std::mutex> lock(cacheMutex);
    
    int cacheKey = getCacheKey(type, length);
    
    // 检查是否已缓存
    if (segmentCaches.find(cacheKey) != segmentCaches.end()) {
        return;  // 已缓存
    }
    
    // 创建新缓存
    SegmentCache cache;
    cache.count = values.size();
    
    // 计算总大小和最大长度
    size_t totalSize = 0;
    int maxLength = 0;
    for (const auto& value : values) {
        totalSize += value.length();
        maxLength = std::max(maxLength, static_cast<int>(value.length()));
    }
    
    cache.maxLength = maxLength;
    
    // 估计缓存所需内存
    size_t estimatedMemSize = totalSize + values.size() * sizeof(int) * 2;
    
    // 如果缓存太大，需要移除一些旧缓存
    if (totalCacheMemory + estimatedMemSize > MAX_CACHE_MEMORY) {
        // 简单策略：移除最大的缓存直到有足够空间
        while (!segmentCaches.empty() && totalCacheMemory + estimatedMemSize > MAX_CACHE_MEMORY) {
            // 找到最大的缓存
            auto maxIt = segmentCaches.begin();
            for (auto it = segmentCaches.begin(); it != segmentCaches.end(); ++it) {
                if (it->second.memSize > maxIt->second.memSize) {
                    maxIt = it;
                }
            }
            
            // 释放内存并移除缓存
            cudaFree(maxIt->second.d_values);
            cudaFree(maxIt->second.d_offsets);
            cudaFree(maxIt->second.d_lengths);
            totalCacheMemory -= maxIt->second.memSize;
            segmentCaches.erase(maxIt);
        }
    }
    
    // 分配主机内存
    std::vector<char> h_values(totalSize);
    std::vector<int> h_offsets(values.size());
    std::vector<int> h_lengths(values.size());
    
    // 填充主机内存
    size_t offset = 0;
    for (size_t i = 0; i < values.size(); i++) {
        h_offsets[i] = offset;
        h_lengths[i] = values[i].length();
        
        for (size_t j = 0; j < values[i].length(); j++) {
            h_values[offset + j] = values[i][j];
        }
        
        offset += values[i].length();
    }
    
    // 分配设备内存
    checkCudaError(cudaMalloc(&cache.d_values, totalSize), "Allocating device values");
    checkCudaError(cudaMalloc(&cache.d_offsets, values.size() * sizeof(int)), "Allocating device offsets");
    checkCudaError(cudaMalloc(&cache.d_lengths, values.size() * sizeof(int)), "Allocating device lengths");
    
    // 复制数据到设备
    checkCudaError(cudaMemcpy(cache.d_values, h_values.data(), totalSize, cudaMemcpyHostToDevice), 
                  "Copying values to device");
    checkCudaError(cudaMemcpy(cache.d_offsets, h_offsets.data(), values.size() * sizeof(int), cudaMemcpyHostToDevice), 
                  "Copying offsets to device");
    checkCudaError(cudaMemcpy(cache.d_lengths, h_lengths.data(), values.size() * sizeof(int), cudaMemcpyHostToDevice), 
                  "Copying lengths to device");
    
    // 更新缓存信息
    cache.memSize = totalSize + values.size() * sizeof(int) * 2;
    totalCacheMemory += cache.memSize;
    
    // 保存缓存
    segmentCaches[cacheKey] = cache;
}

// 使用CUDA流和双缓冲的单段PT生成函数
int generateSingleSegmentGPU(int segmentType, int segmentLength, int valueCount, 
                           std::vector<std::string>& results, int& generatedCount, 
                           size_t offset) {
    // 初始化CUDA资源
    if (!cacheInitialized) {
        initCudaResources();
    }
    
    // 获取缓存键
    int cacheKey = getCacheKey(segmentType, segmentLength);
    
    // 切换当前缓冲区
    currentBuffer = (currentBuffer + 1) % 2;
    int bufferIndex = currentBuffer;
    
    // 检查是否已缓存
    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        if (segmentCaches.find(cacheKey) == segmentCaches.end()) {
            // 未缓存，返回错误
            generatedCount = 0;
            return -1;
        }
    }
    
    // 获取缓存
    const SegmentCache& cache = segmentCaches[cacheKey];
    
    // 限制valueCount不超过可用值数量
    valueCount = std::min(valueCount, cache.count);
    
    // 计算结果的最大长度（包括null终止符）
    int maxResultLength = cache.maxLength + 1;
    
    // 确保结果缓冲区足够大
    size_t requiredSize = valueCount * maxResultLength;
    ensureResultBufferSize(requiredSize, bufferIndex);
    
    // 配置并启动内核
    int threadsPerBlock = 256;
    int blocksPerGrid = (valueCount + threadsPerBlock - 1) / threadsPerBlock;
    
    // 计算共享内存大小 - 每个块最大处理量
    size_t sharedMemSize = min(size_t(48 * 1024), size_t(cache.maxLength * threadsPerBlock));
    
    // 使用流异步启动内核
    optimizedSingleSegmentKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize, cudaStreams[bufferIndex]>>>(
        d_resultBuffer[bufferIndex], cache.d_values, cache.d_offsets, cache.d_lengths, 
        valueCount, maxResultLength
    );
    
    // 检查内核启动错误
    checkCudaError(cudaGetLastError(), "Launching single segment kernel");
    
    // 分配主机内存用于结果
    std::vector<char> h_results(valueCount * maxResultLength);
    
    // 异步复制结果到主机
    checkCudaError(cudaMemcpyAsync(h_results.data(), d_resultBuffer[bufferIndex], valueCount * maxResultLength, 
                             cudaMemcpyDeviceToHost, cudaStreams[bufferIndex]), 
                  "Copying results from device");
    
    // 同步流
    cudaStreamSynchronize(cudaStreams[bufferIndex]);
    
    // 将结果转换为字符串
    for (int i = 0; i < valueCount; i++) {
        const char* str = h_results.data() + i * maxResultLength;
        results[offset + i] = str;  // 使用偏移直接写入results
    }
    
    // 设置生成数量
    generatedCount = valueCount;
    
    return 0;
}

// 优化后的多段PT生成函数
int generateMultiSegmentGPU(const std::string& prefix, int lastSegmentType, 
                          int lastSegmentLength, int valueCount, 
                          std::vector<std::string>& results, int& generatedCount,
                          size_t offset) {
    // 初始化CUDA资源
    if (!cacheInitialized) {
        initCudaResources();
    }
    
    // 获取缓存键
    int cacheKey = getCacheKey(lastSegmentType, lastSegmentLength);
    
    // 切换当前缓冲区
    currentBuffer = (currentBuffer + 1) % 2;
    int bufferIndex = currentBuffer;
    
    // 检查是否已缓存
    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        if (segmentCaches.find(cacheKey) == segmentCaches.end()) {
            // 未缓存，返回错误
            generatedCount = 0;
            return -1;
        }
    }
    
    // 获取缓存
    const SegmentCache& cache = segmentCaches[cacheKey];
    
    // 限制valueCount不超过可用值数量
    valueCount = std::min(valueCount, cache.count);
    
    // 计算结果的最大长度（包括null终止符）
    int maxResultLength = prefix.length() + cache.maxLength + 1;
    
    // 确保结果缓冲区足够大
    size_t requiredSize = valueCount * maxResultLength;
    ensureResultBufferSize(requiredSize, bufferIndex);
    
    // 分配和复制前缀到设备
    char* d_prefix;
    checkCudaError(cudaMalloc(&d_prefix, prefix.length()), "Allocating device prefix");
    checkCudaError(cudaMemcpyAsync(d_prefix, prefix.c_str(), prefix.length(), 
                             cudaMemcpyHostToDevice, cudaStreams[bufferIndex]), 
                  "Copying prefix to device");
    
    // 配置并启动内核
    int threadsPerBlock = 256;
    int blocksPerGrid = (valueCount + threadsPerBlock - 1) / threadsPerBlock;
    
    multiSegmentKernel<<<blocksPerGrid, threadsPerBlock, 0, cudaStreams[bufferIndex]>>>(
        d_resultBuffer[bufferIndex], cache.d_values, cache.d_offsets, cache.d_lengths, 
        valueCount, d_prefix, prefix.length(), maxResultLength
    );
    
    // 检查内核启动错误
    checkCudaError(cudaGetLastError(), "Launching multi segment kernel");
    
    // 分配主机内存用于结果
    std::vector<char> h_results(valueCount * maxResultLength);
    
    // 异步复制结果到主机
    checkCudaError(cudaMemcpyAsync(h_results.data(), d_resultBuffer[bufferIndex], valueCount * maxResultLength, 
                             cudaMemcpyDeviceToHost, cudaStreams[bufferIndex]), 
                  "Copying results from device");
    
    // 同步流
    cudaStreamSynchronize(cudaStreams[bufferIndex]);
    
    // 释放设备内存
    cudaFree(d_prefix);
    
    // 将结果转换为字符串
    for (int i = 0; i < valueCount; i++) {
        const char* str = h_results.data() + i * maxResultLength;
        results[offset + i] = str;  // 使用偏移直接写入results
    }
    
    // 设置生成数量
    generatedCount = valueCount;
    
    return 0;
}