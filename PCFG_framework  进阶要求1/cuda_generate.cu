#include "cuda_generate.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <omp.h>
#include <cstring>  // 添加这个头文件用于std::strcpy

// GPU资源锁，用于多线程环境
std::mutex ThreadSafe::gpu_mutex;

// 定义常量
#define MAX_STRING_LENGTH 64
#define MAX_BATCH_SIZE 32768
#define NUM_STREAMS 4
#define USE_TEXTURE_MEMORY 0
#define USE_VECTOR_OPS 0

// 错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// 段缓存结构
struct SegmentCache {
    int type;
    int length;
    char* d_values;          // 设备上的值
    int* d_valueLengths;     // 每个值的长度
    int* d_valueOffsets;     // 每个值的偏移量
    int numValues;           // 值的数量
    
    SegmentCache() : type(0), length(0), d_values(nullptr), 
                    d_valueLengths(nullptr), d_valueOffsets(nullptr), numValues(0) {}
};

// 双缓冲结构
struct DoubleBuffer {
    char* h_outputBuffer[2]; // 主机输出缓冲区
    char* d_outputBuffer[2]; // 设备输出缓冲区
};

// 全局资源
static std::unordered_map<std::string, SegmentCache> g_segmentCaches;
static DoubleBuffer g_doubleBuffer;
static cudaStream_t g_streams[NUM_STREAMS];
static cudaEvent_t startEvents[NUM_STREAMS], stopEvents[NUM_STREAMS];
static bool g_resourcesInitialized = false;

// GPU资源初始化器类
class GPUResourceInitializer {
public:
    GPUResourceInitializer() {
        initGPUResources();
    }
    
    ~GPUResourceInitializer() {
        freeGPUResources();
    }
};

// 静态初始化器实例
static GPUResourceInitializer g_resourceInitializer;

// 创建纹理引用
texture<char, 1, cudaReadModeElementType> texValues;
texture<int, 1, cudaReadModeElementType> texLengths;
texture<int, 1, cudaReadModeElementType> texOffsets;

// 初始化GPU资源
void initGPUResources() {
    // 防止重复初始化
    if (g_resourcesInitialized) return;
    
    std::lock_guard<std::mutex> lock(ThreadSafe::gpu_mutex);
    
    if (g_resourcesInitialized) return; // 二次检查
    
    // 分配双缓冲区
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaHostAlloc(&g_doubleBuffer.h_outputBuffer[i], 
                               MAX_BATCH_SIZE * MAX_STRING_LENGTH * sizeof(char), 
                               cudaHostAllocDefault));
        CUDA_CHECK(cudaMalloc(&g_doubleBuffer.d_outputBuffer[i], 
                            MAX_BATCH_SIZE * MAX_STRING_LENGTH * sizeof(char)));
    }
    
    // 创建CUDA流和事件
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&g_streams[i]));
        CUDA_CHECK(cudaEventCreate(&startEvents[i]));
        CUDA_CHECK(cudaEventCreate(&stopEvents[i]));
    }
    
    g_resourcesInitialized = true;
    std::cout << "GPU资源初始化完成" << std::endl;
}

// 释放GPU资源
void freeGPUResources() {
    if (!g_resourcesInitialized) return;
    
    std::lock_guard<std::mutex> lock(ThreadSafe::gpu_mutex);
    
    if (!g_resourcesInitialized) return; // 二次检查
    
    // 释放段缓存
    for (auto& pair : g_segmentCaches) {
        SegmentCache& cache = pair.second;
        if (cache.d_values) CUDA_CHECK(cudaFree(cache.d_values));
        if (cache.d_valueLengths) CUDA_CHECK(cudaFree(cache.d_valueLengths));
        if (cache.d_valueOffsets) CUDA_CHECK(cudaFree(cache.d_valueOffsets));
    }
    g_segmentCaches.clear();
    
    // 释放双缓冲区
    for (int i = 0; i < 2; i++) {
        if (g_doubleBuffer.h_outputBuffer[i]) CUDA_CHECK(cudaFreeHost(g_doubleBuffer.h_outputBuffer[i]));
        if (g_doubleBuffer.d_outputBuffer[i]) CUDA_CHECK(cudaFree(g_doubleBuffer.d_outputBuffer[i]));
    }
    
    // 释放CUDA流和事件
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamDestroy(g_streams[i]));
        CUDA_CHECK(cudaEventDestroy(startEvents[i]));
        CUDA_CHECK(cudaEventDestroy(stopEvents[i]));
    }
    
    g_resourcesInitialized = false;
    std::cout << "GPU资源已释放" << std::endl;
}

// 将segment值缓存到GPU
void cacheSegmentValues(int type, int length, const std::vector<std::string>& values) {
    std::lock_guard<std::mutex> lock(ThreadSafe::gpu_mutex);
    
    // 创建缓存键
    std::string cacheKey = std::to_string(type) + "_" + std::to_string(length);
    
    // 检查是否已缓存
    if (g_segmentCaches.find(cacheKey) != g_segmentCaches.end()) {
        return;
    }
    
    SegmentCache cache;
    cache.type = type;
    cache.length = length;
    cache.numValues = values.size();
    
    // 计算所需总大小并分配内存
    int totalSize = 0;
    std::vector<int> valueLengths(values.size());
    std::vector<int> valueOffsets(values.size());
    
    for (size_t i = 0; i < values.size(); i++) {
        valueLengths[i] = values[i].length() + 1; // +1 for null terminator
        valueOffsets[i] = totalSize;
        totalSize += valueLengths[i];
    }
    
    // 分配主机内存
    std::vector<char> h_values(totalSize);
    
    // 填充主机内存
    for (size_t i = 0; i < values.size(); i++) {
        strcpy(&h_values[valueOffsets[i]], values[i].c_str());  // 使用strcpy而非std::strcpy
    }
    
    // 分配并复制到设备内存
    CUDA_CHECK(cudaMalloc(&cache.d_values, totalSize * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&cache.d_valueLengths, cache.numValues * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cache.d_valueOffsets, cache.numValues * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(cache.d_values, h_values.data(), totalSize * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cache.d_valueLengths, valueLengths.data(), cache.numValues * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cache.d_valueOffsets, valueOffsets.data(), cache.numValues * sizeof(int), cudaMemcpyHostToDevice));
    
    // 保存缓存
    g_segmentCaches[cacheKey] = cache;
}

// 用于生成单段密码的CUDA内核
__global__ void singleSegmentKernel(char* outputBuffer, const char* values, 
                                   const int* valueLengths, const int* valueOffsets, 
                                   int valueCount, int maxOutputLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= valueCount) return;
    
    int offset = valueOffsets[idx];
    int length = valueLengths[idx] - 1;  // 减去NULL终止符
    int outPos = idx * maxOutputLen;
    
    // 逐字节复制
    #pragma unroll 4
    for (int i = 0; i <= length; i++) {  // 包括NULL终止符
        #if USE_TEXTURE_MEMORY
        outputBuffer[outPos + i] = tex1Dfetch<char>(texValues, offset + i);
        #else
        outputBuffer[outPos + i] = values[offset + i];
        #endif
    }
}

// 优化版本的单段生成内核
__global__ void optimizedSingleSegmentKernel(char* outputBuffer, 
                                           const char* values, 
                                           const int* valueLengths, 
                                           const int* valueOffsets, 
                                           int valueCount, int maxOutputLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= valueCount) return;
    
    #if USE_TEXTURE_MEMORY
    int offset = tex1Dfetch<int>(texOffsets, idx);
    int length = tex1Dfetch<int>(texLengths, idx) - 1;  // 减去NULL终止符
    #else
    int offset = valueOffsets[idx];
    int length = valueLengths[idx] - 1;  // 减去NULL终止符
    #endif
    
    int outPos = idx * maxOutputLen;
    
    #if USE_VECTOR_OPS
    // 使用向量操作复制（每次复制4字节）
    typedef int4 CopyType;
    CopyType* srcPtr = (CopyType*)(values + offset);
    CopyType* dstPtr = (CopyType*)(outputBuffer + outPos);
    
    int vectorCount = length / sizeof(CopyType);
    for (int i = 0; i < vectorCount; i++) {
        dstPtr[i] = srcPtr[i];
    }
    
    // 复制剩余字节
    int remainingStart = vectorCount * sizeof(CopyType);
    for (int i = remainingStart; i <= length; i++) {
        outputBuffer[outPos + i] = values[offset + i];
    }
    #else
    // 逐字节复制
    #pragma unroll 4
    for (int i = 0; i <= length; i++) {
        #if USE_TEXTURE_MEMORY
        outputBuffer[outPos + i] = tex1Dfetch<char>(texValues, offset + i);
        #else
        outputBuffer[outPos + i] = values[offset + i];
        #endif
    }
    #endif
}

// 多段密码生成内核
__global__ void multiSegmentKernel(char* outputBuffer, const char* values,
                                  const int* valueLengths, const int* valueOffsets,
                                  int valueCount, const char* prefix, int prefixLength, 
                                  int maxOutputLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= valueCount) return;
    
    #if USE_TEXTURE_MEMORY
    int offset = tex1Dfetch<int>(texOffsets, idx);
    int length = tex1Dfetch<int>(texLengths, idx) - 1;  // 减去NULL终止符
    #else
    int offset = valueOffsets[idx];
    int length = valueLengths[idx] - 1;  // 减去NULL终止符
    #endif
    
    int outPos = idx * maxOutputLen;
    
    // 复制前缀
    for (int i = 0; i < prefixLength; i++) {
        outputBuffer[outPos + i] = prefix[i];
    }
    
    // 复制当前段值
    #pragma unroll 4
    for (int i = 0; i <= length; i++) {  // 包括NULL终止符
        #if USE_TEXTURE_MEMORY
        outputBuffer[outPos + prefixLength + i] = tex1Dfetch<char>(texValues, offset + i);
        #else
        outputBuffer[outPos + prefixLength + i] = values[offset + i];
        #endif
    }
}

// 线程安全版本的单段密码生成
void ThreadSafe::generateSingleSegmentGPU(int segmentType, int segmentLength, int valueCount, 
                                        std::vector<std::string>& guesses, int& total_guesses) {
    std::lock_guard<std::mutex> lock(gpu_mutex);
    
    // 调用原始函数
    ::generateSingleSegmentGPU(segmentType, segmentLength, valueCount, guesses, total_guesses);
}

// 线程安全版本的多段密码生成
void ThreadSafe::generateMultiSegmentGPU(const std::string& prefix, int lastSegmentType, 
                                       int lastSegmentLength, int valueCount,
                                       std::vector<std::string>& guesses, int& total_guesses) {
    std::lock_guard<std::mutex> lock(gpu_mutex);
    
    // 调用原始函数
    ::generateMultiSegmentGPU(prefix, lastSegmentType, lastSegmentLength, valueCount, guesses, total_guesses);
}

// 在GPU上生成单段密码
void generateSingleSegmentGPU(int segmentType, int segmentLength, int valueCount, 
                             std::vector<std::string>& guesses, int& total_guesses) {
    // 确保GPU资源已初始化
    if (!g_resourcesInitialized) {
        initGPUResources();
    }
    
    // 检查并获取缓存
    std::string cacheKey = std::to_string(segmentType) + "_" + std::to_string(segmentLength);
    if (g_segmentCaches.find(cacheKey) == g_segmentCaches.end()) {
        std::cerr << "错误：未找到缓存的段值: " << cacheKey << std::endl;
        return;
    }
    
    SegmentCache& cache = g_segmentCaches[cacheKey];
    // 修复变量作用域问题
    int numValuesToProcess = std::min(valueCount, cache.numValues);
    
    // 存储原始猜测大小
    size_t originalSize = guesses.size();
    guesses.resize(originalSize + numValuesToProcess);
    
    int prevBuffer = -1;
    int prevBatchSize = 0;
    int processedValues = 0;
    
    // 批量处理值
    for (processedValues = 0; processedValues < numValuesToProcess; processedValues += MAX_BATCH_SIZE) {
        int batchSize = std::min(MAX_BATCH_SIZE, numValuesToProcess - processedValues);
        int streamIdx = processedValues % NUM_STREAMS;
        int currentBuffer = streamIdx % 2;
        
        // 如果前一批次结果已准备好，处理它们
        if (prevBuffer != -1) {
            cudaEventSynchronize(stopEvents[streamIdx]);
            
            // 处理上一批次的结果
            #pragma omp parallel for
            for (int i = 0; i < prevBatchSize; i++) {
                int idx = processedValues - MAX_BATCH_SIZE + i;
                guesses[originalSize + idx] = std::string(g_doubleBuffer.h_outputBuffer[prevBuffer] + (i * MAX_STRING_LENGTH));
            }
        }
        
        // 设置纹理引用（如果使用）
        #if USE_TEXTURE_MEMORY
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<char>();
        cudaBindTexture(0, texValues, cache.d_values, channelDesc);
        cudaBindTexture(0, texLengths, cache.d_valueLengths, cudaCreateChannelDesc<int>());
        cudaBindTexture(0, texOffsets, cache.d_valueOffsets, cudaCreateChannelDesc<int>());
        #endif
        
        // 启动内核
        int blockSize = 256;
        int numBlocks = (batchSize + blockSize - 1) / blockSize;
        
        // 记录开始事件
        CUDA_CHECK(cudaEventRecord(startEvents[streamIdx], g_streams[streamIdx]));
        
        // 启动优化内核
        optimizedSingleSegmentKernel<<<numBlocks, blockSize, 0, g_streams[streamIdx]>>>(
            g_doubleBuffer.d_outputBuffer[currentBuffer],
            cache.d_values,
            cache.d_valueLengths,
            cache.d_valueOffsets,
            batchSize,
            MAX_STRING_LENGTH
        );
        
        // 检查错误
        CUDA_CHECK(cudaGetLastError());
        
        // 记录结束事件
        CUDA_CHECK(cudaEventRecord(stopEvents[streamIdx], g_streams[streamIdx]));
        
        // 异步复制结果回主机
        CUDA_CHECK(cudaMemcpyAsync(g_doubleBuffer.h_outputBuffer[currentBuffer], 
                                 g_doubleBuffer.d_outputBuffer[streamIdx], 
                                 batchSize * MAX_STRING_LENGTH * sizeof(char), 
                                 cudaMemcpyDeviceToHost, 
                                 g_streams[streamIdx]));
        
        // 更新前一批次信息
        prevBuffer = currentBuffer;
        prevBatchSize = batchSize;
    }
    
    // 处理最后一批次
    if (prevBuffer != -1) {
        int streamIdx = (processedValues - 1) % NUM_STREAMS;
        cudaEventSynchronize(stopEvents[streamIdx]);
        
        // 处理最后一批次的结果
        #pragma omp parallel for
        for (int i = 0; i < prevBatchSize; i++) {
            int idx = processedValues - prevBatchSize + i;
            if (idx < numValuesToProcess) {
                guesses[originalSize + idx] = std::string(g_doubleBuffer.h_outputBuffer[prevBuffer] + (i * MAX_STRING_LENGTH));
            }
        }
    }
    
    // 更新总猜测数
    total_guesses += numValuesToProcess;
    
    // 解绑纹理（如果使用）
    #if USE_TEXTURE_MEMORY
    cudaUnbindTexture(texValues);
    cudaUnbindTexture(texLengths);
    cudaUnbindTexture(texOffsets);
    #endif
}

// 在GPU上生成多段密码
void generateMultiSegmentGPU(const std::string& prefix, int lastSegmentType, 
                            int lastSegmentLength, int valueCount,
                            std::vector<std::string>& guesses, int& total_guesses) {
    // 确保GPU资源已初始化
    if (!g_resourcesInitialized) {
        initGPUResources();
    }
    
    // 检查并获取缓存
    std::string cacheKey = std::to_string(lastSegmentType) + "_" + std::to_string(lastSegmentLength);
    if (g_segmentCaches.find(cacheKey) == g_segmentCaches.end()) {
        std::cerr << "错误：未找到缓存的段值: " << cacheKey << std::endl;
        return;
    }
    
    SegmentCache& cache = g_segmentCaches[cacheKey];
    int numValuesToProcess = std::min(valueCount, cache.numValues);
    
    // 存储原始猜测大小
    size_t originalSize = guesses.size();
    guesses.resize(originalSize + numValuesToProcess);
    
    // 将前缀复制到设备内存
    char* d_prefix;
    CUDA_CHECK(cudaMalloc(&d_prefix, prefix.length() + 1));
    CUDA_CHECK(cudaMemcpy(d_prefix, prefix.c_str(), prefix.length() + 1, cudaMemcpyHostToDevice));
    
    int prevBuffer = -1;
    int prevBatchSize = 0;
    int processedValues = 0;
    
    // 批量处理值
    for (processedValues = 0; processedValues < numValuesToProcess; processedValues += MAX_BATCH_SIZE) {
        int batchSize = std::min(MAX_BATCH_SIZE, numValuesToProcess - processedValues);
        int streamIdx = processedValues % NUM_STREAMS;
        int currentBuffer = streamIdx % 2;
        
        // 如果前一批次结果已准备好，处理它们
        if (prevBuffer != -1) {
            cudaEventSynchronize(stopEvents[streamIdx]);
            
            // 处理上一批次的结果
            #pragma omp parallel for
            for (int i = 0; i < prevBatchSize; i++) {
                int idx = processedValues - MAX_BATCH_SIZE + i;
                guesses[originalSize + idx] = std::string(g_doubleBuffer.h_outputBuffer[prevBuffer] + (i * MAX_STRING_LENGTH));
            }
        }
        
        // 设置纹理引用（如果使用）
        #if USE_TEXTURE_MEMORY
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<char>();
        cudaBindTexture(0, texValues, cache.d_values, channelDesc);
        cudaBindTexture(0, texLengths, cache.d_valueLengths, cudaCreateChannelDesc<int>());
        cudaBindTexture(0, texOffsets, cache.d_valueOffsets, cudaCreateChannelDesc<int>());
        #endif
        
        // 启动内核
        int blockSize = 256;
        int numBlocks = (batchSize + blockSize - 1) / blockSize;
        
        // 记录开始事件
        CUDA_CHECK(cudaEventRecord(startEvents[streamIdx], g_streams[streamIdx]));
        
        // 启动多段内核
        multiSegmentKernel<<<numBlocks, blockSize, 0, g_streams[streamIdx]>>>(
            g_doubleBuffer.d_outputBuffer[currentBuffer],
            cache.d_values,
            cache.d_valueLengths,
            cache.d_valueOffsets,
            batchSize,
            d_prefix,
            prefix.length(),
            MAX_STRING_LENGTH
        );
        
        // 检查错误
        CUDA_CHECK(cudaGetLastError());
        
        // 记录结束事件
        CUDA_CHECK(cudaEventRecord(stopEvents[streamIdx], g_streams[streamIdx]));
        
        // 异步复制结果回主机
        CUDA_CHECK(cudaMemcpyAsync(g_doubleBuffer.h_outputBuffer[currentBuffer], 
                                 g_doubleBuffer.d_outputBuffer[streamIdx], 
                                 batchSize * MAX_STRING_LENGTH * sizeof(char), 
                                 cudaMemcpyDeviceToHost, 
                                 g_streams[streamIdx]));
        
        // 更新前一批次信息
        prevBuffer = currentBuffer;
        prevBatchSize = batchSize;
    }
    
    // 处理最后一批次
    if (prevBuffer != -1) {
        int streamIdx = (processedValues - 1) % NUM_STREAMS;
        cudaEventSynchronize(stopEvents[streamIdx]);
        
        // 处理最后一批次的结果
        #pragma omp parallel for
        for (int i = 0; i < prevBatchSize; i++) {
            int idx = processedValues - prevBatchSize + i;
            if (idx < numValuesToProcess) {
                guesses[originalSize + idx] = std::string(g_doubleBuffer.h_outputBuffer[prevBuffer] + (i * MAX_STRING_LENGTH));
            }
        }
    }
    
    // 释放设备内存
    CUDA_CHECK(cudaFree(d_prefix));
    
    // 更新总猜测数
    total_guesses += numValuesToProcess;
    
    // 解绑纹理（如果使用）
    #if USE_TEXTURE_MEMORY
    cudaUnbindTexture(texValues);
    cudaUnbindTexture(texLengths);
    cudaUnbindTexture(texOffsets);
    #endif
}