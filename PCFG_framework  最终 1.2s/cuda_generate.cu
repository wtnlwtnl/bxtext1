#include "cuda_generate.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

// 错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 配置参数
#define MAX_SEGMENT_TYPES 10  // 扩大类型范围
#define MAX_SEGMENT_LENGTH 50 // 增加最大长度
#define MAX_BATCH_SIZE 32768  // 增大批处理大小
#define NUM_STREAMS 8         // 增加流数量
#define MAX_STRING_LENGTH 256 // 最大字符串长度
#define USE_TEXTURE_MEMORY 1  // 使用纹理内存
#define USE_VECTOR_OPS 0      // 禁用向量操作以避免内存对齐问题

// 缓存结构
struct SegmentCache {
    bool initialized;
    int numValues;
    char* d_values;      // 设备上的值数组
    int* d_valueLengths; // 每个值的长度
    int* d_valueOffsets; // 每个值在数组中的偏移
    size_t totalSize;    // 总内存大小
    
    #if USE_TEXTURE_MEMORY
    cudaTextureObject_t texObj; // 纹理对象
    #endif
};

// 全局缓存数组
SegmentCache g_segmentCache[MAX_SEGMENT_TYPES][MAX_SEGMENT_LENGTH];

// 全局CUDA流
cudaStream_t g_streams[NUM_STREAMS];
bool g_streamsInitialized = false;

// 常量内存前缀缓冲区
__constant__ char d_constPrefix[1024];

// 事件用于精确同步
cudaEvent_t startEvents[NUM_STREAMS];
cudaEvent_t stopEvents[NUM_STREAMS];

// 双缓冲数据结构
typedef struct {
    char* h_outputBuffer[2];
    int* h_outputIndices[2];
    char* d_outputBuffer[NUM_STREAMS];
    int* d_outputIndices[NUM_STREAMS];
    bool initialized;
} DoubleBuffer;

DoubleBuffer g_doubleBuffer = {0};

// 创建纹理对象
#if USE_TEXTURE_MEMORY
cudaTextureObject_t createTextureObject(const void* devPtr, size_t size) {
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = (void*)devPtr;
    resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    resDesc.res.linear.desc.x = 8; // 8 bits for char
    resDesc.res.linear.desc.y = 0;
    resDesc.res.linear.desc.z = 0;
    resDesc.res.linear.desc.w = 0;
    resDesc.res.linear.sizeInBytes = size;

    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;

    cudaTextureObject_t texObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    return texObj;
}
#endif

// 优化的单段字符串生成内核 - 使用共享内存
__global__ void optimizedSingleSegmentKernel(
    #if USE_TEXTURE_MEMORY
    cudaTextureObject_t texValues,
    #else
    const char* __restrict__ values,
    #endif
    const int* __restrict__ valueLengths,
    const int* __restrict__ valueOffsets,
    char* __restrict__ outputBuffer,
    int* __restrict__ outputIndices,
    int numValues,
    int startIdx,
    int batchSize) {
    
    // 线程和块索引
    int localIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (localIdx >= batchSize) return;
    
    int valueIdx = startIdx + localIdx;
    if (valueIdx >= numValues) return;
    
    // 使用共享内存缓存频繁访问的值
    __shared__ int s_lengths[256];
    __shared__ int s_offsets[256];
    
    // 协作加载到共享内存
    if (threadIdx.x < batchSize && threadIdx.x < 256) {
        s_lengths[threadIdx.x] = valueLengths[valueIdx];
        s_offsets[threadIdx.x] = valueOffsets[valueIdx];
    }
    __syncthreads();
    
    // 获取当前值的长度和偏移
    int length = (threadIdx.x < 256) ? s_lengths[threadIdx.x] : valueLengths[valueIdx];
    int offset = (threadIdx.x < 256) ? s_offsets[threadIdx.x] : valueOffsets[valueIdx];
    
    // 存储输出索引
    outputIndices[localIdx] = valueIdx;
    
    // 计算输出位置
    int outPos = localIdx * MAX_STRING_LENGTH;
    
    // 标准内存拷贝 - 安全且无对齐问题
    #pragma unroll 4
    for (int i = 0; i <= length; i++) {
        #if USE_TEXTURE_MEMORY
        outputBuffer[outPos + i] = tex1Dfetch<char>(texValues, offset + i);
        #else
        outputBuffer[outPos + i] = values[offset + i];
        #endif
    }
}

// 优化的多段字符串生成内核
__global__ void optimizedMultiSegmentKernel(
    #if USE_TEXTURE_MEMORY
    cudaTextureObject_t texValues,
    #else
    const char* __restrict__ values,
    #endif
    const int* __restrict__ valueLengths,
    const int* __restrict__ valueOffsets,
    char* __restrict__ outputBuffer,
    int* __restrict__ outputIndices,
    int numValues,
    int startIdx,
    int batchSize,
    const char* prefix,
    int prefixLength) {
    
    // 线程和块索引
    int localIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (localIdx >= batchSize) return;
    
    int valueIdx = startIdx + localIdx;
    if (valueIdx >= numValues) return;
    
    // 使用共享内存缓存频繁访问的值
    __shared__ int s_lengths[256];
    __shared__ int s_offsets[256];
    __shared__ char s_prefix[256];
    
    // 协作加载到共享内存
    if (threadIdx.x < batchSize && threadIdx.x < 256) {
        s_lengths[threadIdx.x] = valueLengths[valueIdx];
        s_offsets[threadIdx.x] = valueOffsets[valueIdx];
    }
    
    // 协作加载前缀到共享内存
    for (int i = threadIdx.x; i < prefixLength && i < 256; i += blockDim.x) {
        s_prefix[i] = prefix[i];
    }
    __syncthreads();
    
    // 获取当前值的长度和偏移
    int length = (threadIdx.x < 256) ? s_lengths[threadIdx.x] : valueLengths[valueIdx];
    int offset = (threadIdx.x < 256) ? s_offsets[threadIdx.x] : valueOffsets[valueIdx];
    
    // 存储输出索引
    outputIndices[localIdx] = valueIdx;
    
    // 计算输出位置
    int outPos = localIdx * MAX_STRING_LENGTH;
    
    // 复制前缀 - 使用安全的逐字节复制
    #pragma unroll 4
    for (int i = 0; i < prefixLength; i++) {
        outputBuffer[outPos + i] = s_prefix[i];
    }
    
    // 复制后缀 - 使用安全的逐字节复制
    #pragma unroll 4
    for (int i = 0; i <= length; i++) {
        #if USE_TEXTURE_MEMORY
        outputBuffer[outPos + prefixLength + i] = tex1Dfetch<char>(texValues, offset + i);
        #else
        outputBuffer[outPos + prefixLength + i] = values[offset + i];
        #endif
    }
}

// 优化的使用常量内存的多段字符串生成内核
__global__ void optimizedConstantMemoryKernel(
    #if USE_TEXTURE_MEMORY
    cudaTextureObject_t texValues,
    #else
    const char* __restrict__ values,
    #endif
    const int* __restrict__ valueLengths,
    const int* __restrict__ valueOffsets,
    char* __restrict__ outputBuffer,
    int* __restrict__ outputIndices,
    int numValues,
    int startIdx,
    int batchSize,
    int prefixLength) {
    
    // 线程和块索引
    int localIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (localIdx >= batchSize) return;
    
    int valueIdx = startIdx + localIdx;
    if (valueIdx >= numValues) return;
    
    // 使用共享内存缓存频繁访问的值
    __shared__ int s_lengths[256];
    __shared__ int s_offsets[256];
    
    // 协作加载到共享内存
    if (threadIdx.x < batchSize && threadIdx.x < 256) {
        s_lengths[threadIdx.x] = valueLengths[valueIdx];
        s_offsets[threadIdx.x] = valueOffsets[valueIdx];
    }
    __syncthreads();
    
    // 获取当前值的长度和偏移
    int length = (threadIdx.x < 256) ? s_lengths[threadIdx.x] : valueLengths[valueIdx];
    int offset = (threadIdx.x < 256) ? s_offsets[threadIdx.x] : valueOffsets[valueIdx];
    
    // 存储输出索引
    outputIndices[localIdx] = valueIdx;
    
    // 计算输出位置
    int outPos = localIdx * MAX_STRING_LENGTH;
    
    // 复制前缀（从常量内存）
    #pragma unroll 4
    for (int i = 0; i < prefixLength; i++) {
        outputBuffer[outPos + i] = d_constPrefix[i];
    }
    
    // 复制后缀 - 使用安全的逐字节复制
    #pragma unroll 4
    for (int i = 0; i <= length; i++) {
        #if USE_TEXTURE_MEMORY
        outputBuffer[outPos + prefixLength + i] = tex1Dfetch<char>(texValues, offset + i);
        #else
        outputBuffer[outPos + prefixLength + i] = values[offset + i];
        #endif
    }
}

// 初始化CUDA流和事件
void initStreams() {
    if (!g_streamsInitialized) {
        for (int i = 0; i < NUM_STREAMS; i++) {
            CUDA_CHECK(cudaStreamCreate(&g_streams[i]));
            CUDA_CHECK(cudaEventCreate(&startEvents[i]));
            CUDA_CHECK(cudaEventCreate(&stopEvents[i]));
        }
        g_streamsInitialized = true;
    }
}

// 初始化双缓冲
void initDoubleBuffer() {
    if (!g_doubleBuffer.initialized) {
        // 分配主机固定内存
        for (int i = 0; i < 2; i++) {
            CUDA_CHECK(cudaHostAlloc(&g_doubleBuffer.h_outputBuffer[i], 
                                    MAX_BATCH_SIZE * MAX_STRING_LENGTH * sizeof(char), 
                                    cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc(&g_doubleBuffer.h_outputIndices[i], 
                                    MAX_BATCH_SIZE * sizeof(int), 
                                    cudaHostAllocDefault));
        }
        
        // 分配设备内存
        for (int i = 0; i < NUM_STREAMS; i++) {
            CUDA_CHECK(cudaMalloc(&g_doubleBuffer.d_outputBuffer[i], 
                                 MAX_BATCH_SIZE * MAX_STRING_LENGTH * sizeof(char)));
            CUDA_CHECK(cudaMalloc(&g_doubleBuffer.d_outputIndices[i], 
                                 MAX_BATCH_SIZE * sizeof(int)));
        }
        
        g_doubleBuffer.initialized = true;
    }
}

// 释放双缓冲
void freeDoubleBuffer() {
    if (g_doubleBuffer.initialized) {
        for (int i = 0; i < 2; i++) {
            CUDA_CHECK(cudaFreeHost(g_doubleBuffer.h_outputBuffer[i]));
            CUDA_CHECK(cudaFreeHost(g_doubleBuffer.h_outputIndices[i]));
        }
        
        for (int i = 0; i < NUM_STREAMS; i++) {
            CUDA_CHECK(cudaFree(g_doubleBuffer.d_outputBuffer[i]));
            CUDA_CHECK(cudaFree(g_doubleBuffer.d_outputIndices[i]));
        }
        
        g_doubleBuffer.initialized = false;
    }
}

// 初始化GPU资源
void initGPUResources() {
    // 初始化流和事件
    initStreams();
    
    // 初始化双缓冲
    initDoubleBuffer();
    
    // 初始化缓存数组
    for (int i = 0; i < MAX_SEGMENT_TYPES; i++) {
        for (int j = 0; j < MAX_SEGMENT_LENGTH; j++) {
            g_segmentCache[i][j].initialized = false;
        }
    }
    
    // 预热GPU
    cudaFree(0);
}

// 释放GPU资源
void freeGPUResources() {
    // 释放所有缓存的segment数据
    for (int i = 0; i < MAX_SEGMENT_TYPES; i++) {
        for (int j = 0; j < MAX_SEGMENT_LENGTH; j++) {
            if (g_segmentCache[i][j].initialized) {
                #if USE_TEXTURE_MEMORY
                CUDA_CHECK(cudaDestroyTextureObject(g_segmentCache[i][j].texObj));
                #endif
                
                CUDA_CHECK(cudaFree(g_segmentCache[i][j].d_values));
                CUDA_CHECK(cudaFree(g_segmentCache[i][j].d_valueLengths));
                CUDA_CHECK(cudaFree(g_segmentCache[i][j].d_valueOffsets));
                g_segmentCache[i][j].initialized = false;
            }
        }
    }
    
    // 释放双缓冲
    freeDoubleBuffer();
    
    // 释放流和事件
    if (g_streamsInitialized) {
        for (int i = 0; i < NUM_STREAMS; i++) {
            CUDA_CHECK(cudaStreamDestroy(g_streams[i]));
            CUDA_CHECK(cudaEventDestroy(startEvents[i]));
            CUDA_CHECK(cudaEventDestroy(stopEvents[i]));
        }
        g_streamsInitialized = false;
    }
}

// 预缓存segment值到GPU
void cacheSegmentValues(int segmentType, int segmentLength, const std::vector<std::string>& values) {
    // 检查缓存索引有效性
    if (segmentType < 1 || segmentType >= MAX_SEGMENT_TYPES || 
        segmentLength < 1 || segmentLength >= MAX_SEGMENT_LENGTH) {
        return; // 无效索引，安静地返回
    }
    
    // 检查是否已缓存
    if (g_segmentCache[segmentType][segmentLength].initialized) {
        return;
    }
    
    // 初始化新缓存条目
    SegmentCache& cache = g_segmentCache[segmentType][segmentLength];
    cache.initialized = false;
    cache.numValues = values.size();
    
    if (cache.numValues == 0) {
        return;
    }
    
    // 计算总内存需求
    size_t totalSize = 0;
    int* valueLengths = (int*)malloc(cache.numValues * sizeof(int));
    int* valueOffsets = (int*)malloc(cache.numValues * sizeof(int));
    
    for (int i = 0; i < cache.numValues; i++) {
        valueLengths[i] = values[i].length();
        valueOffsets[i] = totalSize;
        totalSize += valueLengths[i] + 1; // +1 for null terminator
    }
    cache.totalSize = totalSize;
    
    // 分配主机固定内存并填充数据
    char* h_values;
    CUDA_CHECK(cudaHostAlloc(&h_values, totalSize * sizeof(char), cudaHostAllocDefault));
    
    for (int i = 0; i < cache.numValues; i++) {
        memcpy(h_values + valueOffsets[i], values[i].c_str(), valueLengths[i] + 1);
    }
    
    // 分配设备内存
    CUDA_CHECK(cudaMalloc(&cache.d_values, totalSize * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&cache.d_valueLengths, cache.numValues * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cache.d_valueOffsets, cache.numValues * sizeof(int)));
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(cache.d_values, h_values, totalSize * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cache.d_valueLengths, valueLengths, cache.numValues * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cache.d_valueOffsets, valueOffsets, cache.numValues * sizeof(int), cudaMemcpyHostToDevice));
    
    // 创建纹理对象
    #if USE_TEXTURE_MEMORY
    cache.texObj = createTextureObject(cache.d_values, totalSize * sizeof(char));
    #endif
    
    cache.initialized = true;
    
    // 释放主机内存
    CUDA_CHECK(cudaFreeHost(h_values));
    free(valueLengths);
    free(valueOffsets);
}

// 使用预缓存的数据生成单段猜测
void generateSingleSegmentGPU(int segmentType, int segmentLength, int maxIndices,
                             std::vector<std::string>& guesses, int& total_guesses) {
    // 检查缓存有效性
    if (segmentType < 1 || segmentType >= MAX_SEGMENT_TYPES || 
        segmentLength < 1 || segmentLength >= MAX_SEGMENT_LENGTH) {
        return; // 无效索引，安静地返回
    }
    
    SegmentCache& cache = g_segmentCache[segmentType][segmentLength];
    if (!cache.initialized || cache.numValues == 0) {
        return;
    }
    
    // 确保不超出可用值的数量
    int numValues = (maxIndices < cache.numValues) ? maxIndices : cache.numValues;
    
    // 预分配结果空间
    int originalSize = guesses.size();
    guesses.resize(originalSize + numValues);
    
    // 初始化双缓冲索引
    int currentBuffer = 0;
    
    // 批量处理值
    for (int processedValues = 0; processedValues < numValues; processedValues += MAX_BATCH_SIZE) {
        int batchSize = std::min(MAX_BATCH_SIZE, numValues - processedValues);
        int streamIdx = processedValues % NUM_STREAMS;
        
        // 启动内核
        int blockSize = 256;
        int numBlocks = (batchSize + blockSize - 1) / blockSize;
        
        // 记录开始事件
        CUDA_CHECK(cudaEventRecord(startEvents[streamIdx], g_streams[streamIdx]));
        
        // 启动优化内核
        optimizedSingleSegmentKernel<<<numBlocks, blockSize, 0, g_streams[streamIdx]>>>(
            #if USE_TEXTURE_MEMORY
            cache.texObj,
            #else
            cache.d_values,
            #endif
            cache.d_valueLengths,
            cache.d_valueOffsets,
            g_doubleBuffer.d_outputBuffer[streamIdx],
            g_doubleBuffer.d_outputIndices[streamIdx],
            cache.numValues,
            processedValues,
            batchSize
        );
        
        // 异步复制结果回主机
        CUDA_CHECK(cudaMemcpyAsync(g_doubleBuffer.h_outputBuffer[currentBuffer], 
                                 g_doubleBuffer.d_outputBuffer[streamIdx], 
                                 batchSize * MAX_STRING_LENGTH * sizeof(char), 
                                 cudaMemcpyDeviceToHost, 
                                 g_streams[streamIdx]));
        
        // 记录结束事件
        CUDA_CHECK(cudaEventRecord(stopEvents[streamIdx], g_streams[streamIdx]));
        
        // 如果不是第一批，处理前一批结果
        if (processedValues > 0) {
            int prevStreamIdx = (processedValues - MAX_BATCH_SIZE) % NUM_STREAMS;
            
            // 等待前一批次完成
            CUDA_CHECK(cudaEventSynchronize(stopEvents[prevStreamIdx]));
            
            // 处理前一批次结果
            int prevBatchSize = std::min(MAX_BATCH_SIZE, numValues - (processedValues - MAX_BATCH_SIZE));
            int prevBuffer = 1 - currentBuffer;
            
            // 转换结果为std::string
            #pragma omp parallel for
            for (int i = 0; i < prevBatchSize; i++) {
                int idx = processedValues - MAX_BATCH_SIZE + i;
                guesses[originalSize + idx] = std::string(g_doubleBuffer.h_outputBuffer[prevBuffer] + (i * MAX_STRING_LENGTH));
            }
        }
        
        // 切换缓冲区
        currentBuffer = 1 - currentBuffer;
    }
    
    // 处理最后一批结果
    int lastBatchSize = numValues % MAX_BATCH_SIZE;
    if (lastBatchSize == 0 && numValues > 0) {
        lastBatchSize = MAX_BATCH_SIZE;
    }
    
    int lastStreamIdx = (numValues - lastBatchSize) % NUM_STREAMS;
    
    // 等待最后一批完成
    CUDA_CHECK(cudaEventSynchronize(stopEvents[lastStreamIdx]));
    
    // 处理最后一批结果
    #pragma omp parallel for
    for (int i = 0; i < lastBatchSize; i++) {
        int idx = numValues - lastBatchSize + i;
        guesses[originalSize + idx] = std::string(g_doubleBuffer.h_outputBuffer[1 - currentBuffer] + (i * MAX_STRING_LENGTH));
    }
    
    total_guesses += numValues;
}

// 使用预缓存的数据生成多段猜测
void generateMultiSegmentGPU(const std::string& prefix, int segmentType, int segmentLength, 
                            int maxIndices, std::vector<std::string>& guesses, int& total_guesses) {
    // 检查缓存有效性
    if (segmentType < 1 || segmentType >= MAX_SEGMENT_TYPES || 
        segmentLength < 1 || segmentLength >= MAX_SEGMENT_LENGTH) {
        return; // 无效索引，安静地返回
    }
    
    SegmentCache& cache = g_segmentCache[segmentType][segmentLength];
    if (!cache.initialized || cache.numValues == 0) {
        return;
    }
    
    // 确保不超出可用值的数量
    int numValues = (maxIndices < cache.numValues) ? maxIndices : cache.numValues;
    int prefixLength = prefix.length();
    
    // 检查前缀长度是否在限制内
    if (prefixLength > 1000) {
        return; // 前缀太长
    }
    
    // 预分配结果空间
    int originalSize = guesses.size();
    guesses.resize(originalSize + numValues);
    
    // 将前缀复制到设备内存或常量内存
    bool useConstantMemory = (prefixLength <= 1024);
    char* d_prefix = NULL;
    
    if (useConstantMemory) {
        // 使用常量内存
        CUDA_CHECK(cudaMemcpyToSymbol(d_constPrefix, prefix.c_str(), 
                                     (prefixLength + 1) * sizeof(char)));
    } else {
        // 使用普通设备内存
        CUDA_CHECK(cudaMalloc(&d_prefix, (prefixLength + 1) * sizeof(char)));
        CUDA_CHECK(cudaMemcpy(d_prefix, prefix.c_str(), 
                            (prefixLength + 1) * sizeof(char), 
                            cudaMemcpyHostToDevice));
    }
    
    // 初始化双缓冲索引
    int currentBuffer = 0;
    
    // 批量处理值
    for (int processedValues = 0; processedValues < numValues; processedValues += MAX_BATCH_SIZE) {
        int batchSize = std::min(MAX_BATCH_SIZE, numValues - processedValues);
        int streamIdx = processedValues % NUM_STREAMS;
        
        // 启动内核
        int blockSize = 256;
        int numBlocks = (batchSize + blockSize - 1) / blockSize;
        
        // 记录开始事件
        CUDA_CHECK(cudaEventRecord(startEvents[streamIdx], g_streams[streamIdx]));
        
        if (useConstantMemory) {
            // 使用常量内存内核
            optimizedConstantMemoryKernel<<<numBlocks, blockSize, 0, g_streams[streamIdx]>>>(
                #if USE_TEXTURE_MEMORY
                cache.texObj,
                #else
                cache.d_values,
                #endif
                cache.d_valueLengths,
                cache.d_valueOffsets,
                g_doubleBuffer.d_outputBuffer[streamIdx],
                g_doubleBuffer.d_outputIndices[streamIdx],
                cache.numValues,
                processedValues,
                batchSize,
                prefixLength
            );
        } else {
            // 使用普通内核
            optimizedMultiSegmentKernel<<<numBlocks, blockSize, 0, g_streams[streamIdx]>>>(
                #if USE_TEXTURE_MEMORY
                cache.texObj,
                #else
                cache.d_values,
                #endif
                cache.d_valueLengths,
                cache.d_valueOffsets,
                g_doubleBuffer.d_outputBuffer[streamIdx],
                g_doubleBuffer.d_outputIndices[streamIdx],
                cache.numValues,
                processedValues,
                batchSize,
                d_prefix,
                prefixLength
            );
        }
        
        // 异步复制结果回主机
        CUDA_CHECK(cudaMemcpyAsync(g_doubleBuffer.h_outputBuffer[currentBuffer], 
                                 g_doubleBuffer.d_outputBuffer[streamIdx], 
                                 batchSize * MAX_STRING_LENGTH * sizeof(char), 
                                 cudaMemcpyDeviceToHost, 
                                 g_streams[streamIdx]));
        
        // 记录结束事件
        CUDA_CHECK(cudaEventRecord(stopEvents[streamIdx], g_streams[streamIdx]));
        
        // 如果不是第一批，处理前一批结果
        if (processedValues > 0) {
            int prevStreamIdx = (processedValues - MAX_BATCH_SIZE) % NUM_STREAMS;
            
            // 等待前一批次完成
            CUDA_CHECK(cudaEventSynchronize(stopEvents[prevStreamIdx]));
            
            // 处理前一批次结果
            int prevBatchSize = std::min(MAX_BATCH_SIZE, numValues - (processedValues - MAX_BATCH_SIZE));
            int prevBuffer = 1 - currentBuffer;
            
            // 转换结果为std::string
            #pragma omp parallel for
            for (int i = 0; i < prevBatchSize; i++) {
                int idx = processedValues - MAX_BATCH_SIZE + i;
                guesses[originalSize + idx] = std::string(g_doubleBuffer.h_outputBuffer[prevBuffer] + (i * MAX_STRING_LENGTH));
            }
        }
        
        // 切换缓冲区
        currentBuffer = 1 - currentBuffer;
    }
    
    // 处理最后一批结果
    int lastBatchSize = numValues % MAX_BATCH_SIZE;
    if (lastBatchSize == 0 && numValues > 0) {
        lastBatchSize = MAX_BATCH_SIZE;
    }
    
    int lastStreamIdx = (numValues - lastBatchSize) % NUM_STREAMS;
    
    // 等待最后一批完成
    CUDA_CHECK(cudaEventSynchronize(stopEvents[lastStreamIdx]));
    
    // 处理最后一批结果
    #pragma omp parallel for
    for (int i = 0; i < lastBatchSize; i++) {
        int idx = numValues - lastBatchSize + i;
        guesses[originalSize + idx] = std::string(g_doubleBuffer.h_outputBuffer[1 - currentBuffer] + (i * MAX_STRING_LENGTH));
    }
    
    // 释放设备内存
    if (!useConstantMemory && d_prefix != NULL) {
        CUDA_CHECK(cudaFree(d_prefix));
    }
    
    total_guesses += numValues;
}

// 确保在程序开始时初始化GPU资源，结束时释放
class GPUResourceInitializer {
public:
    GPUResourceInitializer() {
        initGPUResources();
    }
    
    ~GPUResourceInitializer() {
        freeGPUResources();
    }
};

// 全局实例，确保资源自动初始化和释放
static GPUResourceInitializer g_resourceInitializer;