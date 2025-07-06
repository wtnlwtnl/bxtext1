#include "md5.h"
#include <iomanip>
#include <assert.h>
#include <chrono>
#include <stdlib.h>
#include <algorithm>
using namespace std;
using namespace chrono;

// 预分配的缓冲区，避免频繁动态分配内存
#define MAX_BUFFER_SIZE (8*1024*1024)  // 8MB缓冲区
const size_t BATCH_SIZE = 8;  // 增加并行度，改为const size_t类型解决类型不匹配问题
alignas(32) static Byte g_static_buffer[MAX_BUFFER_SIZE]; // 提高对齐到32字节
static size_t g_buffer_offset = 0;

// 从静态缓冲区分配内存，确保32字节对齐
inline __attribute__((always_inline)) Byte* allocate_from_static(size_t size) {
    // 对齐到32字节边界，提高NEON访问效率
    g_buffer_offset = (g_buffer_offset + 31) & ~31;
    
    // 检查缓冲区是否足够
    if (g_buffer_offset + size > MAX_BUFFER_SIZE) {
        g_buffer_offset = 0; // 简单重置而不是复用
    }
    
    // 分配内存
    Byte* result = &g_static_buffer[g_buffer_offset];
    g_buffer_offset += size;
    return result;
}

// 重置静态缓冲区
inline __attribute__((always_inline)) void reset_static_buffer() {
    g_buffer_offset = 0;
}

// 使用内联汇编实现32位整数的循环左移
inline __attribute__((always_inline)) uint32_t asm_rotl32(uint32_t x, int n) {
    uint32_t result;
    __asm__ __volatile__ (
        "mov %w0, %w1, ror %2"
        : "=r" (result)
        : "r" (x), "I" (32-n)
    );
    return result;
}

/**
 * StringProcess: 将单个输入字符串转换成MD5计算所需的消息数组
 */
Byte *StringProcess(string input, int *n_byte) {
    const Byte *blocks = (const Byte *)input.c_str();
    int length = input.length();
    int bitLength = length * 8;
    
    // 计算填充
    int paddingBits = bitLength % 512;
    if (paddingBits > 448) {
        paddingBits = 512 - (paddingBits - 448);
    } else if (paddingBits < 448) {
        paddingBits = 448 - paddingBits;
    } else if (paddingBits == 448) {
        paddingBits = 512;
    }
    
    int paddingBytes = paddingBits / 8;
    int paddedLength = length + paddingBytes + 8;
    Byte *paddedMessage = new Byte[paddedLength];
    
    // 复制原始消息和填充
    memcpy(paddedMessage, blocks, length);
    paddedMessage[length] = 0x80;
    memset(paddedMessage + length + 1, 0, paddingBytes - 1);
    
    // 添加消息长度（64比特，小端格式）
    uint64_t bitLen = static_cast<uint64_t>(length) * 8;
    for (int i = 0; i < 8; ++i) {
        paddedMessage[length + paddingBytes + i] = (bitLen >> (i * 8)) & 0xFF;
    }
    
    *n_byte = paddedLength;
    return paddedMessage;
}

/**
 * MD5Hash: 标准MD5哈希实现
 */
void MD5Hash(string input, bit32 *state) {
    Byte *paddedMessage;
    int messageLength;
    paddedMessage = StringProcess(input, &messageLength);
    int n_blocks = messageLength / 64;
    
    state[0] = 0x67452301;
    state[1] = 0xefcdab89;
    state[2] = 0x98badcfe;
    state[3] = 0x10325476;
    
    // 逐block地更新state
    for (int i = 0; i < n_blocks; i++) {
        bit32 x[16];
        
        // 处理每个块
        for (int j = 0; j < 16; ++j) {
            x[j] = (paddedMessage[4*j + i*64]) |
                   (paddedMessage[4*j + 1 + i*64] << 8) |
                   (paddedMessage[4*j + 2 + i*64] << 16) |
                   (paddedMessage[4*j + 3 + i*64] << 24);
        }
        
        bit32 a = state[0], b = state[1], c = state[2], d = state[3];
        
        /* Round 1 */
        FF(a, b, c, d, x[0], s11, 0xd76aa478);
        FF(d, a, b, c, x[1], s12, 0xe8c7b756);
        FF(c, d, a, b, x[2], s13, 0x242070db);
        FF(b, c, d, a, x[3], s14, 0xc1bdceee);
        FF(a, b, c, d, x[4], s11, 0xf57c0faf);
        FF(d, a, b, c, x[5], s12, 0x4787c62a);
        FF(c, d, a, b, x[6], s13, 0xa8304613);
        FF(b, c, d, a, x[7], s14, 0xfd469501);
        FF(a, b, c, d, x[8], s11, 0x698098d8);
        FF(d, a, b, c, x[9], s12, 0x8b44f7af);
        FF(c, d, a, b, x[10], s13, 0xffff5bb1);
        FF(b, c, d, a, x[11], s14, 0x895cd7be);
        FF(a, b, c, d, x[12], s11, 0x6b901122);
        FF(d, a, b, c, x[13], s12, 0xfd987193);
        FF(c, d, a, b, x[14], s13, 0xa679438e);
        FF(b, c, d, a, x[15], s14, 0x49b40821);
        
        /* Round 2 */
        GG(a, b, c, d, x[1], s21, 0xf61e2562);
        GG(d, a, b, c, x[6], s22, 0xc040b340);
        GG(c, d, a, b, x[11], s23, 0x265e5a51);
        GG(b, c, d, a, x[0], s24, 0xe9b6c7aa);
        GG(a, b, c, d, x[5], s21, 0xd62f105d);
        GG(d, a, b, c, x[10], s22, 0x2441453);
        GG(c, d, a, b, x[15], s23, 0xd8a1e681);
        GG(b, c, d, a, x[4], s24, 0xe7d3fbc8);
        GG(a, b, c, d, x[9], s21, 0x21e1cde6);
        GG(d, a, b, c, x[14], s22, 0xc33707d6);
        GG(c, d, a, b, x[3], s23, 0xf4d50d87);
        GG(b, c, d, a, x[8], s24, 0x455a14ed);
        GG(a, b, c, d, x[13], s21, 0xa9e3e905);
        GG(d, a, b, c, x[2], s22, 0xfcefa3f8);
        GG(c, d, a, b, x[7], s23, 0x676f02d9);
        GG(b, c, d, a, x[12], s24, 0x8d2a4c8a);
        
        /* Round 3 */
        HH(a, b, c, d, x[5], s31, 0xfffa3942);
        HH(d, a, b, c, x[8], s32, 0x8771f681);
        HH(c, d, a, b, x[11], s33, 0x6d9d6122);
        HH(b, c, d, a, x[14], s34, 0xfde5380c);
        HH(a, b, c, d, x[1], s31, 0xa4beea44);
        HH(d, a, b, c, x[4], s32, 0x4bdecfa9);
        HH(c, d, a, b, x[7], s33, 0xf6bb4b60);
        HH(b, c, d, a, x[10], s34, 0xbebfbc70);
        HH(a, b, c, d, x[13], s31, 0x289b7ec6);
        HH(d, a, b, c, x[0], s32, 0xeaa127fa);
        HH(c, d, a, b, x[3], s33, 0xd4ef3085);
        HH(b, c, d, a, x[6], s34, 0x4881d05);
        HH(a, b, c, d, x[9], s31, 0xd9d4d039);
        HH(d, a, b, c, x[12], s32, 0xe6db99e5);
        HH(c, d, a, b, x[15], s33, 0x1fa27cf8);
        HH(b, c, d, a, x[2], s34, 0xc4ac5665);
        
        /* Round 4 */
        II(a, b, c, d, x[0], s41, 0xf4292244);
        II(d, a, b, c, x[7], s42, 0x432aff97);
        II(c, d, a, b, x[14], s43, 0xab9423a7);
        II(b, c, d, a, x[5], s44, 0xfc93a039);
        II(a, b, c, d, x[12], s41, 0x655b59c3);
        II(d, a, b, c, x[3], s42, 0x8f0ccc92);
        II(c, d, a, b, x[10], s43, 0xffeff47d);
        II(b, c, d, a, x[1], s44, 0x85845dd1);
        II(a, b, c, d, x[8], s41, 0x6fa87e4f);
        II(d, a, b, c, x[15], s42, 0xfe2ce6e0);
        II(c, d, a, b, x[6], s43, 0xa3014314);
        II(b, c, d, a, x[13], s44, 0x4e0811a1);
        II(a, b, c, d, x[4], s41, 0xf7537e82);
        II(d, a, b, c, x[11], s42, 0xbd3af235);
        II(c, d, a, b, x[2], s43, 0x2ad7d2bb);
        II(b, c, d, a, x[9], s44, 0xeb86d391);
        
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
    }
    
    // 字节序转换
    for (int i = 0; i < 4; i++) {
        uint32_t value = state[i];
        state[i] = __builtin_bswap32(value); // 使用内建函数
    }
    
    delete[] paddedMessage;
}

// 辅助函数 - 从小端字节序加载 32 位整数
inline __attribute__((always_inline)) uint32_t load_le_u32(const Byte* ptr) {
    // 直接读取可能未对齐的指针，使用更有效的方法
    uint32_t value;
    memcpy(&value, ptr, sizeof(uint32_t));
    return value;
}

/**
 * StringProcess_NEON: 优化版本 - 预处理多个字符串
 */
void StringProcess_NEON(const vector<string>& inputs, vector<Byte*>& paddedMessages, vector<int>& messageLengths) {
    reset_static_buffer();
    
    size_t batchSize = inputs.size();
    paddedMessages.resize(batchSize);
    messageLengths.resize(batchSize);
    
    // 使用静态缓冲区分配和处理字符串
    for (size_t i = 0; i < batchSize; i++) {
        const string& input = inputs[i];
        int length = input.length();
        
        // 计算填充后的消息长度
        int bitLength = length * 8;
        int paddingBits = bitLength % 512;
        if (paddingBits > 448) {
            paddingBits = 512 - (paddingBits - 448);
        } else if (paddingBits < 448) {
            paddingBits = 448 - paddingBits;
        } else if (paddingBits == 448) {
            paddingBits = 512;
        }
        
        int paddingBytes = paddingBits / 8;
        int paddedLength = length + paddingBytes + 8;
        
        // 从静态缓冲区分配内存
        Byte* paddedMessage = allocate_from_static(paddedLength);
        const Byte* blocks = reinterpret_cast<const Byte*>(input.c_str());
        
        // 添加预取指令
        for (int j = 0; j < length; j += 64) {
            __builtin_prefetch(blocks + j, 0, 3);
        }
        
        // 内存复制
        memcpy(paddedMessage, blocks, length);
        
        // 添加填充字节 - 第一个字节是0x80
        paddedMessage[length] = 0x80;
        memset(paddedMessage + length + 1, 0, paddingBytes - 1);
        
        // 添加消息长度（64比特，小端格式）
        uint64_t bitLen = static_cast<uint64_t>(length) * 8;
        for (int j = 0; j < 8; ++j) {
            paddedMessage[length + paddingBytes + j] = (bitLen >> (j * 8)) & 0xFF;
        }
        
        paddedMessages[i] = paddedMessage;
        messageLengths[i] = paddedLength;
    }
}

/**
 * MD5Hash_NEON: 完全优化版本的MD5哈希
 */
void MD5Hash_NEON(const vector<string>& inputs, vector<bit32*>& states) {
    const size_t batchSize = inputs.size();
    if (batchSize == 0) return;
    
    // 优化: 对小批量输入使用串行版本
    if (batchSize <= 2) {
        states.resize(batchSize);
        for (size_t i = 0; i < batchSize; i++) {
            states[i] = new bit32[4];
            MD5Hash(inputs[i], states[i]);
        }
        return;
    }
    
    // 准备消息数组
    vector<Byte*> paddedMessages;
    vector<int> messageLengths;
    StringProcess_NEON(inputs, paddedMessages, messageLengths);
    
    // 预分配结果空间
    states.resize(batchSize);
    for (size_t i = 0; i < batchSize; i++) {
        states[i] = new bit32[4];
    }
    
    // 处理所有字符串，以BATCH_SIZE为一组
    for (size_t batch = 0; batch < batchSize; batch += BATCH_SIZE) {
        // 确定当前批次大小
        size_t currentBatchSize = std::min(BATCH_SIZE, batchSize - batch);
        
        // 对齐初始化数据
        alignas(32) uint32_t a_values[BATCH_SIZE] = {0};
        alignas(32) uint32_t b_values[BATCH_SIZE] = {0};
        alignas(32) uint32_t c_values[BATCH_SIZE] = {0};
        alignas(32) uint32_t d_values[BATCH_SIZE] = {0};
        
        // 初始化所有需要处理的状态
        for (size_t i = 0; i < currentBatchSize; i++) {
            a_values[i] = 0x67452301;
            b_values[i] = 0xefcdab89;
            c_values[i] = 0x98badcfe;
            d_values[i] = 0x10325476;
        }
        
        // 找出这一批中最大块数
        int maxBlocks = 0;
        for (size_t i = 0; i < currentBatchSize; i++) {
            maxBlocks = std::max(maxBlocks, messageLengths[batch + i] / 64);
        }
        
        // 根据当前批次大小选择处理方法
        if (currentBatchSize <= 4) {
            // 标准NEON处理，一次处理4个字符串
            neon_reg a = vld1q_u32(a_values);
            neon_reg b = vld1q_u32(b_values);
            neon_reg c = vld1q_u32(c_values);
            neon_reg d = vld1q_u32(d_values);
            
            // 处理每个块
            for (int blockIdx = 0; blockIdx < maxBlocks; blockIdx++) {
                neon_reg x[16];
                uint32_t data_array[16][4] = {{0}};
                
                // 高效加载数据
                for (size_t i = 0; i < currentBatchSize; i++) {
                    if (blockIdx < messageLengths[batch + i] / 64) {
                        for (int j = 0; j < 16; j++) {
                            const Byte* ptr = &paddedMessages[batch + i][4 * j + blockIdx * 64];
                            data_array[j][i] = load_le_u32(ptr);
                        }
                    }
                }
                
                // 加载数据到NEON寄存器
                for (int j = 0; j < 16; j++) {
                    x[j] = vld1q_u32(data_array[j]);
                }
                
                neon_reg save_a = a, save_b = b, save_c = c, save_d = d;
                
                // 标准MD5循环，使用NEON指令
                /* Round 1 */
                FF_NEON(a, b, c, d, x[0], s11, 0xd76aa478);
                FF_NEON(d, a, b, c, x[1], s12, 0xe8c7b756);
                FF_NEON(c, d, a, b, x[2], s13, 0x242070db);
                FF_NEON(b, c, d, a, x[3], s14, 0xc1bdceee);
                FF_NEON(a, b, c, d, x[4], s11, 0xf57c0faf);
                FF_NEON(d, a, b, c, x[5], s12, 0x4787c62a);
                FF_NEON(c, d, a, b, x[6], s13, 0xa8304613);
                FF_NEON(b, c, d, a, x[7], s14, 0xfd469501);
                FF_NEON(a, b, c, d, x[8], s11, 0x698098d8);
                FF_NEON(d, a, b, c, x[9], s12, 0x8b44f7af);
                FF_NEON(c, d, a, b, x[10], s13, 0xffff5bb1);
                FF_NEON(b, c, d, a, x[11], s14, 0x895cd7be);
                FF_NEON(a, b, c, d, x[12], s11, 0x6b901122);
                FF_NEON(d, a, b, c, x[13], s12, 0xfd987193);
                FF_NEON(c, d, a, b, x[14], s13, 0xa679438e);
                FF_NEON(b, c, d, a, x[15], s14, 0x49b40821);
                
                /* Round 2 */
                GG_NEON(a, b, c, d, x[1], s21, 0xf61e2562);
                GG_NEON(d, a, b, c, x[6], s22, 0xc040b340);
                GG_NEON(c, d, a, b, x[11], s23, 0x265e5a51);
                GG_NEON(b, c, d, a, x[0], s24, 0xe9b6c7aa);
                GG_NEON(a, b, c, d, x[5], s21, 0xd62f105d);
                GG_NEON(d, a, b, c, x[10], s22, 0x2441453);
                GG_NEON(c, d, a, b, x[15], s23, 0xd8a1e681);
                GG_NEON(b, c, d, a, x[4], s24, 0xe7d3fbc8);
                GG_NEON(a, b, c, d, x[9], s21, 0x21e1cde6);
                GG_NEON(d, a, b, c, x[14], s22, 0xc33707d6);
                GG_NEON(c, d, a, b, x[3], s23, 0xf4d50d87);
                GG_NEON(b, c, d, a, x[8], s24, 0x455a14ed);
                GG_NEON(a, b, c, d, x[13], s21, 0xa9e3e905);
                GG_NEON(d, a, b, c, x[2], s22, 0xfcefa3f8);
                GG_NEON(c, d, a, b, x[7], s23, 0x676f02d9);
                GG_NEON(b, c, d, a, x[12], s24, 0x8d2a4c8a);
                
                /* Round 3 */
                HH_NEON(a, b, c, d, x[5], s31, 0xfffa3942);
                HH_NEON(d, a, b, c, x[8], s32, 0x8771f681);
                HH_NEON(c, d, a, b, x[11], s33, 0x6d9d6122);
                HH_NEON(b, c, d, a, x[14], s34, 0xfde5380c);
                HH_NEON(a, b, c, d, x[1], s31, 0xa4beea44);
                HH_NEON(d, a, b, c, x[4], s32, 0x4bdecfa9);
                HH_NEON(c, d, a, b, x[7], s33, 0xf6bb4b60);
                HH_NEON(b, c, d, a, x[10], s34, 0xbebfbc70);
                HH_NEON(a, b, c, d, x[13], s31, 0x289b7ec6);
                HH_NEON(d, a, b, c, x[0], s32, 0xeaa127fa);
                HH_NEON(c, d, a, b, x[3], s33, 0xd4ef3085);
                HH_NEON(b, c, d, a, x[6], s34, 0x4881d05);
                HH_NEON(a, b, c, d, x[9], s31, 0xd9d4d039);
                HH_NEON(d, a, b, c, x[12], s32, 0xe6db99e5);
                HH_NEON(c, d, a, b, x[15], s33, 0x1fa27cf8);
                HH_NEON(b, c, d, a, x[2], s34, 0xc4ac5665);
                
                /* Round 4 */
                II_NEON(a, b, c, d, x[0], s41, 0xf4292244);
                II_NEON(d, a, b, c, x[7], s42, 0x432aff97);
                II_NEON(c, d, a, b, x[14], s43, 0xab9423a7);
                II_NEON(b, c, d, a, x[5], s44, 0xfc93a039);
                II_NEON(a, b, c, d, x[12], s41, 0x655b59c3);
                II_NEON(d, a, b, c, x[3], s42, 0x8f0ccc92);
                II_NEON(c, d, a, b, x[10], s43, 0xffeff47d);
                II_NEON(b, c, d, a, x[1], s44, 0x85845dd1);
                II_NEON(a, b, c, d, x[8], s41, 0x6fa87e4f);
                II_NEON(d, a, b, c, x[15], s42, 0xfe2ce6e0);
                II_NEON(c, d, a, b, x[6], s43, 0xa3014314);
                II_NEON(b, c, d, a, x[13], s44, 0x4e0811a1);
                II_NEON(a, b, c, d, x[4], s41, 0xf7537e82);
                II_NEON(d, a, b, c, x[11], s42, 0xbd3af235);
                II_NEON(c, d, a, b, x[2], s43, 0x2ad7d2bb);
                II_NEON(b, c, d, a, x[9], s44, 0xeb86d391);
                
                // 更新状态
                a = vaddq_u32(a, save_a);
                b = vaddq_u32(b, save_b);
                c = vaddq_u32(c, save_c);
                d = vaddq_u32(d, save_d);
            }
            
            // 保存结果
            uint32_t a_result[4], b_result[4], c_result[4], d_result[4];
            vst1q_u32(a_result, a);
            vst1q_u32(b_result, b);
            vst1q_u32(c_result, c);
            vst1q_u32(d_result, d);
            
            // 字节序转换并保存结果
            for (size_t i = 0; i < currentBatchSize; i++) {
                states[batch + i][0] = __builtin_bswap32(a_result[i]);
                states[batch + i][1] = __builtin_bswap32(b_result[i]);
                states[batch + i][2] = __builtin_bswap32(c_result[i]);
                states[batch + i][3] = __builtin_bswap32(d_result[i]);
            }
        }
        else {
            // 增加并行度，分成两组处理
            neon_reg a1 = vld1q_u32(a_values);
            neon_reg b1 = vld1q_u32(b_values);
            neon_reg c1 = vld1q_u32(c_values);
            neon_reg d1 = vld1q_u32(d_values);
            
            neon_reg a2 = vld1q_u32(a_values + 4);
            neon_reg b2 = vld1q_u32(b_values + 4);
            neon_reg c2 = vld1q_u32(c_values + 4);
            neon_reg d2 = vld1q_u32(d_values + 4);
            
            // 处理每个块
            for (int blockIdx = 0; blockIdx < maxBlocks; blockIdx++) {
                neon_reg x1[16], x2[16];
                alignas(32) uint32_t data_array1[16][4] = {{0}};
                alignas(32) uint32_t data_array2[16][4] = {{0}};
                
                // 分两组加载数据
                for (size_t i = 0; i < 4; i++) {
                    if (i < currentBatchSize && blockIdx < messageLengths[batch + i] / 64) {
                        for (int j = 0; j < 16; j++) {
                            const Byte* ptr = &paddedMessages[batch + i][4 * j + blockIdx * 64];
                            data_array1[j][i] = load_le_u32(ptr);
                        }
                    }
                }
                
                for (size_t i = 4; i < 8; i++) {
                    if (i < currentBatchSize && blockIdx < messageLengths[batch + i] / 64) {
                        for (int j = 0; j < 16; j++) {
                            const Byte* ptr = &paddedMessages[batch + i][4 * j + blockIdx * 64];
                            data_array2[j][i-4] = load_le_u32(ptr);
                        }
                    }
                }
                
                // 加载数据到NEON寄存器
                for (int j = 0; j < 16; j++) {
                    x1[j] = vld1q_u32(data_array1[j]);
                    x2[j] = vld1q_u32(data_array2[j]);
                }
                
                // 保存原始状态
                neon_reg save_a1 = a1, save_b1 = b1, save_c1 = c1, save_d1 = d1;
                neon_reg save_a2 = a2, save_b2 = b2, save_c2 = c2, save_d2 = d2;
                
                // 第一组MD5计算 - 完整实现
                /* Round 1 - 组1 */
                FF_NEON(a1, b1, c1, d1, x1[0], s11, 0xd76aa478);
                FF_NEON(d1, a1, b1, c1, x1[1], s12, 0xe8c7b756);
                FF_NEON(c1, d1, a1, b1, x1[2], s13, 0x242070db);
                FF_NEON(b1, c1, d1, a1, x1[3], s14, 0xc1bdceee);
                FF_NEON(a1, b1, c1, d1, x1[4], s11, 0xf57c0faf);
                FF_NEON(d1, a1, b1, c1, x1[5], s12, 0x4787c62a);
                FF_NEON(c1, d1, a1, b1, x1[6], s13, 0xa8304613);
                FF_NEON(b1, c1, d1, a1, x1[7], s14, 0xfd469501);
                FF_NEON(a1, b1, c1, d1, x1[8], s11, 0x698098d8);
                FF_NEON(d1, a1, b1, c1, x1[9], s12, 0x8b44f7af);
                FF_NEON(c1, d1, a1, b1, x1[10], s13, 0xffff5bb1);
                FF_NEON(b1, c1, d1, a1, x1[11], s14, 0x895cd7be);
                FF_NEON(a1, b1, c1, d1, x1[12], s11, 0x6b901122);
                FF_NEON(d1, a1, b1, c1, x1[13], s12, 0xfd987193);
                FF_NEON(c1, d1, a1, b1, x1[14], s13, 0xa679438e);
                FF_NEON(b1, c1, d1, a1, x1[15], s14, 0x49b40821);
                
                /* Round 2 - 组1 */
                GG_NEON(a1, b1, c1, d1, x1[1], s21, 0xf61e2562);
                GG_NEON(d1, a1, b1, c1, x1[6], s22, 0xc040b340);
                GG_NEON(c1, d1, a1, b1, x1[11], s23, 0x265e5a51);
                GG_NEON(b1, c1, d1, a1, x1[0], s24, 0xe9b6c7aa);
                GG_NEON(a1, b1, c1, d1, x1[5], s21, 0xd62f105d);
                GG_NEON(d1, a1, b1, c1, x1[10], s22, 0x2441453);
                GG_NEON(c1, d1, a1, b1, x1[15], s23, 0xd8a1e681);
                GG_NEON(b1, c1, d1, a1, x1[4], s24, 0xe7d3fbc8);
                GG_NEON(a1, b1, c1, d1, x1[9], s21, 0x21e1cde6);
                GG_NEON(d1, a1, b1, c1, x1[14], s22, 0xc33707d6);
                GG_NEON(c1, d1, a1, b1, x1[3], s23, 0xf4d50d87);
                GG_NEON(b1, c1, d1, a1, x1[8], s24, 0x455a14ed);
                GG_NEON(a1, b1, c1, d1, x1[13], s21, 0xa9e3e905);
                GG_NEON(d1, a1, b1, c1, x1[2], s22, 0xfcefa3f8);
                GG_NEON(c1, d1, a1, b1, x1[7], s23, 0x676f02d9);
                GG_NEON(b1, c1, d1, a1, x1[12], s24, 0x8d2a4c8a);
                
                /* Round 3 - 组1 */
                HH_NEON(a1, b1, c1, d1, x1[5], s31, 0xfffa3942);
                HH_NEON(d1, a1, b1, c1, x1[8], s32, 0x8771f681);
                HH_NEON(c1, d1, a1, b1, x1[11], s33, 0x6d9d6122);
                HH_NEON(b1, c1, d1, a1, x1[14], s34, 0xfde5380c);
                HH_NEON(a1, b1, c1, d1, x1[1], s31, 0xa4beea44);
                HH_NEON(d1, a1, b1, c1, x1[4], s32, 0x4bdecfa9);
                HH_NEON(c1, d1, a1, b1, x1[7], s33, 0xf6bb4b60);
                HH_NEON(b1, c1, d1, a1, x1[10], s34, 0xbebfbc70);
                HH_NEON(a1, b1, c1, d1, x1[13], s31, 0x289b7ec6);
                HH_NEON(d1, a1, b1, c1, x1[0], s32, 0xeaa127fa);
                HH_NEON(c1, d1, a1, b1, x1[3], s33, 0xd4ef3085);
                HH_NEON(b1, c1, d1, a1, x1[6], s34, 0x4881d05);
                HH_NEON(a1, b1, c1, d1, x1[9], s31, 0xd9d4d039);
                HH_NEON(d1, a1, b1, c1, x1[12], s32, 0xe6db99e5);
                HH_NEON(c1, d1, a1, b1, x1[15], s33, 0x1fa27cf8);
                HH_NEON(b1, c1, d1, a1, x1[2], s34, 0xc4ac5665);
                
                /* Round 4 - 组1 */
                II_NEON(a1, b1, c1, d1, x1[0], s41, 0xf4292244);
                II_NEON(d1, a1, b1, c1, x1[7], s42, 0x432aff97);
                II_NEON(c1, d1, a1, b1, x1[14], s43, 0xab9423a7);
                II_NEON(b1, c1, d1, a1, x1[5], s44, 0xfc93a039);
                II_NEON(a1, b1, c1, d1, x1[12], s41, 0x655b59c3);
                II_NEON(d1, a1, b1, c1, x1[3], s42, 0x8f0ccc92);
                II_NEON(c1, d1, a1, b1, x1[10], s43, 0xffeff47d);
                II_NEON(b1, c1, d1, a1, x1[1], s44, 0x85845dd1);
                II_NEON(a1, b1, c1, d1, x1[8], s41, 0x6fa87e4f);
                II_NEON(d1, a1, b1, c1, x1[15], s42, 0xfe2ce6e0);
                II_NEON(c1, d1, a1, b1, x1[6], s43, 0xa3014314);
                II_NEON(b1, c1, d1, a1, x1[13], s44, 0x4e0811a1);
                II_NEON(a1, b1, c1, d1, x1[4], s41, 0xf7537e82);
                II_NEON(d1, a1, b1, c1, x1[11], s42, 0xbd3af235);
                II_NEON(c1, d1, a1, b1, x1[2], s43, 0x2ad7d2bb);
                II_NEON(b1, c1, d1, a1, x1[9], s44, 0xeb86d391);
                
                // 第二组同时计算 - 完整实现
                /* Round 1 - 组2 */
                FF_NEON(a2, b2, c2, d2, x2[0], s11, 0xd76aa478);
                FF_NEON(d2, a2, b2, c2, x2[1], s12, 0xe8c7b756);
                FF_NEON(c2, d2, a2, b2, x2[2], s13, 0x242070db);
                FF_NEON(b2, c2, d2, a2, x2[3], s14, 0xc1bdceee);
                FF_NEON(a2, b2, c2, d2, x2[4], s11, 0xf57c0faf);
                FF_NEON(d2, a2, b2, c2, x2[5], s12, 0x4787c62a);
                FF_NEON(c2, d2, a2, b2, x2[6], s13, 0xa8304613);
                FF_NEON(b2, c2, d2, a2, x2[7], s14, 0xfd469501);
                FF_NEON(a2, b2, c2, d2, x2[8], s11, 0x698098d8);
                FF_NEON(d2, a2, b2, c2, x2[9], s12, 0x8b44f7af);
                FF_NEON(c2, d2, a2, b2, x2[10], s13, 0xffff5bb1);
                FF_NEON(b2, c2, d2, a2, x2[11], s14, 0x895cd7be);
                FF_NEON(a2, b2, c2, d2, x2[12], s11, 0x6b901122);
                FF_NEON(d2, a2, b2, c2, x2[13], s12, 0xfd987193);
                FF_NEON(c2, d2, a2, b2, x2[14], s13, 0xa679438e);
                FF_NEON(b2, c2, d2, a2, x2[15], s14, 0x49b40821);
                
                /* Round 2 - 组2 */
                GG_NEON(a2, b2, c2, d2, x2[1], s21, 0xf61e2562);
                GG_NEON(d2, a2, b2, c2, x2[6], s22, 0xc040b340);
                GG_NEON(c2, d2, a2, b2, x2[11], s23, 0x265e5a51);
                GG_NEON(b2, c2, d2, a2, x2[0], s24, 0xe9b6c7aa);
                GG_NEON(a2, b2, c2, d2, x2[5], s21, 0xd62f105d);
                GG_NEON(d2, a2, b2, c2, x2[10], s22, 0x2441453);
                GG_NEON(c2, d2, a2, b2, x2[15], s23, 0xd8a1e681);
                GG_NEON(b2, c2, d2, a2, x2[4], s24, 0xe7d3fbc8);
                GG_NEON(a2, b2, c2, d2, x2[9], s21, 0x21e1cde6);
                GG_NEON(d2, a2, b2, c2, x2[14], s22, 0xc33707d6);
                GG_NEON(c2, d2, a2, b2, x2[3], s23, 0xf4d50d87);
                GG_NEON(b2, c2, d2, a2, x2[8], s24, 0x455a14ed);
                GG_NEON(a2, b2, c2, d2, x2[13], s21, 0xa9e3e905);
                GG_NEON(d2, a2, b2, c2, x2[2], s22, 0xfcefa3f8);
                GG_NEON(c2, d2, a2, b2, x2[7], s23, 0x676f02d9);
                GG_NEON(b2, c2, d2, a2, x2[12], s24, 0x8d2a4c8a);
                
                /* Round 3 - 组2 */
                HH_NEON(a2, b2, c2, d2, x2[5], s31, 0xfffa3942);
                HH_NEON(d2, a2, b2, c2, x2[8], s32, 0x8771f681);
                HH_NEON(c2, d2, a2, b2, x2[11], s33, 0x6d9d6122);
                HH_NEON(b2, c2, d2, a2, x2[14], s34, 0xfde5380c);
                HH_NEON(a2, b2, c2, d2, x2[1], s31, 0xa4beea44);
                HH_NEON(d2, a2, b2, c2, x2[4], s32, 0x4bdecfa9);
                HH_NEON(c2, d2, a2, b2, x2[7], s33, 0xf6bb4b60);
                HH_NEON(b2, c2, d2, a2, x2[10], s34, 0xbebfbc70);
                HH_NEON(a2, b2, c2, d2, x2[13], s31, 0x289b7ec6);
                HH_NEON(d2, a2, b2, c2, x2[0], s32, 0xeaa127fa);
                HH_NEON(c2, d2, a2, b2, x2[3], s33, 0xd4ef3085);
                HH_NEON(b2, c2, d2, a2, x2[6], s34, 0x4881d05);
                HH_NEON(a2, b2, c2, d2, x2[9], s31, 0xd9d4d039);
                HH_NEON(d2, a2, b2, c2, x2[12], s32, 0xe6db99e5);
                HH_NEON(c2, d2, a2, b2, x2[15], s33, 0x1fa27cf8);
                HH_NEON(b2, c2, d2, a2, x2[2], s34, 0xc4ac5665);
                
                /* Round 4 - 组2 */
                II_NEON(a2, b2, c2, d2, x2[0], s41, 0xf4292244);
                II_NEON(d2, a2, b2, c2, x2[7], s42, 0x432aff97);
                II_NEON(c2, d2, a2, b2, x2[14], s43, 0xab9423a7);
                II_NEON(b2, c2, d2, a2, x2[5], s44, 0xfc93a039);
                II_NEON(a2, b2, c2, d2, x2[12], s41, 0x655b59c3);
                II_NEON(d2, a2, b2, c2, x2[3], s42, 0x8f0ccc92);
                II_NEON(c2, d2, a2, b2, x2[10], s43, 0xffeff47d);
                II_NEON(b2, c2, d2, a2, x2[1], s44, 0x85845dd1);
                II_NEON(a2, b2, c2, d2, x2[8], s41, 0x6fa87e4f);
                II_NEON(d2, a2, b2, c2, x2[15], s42, 0xfe2ce6e0);
                II_NEON(c2, d2, a2, b2, x2[6], s43, 0xa3014314);
                II_NEON(b2, c2, d2, a2, x2[13], s44, 0x4e0811a1);
                II_NEON(a2, b2, c2, d2, x2[4], s41, 0xf7537e82);
                II_NEON(d2, a2, b2, c2, x2[11], s42, 0xbd3af235);
                II_NEON(c2, d2, a2, b2, x2[2], s43, 0x2ad7d2bb);
                II_NEON(b2, c2, d2, a2, x2[9], s44, 0xeb86d391);
                
                // 更新两组的状态
                a1 = vaddq_u32(a1, save_a1);
                b1 = vaddq_u32(b1, save_b1);
                c1 = vaddq_u32(c1, save_c1);
                d1 = vaddq_u32(d1, save_d1);
                
                a2 = vaddq_u32(a2, save_a2);
                b2 = vaddq_u32(b2, save_b2);
                c2 = vaddq_u32(c2, save_c2);
                d2 = vaddq_u32(d2, save_d2);
            }
            
            // 保存两组结果
            uint32_t a_result[8], b_result[8], c_result[8], d_result[8];
            vst1q_u32(a_result, a1);
            vst1q_u32(b_result, b1);
            vst1q_u32(c_result, c1);
            vst1q_u32(d_result, d1);
            
            vst1q_u32(a_result + 4, a2);
            vst1q_u32(b_result + 4, b2);
            vst1q_u32(c_result + 4, c2);
            vst1q_u32(d_result + 4, d2);
            
            // 字节序转换并保存所有结果
            for (size_t i = 0; i < currentBatchSize; i++) {
                states[batch + i][0] = __builtin_bswap32(a_result[i]);
                states[batch + i][1] = __builtin_bswap32(b_result[i]);
                states[batch + i][2] = __builtin_bswap32(c_result[i]);
                states[batch + i][3] = __builtin_bswap32(d_result[i]);
            }
        }
    }
}