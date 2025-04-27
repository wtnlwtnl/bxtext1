#include "md5.h"
#include <iomanip>
#include <assert.h>
#include <chrono>
#include <vector>
#include <immintrin.h>

using namespace std;
using namespace chrono;

/**
 * 处理字符串，填充至64字节的倍数，并添加长度信息
 * 
 * @param s 输入字符串
 * @param messageLength 输出消息长度
 * @return 处理后的字节数组
 */
Byte* StringProcess(string s, int* messageLength) {
    // 计算需要的字节数
    unsigned int originalLength = s.length();
    unsigned int zeroPadding = (56 - (originalLength % 64)) % 64;
    unsigned int totalLength = originalLength + zeroPadding + 8;
    
    // 分配内存
    Byte* out = new Byte[totalLength];
    
    // 复制原始字符串
    for (unsigned int i = 0; i < originalLength; i++) {
        out[i] = s[i];
    }
    
    // 添加1位
    out[originalLength] = 0x80;
    
    // 添加0填充
    for (unsigned int i = originalLength + 1; i < originalLength + zeroPadding; i++) {
        out[i] = 0;
    }
    
    // 添加原始长度（以位为单位）
    unsigned long long bitLength = originalLength * 8;
    unsigned int offset = originalLength + zeroPadding;
    
    out[offset] = bitLength & 0xFF;
    out[offset + 1] = (bitLength >> 8) & 0xFF;
    out[offset + 2] = (bitLength >> 16) & 0xFF;
    out[offset + 3] = (bitLength >> 24) & 0xFF;
    out[offset + 4] = (bitLength >> 32) & 0xFF;
    out[offset + 5] = (bitLength >> 40) & 0xFF;
    out[offset + 6] = (bitLength >> 48) & 0xFF;
    out[offset + 7] = (bitLength >> 56) & 0xFF;
    
    *messageLength = totalLength;
    return out;
}

/**
 * 使用 SSE 指令集的并行 F 函数实现
 */
inline __m128i simd_F(__m128i x, __m128i y, __m128i z) {
    __m128i x_and_y = _mm_and_si128(x, y);
    __m128i not_x = _mm_xor_si128(x, _mm_set1_epi32(0xFFFFFFFF));
    __m128i not_x_and_z = _mm_and_si128(not_x, z);
    return _mm_or_si128(x_and_y, not_x_and_z);
}

/**
 * 使用 SSE 指令集的并行 G 函数实现
 */
inline __m128i simd_G(__m128i x, __m128i y, __m128i z) {
    __m128i x_and_z = _mm_and_si128(x, z);
    __m128i not_z = _mm_xor_si128(z, _mm_set1_epi32(0xFFFFFFFF));
    __m128i y_and_not_z = _mm_and_si128(y, not_z);
    return _mm_or_si128(x_and_z, y_and_not_z);
}

/**
 * 使用 SSE 指令集的并行 H 函数实现
 */
inline __m128i simd_H(__m128i x, __m128i y, __m128i z) {
    return _mm_xor_si128(_mm_xor_si128(x, y), z);
}

/**
 * 使用 SSE 指令集的并行 I 函数实现
 */
inline __m128i simd_I(__m128i x, __m128i y, __m128i z) {
    __m128i x_or_not_z = _mm_or_si128(x, _mm_xor_si128(z, _mm_set1_epi32(0xFFFFFFFF)));
    return _mm_xor_si128(y, x_or_not_z);
}

/**
 * 使用 SSE 指令集的并行左旋转操作
 */
inline __m128i simd_ROTATELEFT(__m128i val, int shift) {
    return _mm_or_si128(_mm_slli_epi32(val, shift), _mm_srli_epi32(val, 32 - shift));
}

/**
 * 并行版本的 FF 函数
 */
inline void simd_FF(__m128i &a, __m128i b, __m128i c, __m128i d, __m128i x, int s, uint32_t ac) {
    a = _mm_add_epi32(a, _mm_add_epi32(_mm_add_epi32(simd_F(b, c, d), x), _mm_set1_epi32(ac)));
    a = simd_ROTATELEFT(a, s);
    a = _mm_add_epi32(a, b);
}

/**
 * 并行版本的 GG 函数
 */
inline void simd_GG(__m128i &a, __m128i b, __m128i c, __m128i d, __m128i x, int s, uint32_t ac) {
    a = _mm_add_epi32(a, _mm_add_epi32(_mm_add_epi32(simd_G(b, c, d), x), _mm_set1_epi32(ac)));
    a = simd_ROTATELEFT(a, s);
    a = _mm_add_epi32(a, b);
}

/**
 * 并行版本的 HH 函数
 */
inline void simd_HH(__m128i &a, __m128i b, __m128i c, __m128i d, __m128i x, int s, uint32_t ac) {
    a = _mm_add_epi32(a, _mm_add_epi32(_mm_add_epi32(simd_H(b, c, d), x), _mm_set1_epi32(ac)));
    a = simd_ROTATELEFT(a, s);
    a = _mm_add_epi32(a, b);
}

/**
 * 并行版本的 II 函数
 */
inline void simd_II(__m128i &a, __m128i b, __m128i c, __m128i d, __m128i x, int s, uint32_t ac) {
    a = _mm_add_epi32(a, _mm_add_epi32(_mm_add_epi32(simd_I(b, c, d), x), _mm_set1_epi32(ac)));
    a = simd_ROTATELEFT(a, s);
    a = _mm_add_epi32(a, b);
}

/**
 * 并行处理多个输入的 MD5 哈希计算
 */
void MD5Hash_SIMD(string inputs[], int count, bit32 states[][4]) {
    // 处理输入字符串，准备消息数组
    vector<Byte*> paddedMessages;
    vector<int> messageLengths;
    
    // 每次处理4个输入
    for (int batch = 0; batch < count; batch += 4) {
        int batchSize = min(4, count - batch);
        
        // 为当前批次准备输入
        Byte* messages[4] = {nullptr, nullptr, nullptr, nullptr};
        int lengths[4] = {0, 0, 0, 0};
        
        // 处理每个输入
        for (int i = 0; i < batchSize; i++) {
            messages[i] = StringProcess(inputs[batch + i], &lengths[i]);
            paddedMessages.push_back(messages[i]);
            messageLengths.push_back(lengths[i]);
        }
        
        // 初始化状态
        __m128i state0 = _mm_set1_epi32(0x67452301);
        __m128i state1 = _mm_set1_epi32(0xefcdab89);
        __m128i state2 = _mm_set1_epi32(0x98badcfe);
        __m128i state3 = _mm_set1_epi32(0x10325476);
        
        // 确定最大块数
        int maxBlocks = 0;
        for (int i = 0; i < batchSize; i++) {
            int blocks = lengths[i] / 64;
            maxBlocks = max(maxBlocks, blocks);
        }
        
        // 对每个块进行处理
        for (int blockIdx = 0; blockIdx < maxBlocks; blockIdx++) {
            // 准备当前块的 x 数组
            __m128i x[16];
            for (int i = 0; i < 16; i++) {
                uint32_t values[4] = {0, 0, 0, 0};
                
                for (int j = 0; j < batchSize; j++) {
                    if (blockIdx < lengths[j] / 64) {
                        int offset = blockIdx * 64;
                        values[j] = (messages[j][4 * i + offset]) |
                                   (messages[j][4 * i + 1 + offset] << 8) |
                                   (messages[j][4 * i + 2 + offset] << 16) |
                                   (messages[j][4 * i + 3 + offset] << 24);
                    }
                }
                
                x[i] = _mm_set_epi32(values[3], values[2], values[1], values[0]);
            }
            
            // 保存当前状态
            __m128i a = state0;
            __m128i b = state1;
            __m128i c = state2;
            __m128i d = state3;
            
            // 执行 MD5 的四轮运算
            /* Round 1 */
            simd_FF(a, b, c, d, x[0], s11, 0xd76aa478);
            simd_FF(d, a, b, c, x[1], s12, 0xe8c7b756);
            simd_FF(c, d, a, b, x[2], s13, 0x242070db);
            simd_FF(b, c, d, a, x[3], s14, 0xc1bdceee);
            simd_FF(a, b, c, d, x[4], s11, 0xf57c0faf);
            simd_FF(d, a, b, c, x[5], s12, 0x4787c62a);
            simd_FF(c, d, a, b, x[6], s13, 0xa8304613);
            simd_FF(b, c, d, a, x[7], s14, 0xfd469501);
            simd_FF(a, b, c, d, x[8], s11, 0x698098d8);
            simd_FF(d, a, b, c, x[9], s12, 0x8b44f7af);
            simd_FF(c, d, a, b, x[10], s13, 0xffff5bb1);
            simd_FF(b, c, d, a, x[11], s14, 0x895cd7be);
            simd_FF(a, b, c, d, x[12], s11, 0x6b901122);
            simd_FF(d, a, b, c, x[13], s12, 0xfd987193);
            simd_FF(c, d, a, b, x[14], s13, 0xa679438e);
            simd_FF(b, c, d, a, x[15], s14, 0x49b40821);
            
            /* Round 2 */
            simd_GG(a, b, c, d, x[1], s21, 0xf61e2562);
            simd_GG(d, a, b, c, x[6], s22, 0xc040b340);
            simd_GG(c, d, a, b, x[11], s23, 0x265e5a51);
            simd_GG(b, c, d, a, x[0], s24, 0xe9b6c7aa);
            simd_GG(a, b, c, d, x[5], s21, 0xd62f105d);
            simd_GG(d, a, b, c, x[10], s22, 0x2441453);
            simd_GG(c, d, a, b, x[15], s23, 0xd8a1e681);
            simd_GG(b, c, d, a, x[4], s24, 0xe7d3fbc8);
            simd_GG(a, b, c, d, x[9], s21, 0x21e1cde6);
            simd_GG(d, a, b, c, x[14], s22, 0xc33707d6);
            simd_GG(c, d, a, b, x[3], s23, 0xf4d50d87);
            simd_GG(b, c, d, a, x[8], s24, 0x455a14ed);
            simd_GG(a, b, c, d, x[13], s21, 0xa9e3e905);
            simd_GG(d, a, b, c, x[2], s22, 0xfcefa3f8);
            simd_GG(c, d, a, b, x[7], s23, 0x676f02d9);
            simd_GG(b, c, d, a, x[12], s24, 0x8d2a4c8a);
            
            /* Round 3 */
            simd_HH(a, b, c, d, x[5], s31, 0xfffa3942);
            simd_HH(d, a, b, c, x[8], s32, 0x8771f681);
            simd_HH(c, d, a, b, x[11], s33, 0x6d9d6122);
            simd_HH(b, c, d, a, x[14], s34, 0xfde5380c);
            simd_HH(a, b, c, d, x[1], s31, 0xa4beea44);
            simd_HH(d, a, b, c, x[4], s32, 0x4bdecfa9);
            simd_HH(c, d, a, b, x[7], s33, 0xf6bb4b60);
            simd_HH(b, c, d, a, x[10], s34, 0xbebfbc70);
            simd_HH(a, b, c, d, x[13], s31, 0x289b7ec6);
            simd_HH(d, a, b, c, x[0], s32, 0xeaa127fa);
            simd_HH(c, d, a, b, x[3], s33, 0xd4ef3085);
            simd_HH(b, c, d, a, x[6], s34, 0x4881d05);
            simd_HH(a, b, c, d, x[9], s31, 0xd9d4d039);
            simd_HH(d, a, b, c, x[12], s32, 0xe6db99e5);
            simd_HH(c, d, a, b, x[15], s33, 0x1fa27cf8);
            simd_HH(b, c, d, a, x[2], s34, 0xc4ac5665);
            
            /* Round 4 */
            simd_II(a, b, c, d, x[0], s41, 0xf4292244);
            simd_II(d, a, b, c, x[7], s42, 0x432aff97);
            simd_II(c, d, a, b, x[14], s43, 0xab9423a7);
            simd_II(b, c, d, a, x[5], s44, 0xfc93a039);
            simd_II(a, b, c, d, x[12], s41, 0x655b59c3);
            simd_II(d, a, b, c, x[3], s42, 0x8f0ccc92);
            simd_II(c, d, a, b, x[10], s43, 0xffeff47d);
            simd_II(b, c, d, a, x[1], s44, 0x85845dd1);
            simd_II(a, b, c, d, x[8], s41, 0x6fa87e4f);
            simd_II(d, a, b, c, x[15], s42, 0xfe2ce6e0);
            simd_II(c, d, a, b, x[6], s43, 0xa3014314);
            simd_II(b, c, d, a, x[13], s44, 0x4e0811a1);
            simd_II(a, b, c, d, x[4], s41, 0xf7537e82);
            simd_II(d, a, b, c, x[11], s42, 0xbd3af235);
            simd_II(c, d, a, b, x[2], s43, 0x2ad7d2bb);
            simd_II(b, c, d, a, x[9], s44, 0xeb86d391);
            
            // 更新状态
            state0 = _mm_add_epi32(state0, a);
            state1 = _mm_add_epi32(state1, b);
            state2 = _mm_add_epi32(state2, c);
            state3 = _mm_add_epi32(state3, d);
        }
        
        // 将结果从 SIMD 寄存器转移到输出数组
        uint32_t state0_values[4], state1_values[4], state2_values[4], state3_values[4];
        _mm_storeu_si128((__m128i*)state0_values, state0);
        _mm_storeu_si128((__m128i*)state1_values, state1);
        _mm_storeu_si128((__m128i*)state2_values, state2);
        _mm_storeu_si128((__m128i*)state3_values, state3);
        
        // 保存结果到输出数组
        for (int i = 0; i < batchSize; i++) {
            states[batch + i][0] = state0_values[i];
            states[batch + i][1] = state1_values[i];
            states[batch + i][2] = state2_values[i];
            states[batch + i][3] = state3_values[i];
        }
    }
    
    // 释放动态分配的内存
    for (Byte* msg : paddedMessages) {
        delete[] msg;
    }
}