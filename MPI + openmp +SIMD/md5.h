#ifndef MD5_H
#define MD5_H

#include <iostream>
#include <string>
#include <cstring>
#include <arm_neon.h>
#include <vector>

using namespace std;

// 定义了Byte，便于使用
typedef unsigned char Byte;
// 定义了32比特
typedef unsigned int bit32;
// 添加 NEON 寄存器类型定义
typedef uint32x4_t neon_reg;

// MD5的一系列参数。参数是固定的，其实你不需要看懂这些
#define s11 7
#define s12 12
#define s13 17
#define s14 22
#define s21 5
#define s22 9
#define s23 14
#define s24 20
#define s31 4
#define s32 11
#define s33 16
#define s34 23
#define s41 6
#define s42 10
#define s43 15
#define s44 21

/**
 * @Basic MD5 functions.
 *
 * @param there bit32.
 *
 * @return one bit32.
 */
// 定义了一系列MD5中的具体函数 - 串行版本
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

// 串行版本的宏
#define ROTATELEFT(num, n) (((num) << (n)) | ((num) >> (32-(n))))
#define FF(a, b, c, d, x, s, ac) { \
  (a) += F ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}
#define GG(a, b, c, d, x, s, ac) { \
  (a) += G ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}
#define HH(a, b, c, d, x, s, ac) { \
  (a) += H ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}
#define II(a, b, c, d, x, s, ac) { \
  (a) += I ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

// NEON 并行版本的函数
inline neon_reg F_NEON(neon_reg x, neon_reg y, neon_reg z) {
  neon_reg not_x = vmvnq_u32(x);           // ~x
  neon_reg x_and_y = vandq_u32(x, y);      // (x) & (y)
  neon_reg notx_and_z = vandq_u32(not_x, z); // (~x) & (z)
  return vorrq_u32(x_and_y, notx_and_z);   // ((x) & (y)) | ((~x) & (z))
}

inline neon_reg G_NEON(neon_reg x, neon_reg y, neon_reg z) {
  neon_reg not_z = vmvnq_u32(z);           // ~z
  neon_reg x_and_z = vandq_u32(x, z);      // (x) & (z)
  neon_reg y_and_notz = vandq_u32(y, not_z); // (y) & (~z)
  return vorrq_u32(x_and_z, y_and_notz);   // ((x) & (z)) | ((y) & (~z))
}

inline neon_reg H_NEON(neon_reg x, neon_reg y, neon_reg z) {
  return veorq_u32(veorq_u32(x, y), z);    // (x) ^ (y) ^ (z)
}

inline neon_reg I_NEON(neon_reg x, neon_reg y, neon_reg z) {
  neon_reg not_z = vmvnq_u32(z);           // ~z
  neon_reg x_or_notz = vorrq_u32(x, not_z); // (x) | (~z)
  return veorq_u32(y, x_or_notz);          // (y) ^ ((x) | (~z))
}

inline neon_reg ROTATELEFT_NEON(neon_reg x, int n) {
  return vorrq_u32(vshlq_n_u32(x, n), vshrq_n_u32(x, 32-n));
}

// NEON 并行版本的宏
#define FF_NEON(a, b, c, d, x, s, ac) { \
  a = vaddq_u32(a, vaddq_u32(vaddq_u32(F_NEON(b, c, d), x), vdupq_n_u32(ac))); \
  a = ROTATELEFT_NEON(a, s); \
  a = vaddq_u32(a, b); \
}

#define GG_NEON(a, b, c, d, x, s, ac) { \
  a = vaddq_u32(a, vaddq_u32(vaddq_u32(G_NEON(b, c, d), x), vdupq_n_u32(ac))); \
  a = ROTATELEFT_NEON(a, s); \
  a = vaddq_u32(a, b); \
}

#define HH_NEON(a, b, c, d, x, s, ac) { \
  a = vaddq_u32(a, vaddq_u32(vaddq_u32(H_NEON(b, c, d), x), vdupq_n_u32(ac))); \
  a = ROTATELEFT_NEON(a, s); \
  a = vaddq_u32(a, b); \
}

#define II_NEON(a, b, c, d, x, s, ac) { \
  a = vaddq_u32(a, vaddq_u32(vaddq_u32(I_NEON(b, c, d), x), vdupq_n_u32(ac))); \
  a = ROTATELEFT_NEON(a, s); \
  a = vaddq_u32(a, b); \
}

// 函数声明
Byte *StringProcess(string input, int *n_byte);
void MD5Hash(string input, bit32 *state);
void MD5Hash_NEON(const vector<string>& inputs, vector<bit32*>& states);

#endif // MD5_H