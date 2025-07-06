#include "md5.h"
#include <iomanip>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>

using namespace std;

// 辅助函数：将MD5哈希结果转换为十六进制字符串
string hashToString(bit32* state) {
    stringstream ss;
    for (int i = 0; i < 4; i++) {
        ss << std::setw(8) << std::setfill('0') << hex << state[i];
    }
    return ss.str();
}

// 验证MD5哈希函数正确性的主函数
int main(int argc, char* argv[])
{
    // 测试字符串数组
    vector<string> testInputs = {
        "0123456789012345678901234567890123456789012345678901234567890123",
        "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijkl",
        "abcdefGHIJKLmnopqrSTUVWXyzABCDEFghijklMNOPQRstuvwxYZABCDEFGHIJKL",
        "abcd1234EFGH5678ijkl9012MNOP3456qrst7890UVWX1234yzAB5678CDEF9012"
    };

    // 存储结果
    vector<string> serialResults;
    vector<string> parallelResults;
    vector<bit32*> serialStates;
    vector<bit32*> parallelStates;
    
    // 1. 计算串行MD5结果（无输出）
    for (const auto& input : testInputs) {
        bit32* state = new bit32[4];
        MD5Hash(input, state);
        
        string result = hashToString(state);
        serialResults.push_back(result);
        serialStates.push_back(state);
    }

    // 2. 计算并行MD5结果（无输出）
    for (size_t i = 0; i < testInputs.size(); i++) {
        parallelStates.push_back(new bit32[4]);
    }
    
    MD5Hash_NEON(testInputs, parallelStates);
    
    for (size_t i = 0; i < testInputs.size(); i++) {
        string result = hashToString(parallelStates[i]);
        parallelResults.push_back(result);
    }

    // 3. 只输出结果比较
    cout << "--------------------------------------------------------------------------------------------------------" << endl;
    cout << setw(35) << left << "测试编号" 
         << setw(35) << "串行MD5结果" 
         << setw(35) << "并行MD5结果" 
         << setw(10) << "是否一致" << endl;
    cout << "--------------------------------------------------------------------------------------------------------" << endl;
    
    bool allCorrect = true;
    for (size_t i = 0; i < serialResults.size(); i++) {
        bool match = serialResults[i] == parallelResults[i];
        cout << setw(10) << left << i+1
             << setw(35) << serialResults[i] 
             << setw(35) << parallelResults[i]
             << setw(10) << (match ? "✓" : "✗") << endl;
        
        if (!match) allCorrect = false;
    }
    cout << "--------------------------------------------------------------------------------------------------------" << endl;
    cout << "总体结果: " << (allCorrect ? "全部通过 ✓" : "存在不匹配 ✗") << endl;
    
    // 释放内存
    for (auto& state : serialStates) {
        delete[] state;
    }
    
    for (auto& state : parallelStates) {
        delete[] state;
    }
    
    return 0;
}

//g++ correctness.cpp md5.cpp -o correctness_test