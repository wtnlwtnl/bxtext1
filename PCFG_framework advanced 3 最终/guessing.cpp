#include "PCFG.h"
#include "cuda_generate.h"
#include <algorithm>
#include <chrono>
#include <omp.h>
using namespace std;

// 性能指标跟踪变量
double cpu_time_total = 0.001; // 避免除零
double gpu_time_total = 0.001; // 避免除零
size_t cpu_count = 0;
size_t gpu_count = 0;
double cpu_time_per_value = 0.0001; // 初始估计值
double gpu_time_per_value = 0.00001; // 初始估计值
double gpu_transfer_overhead = 0.0005; // GPU传输开销估计(秒)

// 更精确地计算PT的估计工作量
size_t estimatePTWorkload(const PT& pt) {
    size_t workload = 1;
    
    // 如果是单段PT
    if (pt.content.size() == 1) {
        // 基础工作量等于可能的值数量
        workload = pt.max_indices[0];
        
        // 根据段类型调整权重
        const segment& seg = pt.content[0];
        if (seg.type == 1) { // 字母段
            // 长字母段的组合数量会非常多
            if (seg.length > 8) {
                workload *= 5;  // 更高权重
            }
            else if (seg.length > 5) {
                workload *= 2;
            }
        }
    }
    // 多段PT，计算总组合数
    else {
        for (size_t i = 0; i < pt.max_indices.size(); i++) {
            // 限制最大值以防止整数溢出
            if (workload < 1000000) {
                workload *= pt.max_indices[i];
            }
        }
    }
    
    return workload;
}

// 决定PT处理设备，考虑动态性能指标
bool shouldProcessOnGPU(const PT& pt) {
    size_t workload = estimatePTWorkload(pt);
    
    // 预测CPU和GPU处理时间
    double predicted_cpu_time = workload * cpu_time_per_value;
    double predicted_gpu_time = workload * gpu_time_per_value + gpu_transfer_overhead;
    
    // 如果GPU预测时间明显更短，使用GPU
    if (predicted_gpu_time < predicted_cpu_time * 0.8) {
        return true;
    }
    
    // GPU队列过长时优先使用CPU
    if (gpu_count > cpu_count * 3) {
        return false;
    }
    
    // 特定类型PT的启发式规则
    // 1. 长字母段通常适合GPU处理
    if (pt.content.size() == 1 && pt.content[0].type == 1 && 
        pt.content[0].length > 7 && pt.max_indices[0] > 500) {
        return true;
    }
    
    // 2. 非常小的工作量不值得GPU处理
    if (workload < 1000) {
        return false;
    }
    
    // 3. 多段复杂PT适合GPU批量处理
    if (pt.content.size() > 2 && workload > 10000) {
        return true;
    }
    
    // 默认使用CPU - 适用于中小规模任务
    return false;
}

// 更新处理性能统计
void updatePerformanceStats(bool usedGPU, size_t workload, double processingTime) {
    if (usedGPU) {
        gpu_time_total += processingTime;
        gpu_count++;
        // 指数移动平均更新GPU处理速度估计
        gpu_time_per_value = gpu_time_per_value * 0.7 + (processingTime / workload) * 0.3;
    } else {
        cpu_time_total += processingTime;
        cpu_count++;
        // 指数移动平均更新CPU处理速度估计
        cpu_time_per_value = cpu_time_per_value * 0.7 + (processingTime / workload) * 0.3;
    }
    
    // 周期性调整GPU传输开销估计
    if (gpu_count > 0 && gpu_count % 10 == 0) {
        double avg_gpu_time = gpu_time_total / gpu_count;
        double avg_workload = gpu_count > 0 ? gpu_time_total / (gpu_time_per_value * gpu_count) : 1000;
        gpu_transfer_overhead = max(0.0001, avg_gpu_time - avg_workload * gpu_time_per_value);
    }
}

// 添加这里缺少的函数定义
void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;

    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

void PriorityQueue::init()
{
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
                cacheSegmentValues(seg.type, seg.length, m.letters[m.FindLetter(seg)].ordered_values);
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
                cacheSegmentValues(seg.type, seg.length, m.digits[m.FindDigit(seg)].ordered_values);
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
                cacheSegmentValues(seg.type, seg.length, m.symbols[m.FindSymbol(seg)].ordered_values);
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    
    std::cout << "Queue initialized with " << priority.size() << " PTs" << std::endl;
}

// CPU版本的Generate函数 - 在CPU上处理PT生成密码
void PriorityQueue::GenerateCPU(PT pt)
{
    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        int segmentType = pt.content[0].type;
        int segmentLength = pt.content[0].length;
        
        // 获取适当的segment列表
        vector<string>* segmentValues = nullptr;
        if (segmentType == 1) { // 字母
            segmentValues = &m.letters[m.FindLetter(pt.content[0])].ordered_values;
        } else if (segmentType == 2) { // 数字
            segmentValues = &m.digits[m.FindDigit(pt.content[0])].ordered_values;
        } else if (segmentType == 3) { // 符号
            segmentValues = &m.symbols[m.FindSymbol(pt.content[0])].ordered_values;
        }
        
        if (segmentValues) {
            size_t valueCount = std::min(segmentValues->size(), size_t(pt.max_indices[0]));
            size_t originalSize = guesses.size();
            guesses.resize(originalSize + valueCount);
            
            // 在CPU上生成所有猜测
            for (size_t i = 0; i < valueCount; i++) {
                guesses[originalSize + i] = (*segmentValues)[i];
            }
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 构建除了最后一个segment之外的前缀
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 获取最后一个segment的值
        int lastSegmentType = pt.content[pt.content.size() - 1].type;
        int lastSegmentLength = pt.content[pt.content.size() - 1].length;
        
        // 获取最后一个segment的所有可能值
        vector<string>* lastSegmentValues = nullptr;
        if (lastSegmentType == 1) { // 字母
            lastSegmentValues = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])].ordered_values;
        } else if (lastSegmentType == 2) { // 数字
            lastSegmentValues = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])].ordered_values;
        } else if (lastSegmentType == 3) { // 符号
            lastSegmentValues = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])].ordered_values;
        }
        
        if (lastSegmentValues) {
            size_t valueCount = std::min(lastSegmentValues->size(), size_t(pt.max_indices[pt.content.size() - 1]));
            size_t originalSize = guesses.size();
            guesses.resize(originalSize + valueCount);
            
            // 在CPU上生成所有猜测
            for (size_t i = 0; i < valueCount; i++) {
                guesses[originalSize + i] = guess + (*lastSegmentValues)[i];
            }
        }
    }
}

// 这个函数是PCFG并行化算法的主要载体
void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 获取segment类型和长度
        int segmentType = pt.content[0].type;
        int segmentLength = pt.content[0].length;
        int valueCount = pt.max_indices[0];
        
        // 分配存储空间
        size_t originalSize = guesses.size();
        guesses.resize(originalSize + valueCount);
        
        // 调用CUDA函数
        int generated = 0;
        generateSingleSegmentGPU(segmentType, segmentLength, valueCount, guesses, generated, originalSize);
        
        // 如果生成数量小于期望值，调整大小
        if (generated < valueCount) {
            guesses.resize(originalSize + generated);
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 构建除了最后一个segment之外的前缀
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 获取最后一个segment的类型和长度
        int lastSegmentType = pt.content[pt.content.size() - 1].type;
        int lastSegmentLength = pt.content[pt.content.size() - 1].length;
        int valueCount = pt.max_indices[pt.content.size() - 1];
        
        // 分配存储空间
        size_t originalSize = guesses.size();
        guesses.resize(originalSize + valueCount);
        
        // 调用CUDA函数
        int generated = 0;
        generateMultiSegmentGPU(guess, lastSegmentType, lastSegmentLength, 
                               valueCount, guesses, generated, originalSize);
        
        // 如果生成数量小于期望值，调整大小
        if (generated < valueCount) {
            guesses.resize(originalSize + generated);
        }
    }
}

// PT::NewPTs的实现
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }
}

// PopNext方法：出队一个PT
void PriorityQueue::PopNext()
{
    // 调用多PT版本，但只处理一个PT
    PopNextMultiple(1);
}

// 添加PopNextMultiple函数实现
void PriorityQueue::PopNextMultiple(int count)
{
    // 确保不超出队列中可用的PT数量
    count = min(count, (int)priority.size());
    if (count <= 0) return;
    
    // 记录处理前的猜测数和时间
    size_t startGuessCount = guesses.size();
    
    // 分析每个PT的特性，计算工作量
    vector<size_t> workloads(count);
    vector<bool> useGPU(count);
    
    for (int i = 0; i < count; i++) {
        workloads[i] = estimatePTWorkload(priority[i]);
        useGPU[i] = shouldProcessOnGPU(priority[i]);
    }
    
    // 创建PT索引并按工作量排序
    vector<int> indices(count);
    for (int i = 0; i < count; i++) indices[i] = i;
    
    // 按工作量降序排列，大工作量优先处理
    sort(indices.begin(), indices.end(), [&workloads](int a, int b) {
        return workloads[a] > workloads[b];
    });
    
    // 组织GPU批处理组，将相似工作量的PT分组处理
    vector<vector<int>> gpuBatches;
    vector<int> currentBatch;
    size_t batchWorkload = 0;
    
    // 最大批次工作量阈值 - 可动态调整
    const size_t MAX_BATCH_WORKLOAD = 1000000; 
    
    for (int idx : indices) {
        if (useGPU[idx]) {
            // 如果当前批次加上这个PT会超过阈值，开始新批次
            if (!currentBatch.empty() && batchWorkload + workloads[idx] > MAX_BATCH_WORKLOAD) {
                gpuBatches.push_back(currentBatch);
                currentBatch.clear();
                batchWorkload = 0;
            }
            
            currentBatch.push_back(idx);
            batchWorkload += workloads[idx];
        }
    }
    
    // 添加最后一个批次
    if (!currentBatch.empty()) {
        gpuBatches.push_back(currentBatch);
    }
    
    // 处理GPU批次
    for (const auto& batch : gpuBatches) {
        // 对每个GPU批次，按顺序处理
        for (int idx : batch) {
            auto startTime = std::chrono::high_resolution_clock::now();
            
            // GPU处理
            Generate(priority[idx]);
            
            auto endTime = std::chrono::high_resolution_clock::now();
            double processingTime = std::chrono::duration<double>(endTime - startTime).count();
            
            // 更新性能统计
            updatePerformanceStats(true, workloads[idx], processingTime);
        }
    }
    
    // 并行处理CPU任务
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < count; i++) {
        int idx = indices[i];
        if (!useGPU[idx]) {
            auto startTime = std::chrono::high_resolution_clock::now();
            
            // 创建本地队列并设置模型
            PriorityQueue localQueue;
            localQueue.m = this->m;
            
            // CPU处理
            localQueue.GenerateCPU(priority[idx]);
            
            auto endTime = std::chrono::high_resolution_clock::now();
            double processingTime = std::chrono::duration<double>(endTime - startTime).count();
            
            // 更新性能统计
            #pragma omp critical
            {
                updatePerformanceStats(false, workloads[idx], processingTime);
                // 合并结果
                guesses.insert(guesses.end(), localQueue.guesses.begin(), localQueue.guesses.end());
            }
        }
    }
    
    // 收集所有新生成的PT
    vector<PT> all_new_pts;
    for (int i = 0; i < count; i++) {
        vector<PT> new_pts = priority[i].NewPTs();
        all_new_pts.insert(all_new_pts.end(), new_pts.begin(), new_pts.end());
    }
    
    // 移除已处理的PT
    priority.erase(priority.begin(), priority.begin() + count);
    
    // 为所有新PT计算概率并插入回优先队列
    for (PT& pt : all_new_pts) {
        // 计算概率
        CalProb(pt);
        
        // 根据概率将PT插入到优先队列的适当位置
        auto pos = std::lower_bound(priority.begin(), priority.end(), pt, 
                                   [](const PT& a, const PT& b) { return a.prob > b.prob; });
        priority.insert(pos, pt);
    }
    
    // 更新总猜测数
    total_guesses += (guesses.size() - startGuessCount);
    
    // 每处理10批PT，输出性能统计信息
    static int batchCounter = 0;
    if (++batchCounter % 10 == 0) {
        double cpu_efficiency = cpu_count > 0 ? cpu_time_total / cpu_count : 0;
        double gpu_efficiency = gpu_count > 0 ? gpu_time_total / gpu_count : 0;
        
        cout << "性能统计: CPU处理 " << cpu_count << " PTs (平均 " 
             << cpu_efficiency*1000 << " ms/PT), GPU处理 " << gpu_count 
             << " PTs (平均 " << gpu_efficiency*1000 << " ms/PT)" << endl;
    }
}

// 添加关闭工作线程的函数
void shutdownWorkerThreads() {
    // 无需关闭任何线程
}