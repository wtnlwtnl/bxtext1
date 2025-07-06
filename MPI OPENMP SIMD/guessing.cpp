#include "PCFG.h"
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <algorithm>
using namespace std;

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是"纯粹的"PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中"123456"为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{
    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    //Generate(priority.front());
    GenerateHybrid(priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
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
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
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

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}

// 字符串序列化函数 - 将字符串向量转换为可通过MPI发送的形式
void serialize_strings(const vector<string>& strings, vector<char>& buffer, vector<int>& sizes) {
    sizes.resize(strings.size());
    int total_size = 0;
    
    // 计算总大小和每个字符串的大小
    for (size_t i = 0; i < strings.size(); i++) {
        sizes[i] = strings[i].size();
        total_size += sizes[i];
    }
    
    buffer.resize(total_size);
    
    // 复制所有字符串到缓冲区
    size_t pos = 0;
    for (const auto& str : strings) {
        copy(str.begin(), str.end(), buffer.begin() + pos);
        pos += str.size();
    }
}

// 字符串反序列化函数 - 从缓冲区重建字符串向量
vector<string> deserialize_strings(const vector<char>& buffer, const vector<int>& sizes) {
    vector<string> result;
    size_t pos = 0;
    
    for (int size : sizes) {
        result.push_back(string(buffer.begin() + pos, buffer.begin() + pos + size));
        pos += size;
    }
    
    return result;
}

// 混合并行版本的Generate函数 (MPI+OpenMP)
void PriorityQueue::GenerateHybrid(PT pt) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 计算PT的概率，这是在每个进程中都需要做的
    CalProb(pt);
    
    // 为每个MPI进程设置合适的OpenMP线程数
    // 一般建议每个处理器使用4-8个线程，具体视系统而定
    int num_threads = 4; // 每个MPI进程使用4个线程
    omp_set_num_threads(num_threads);
    
    if (pt.content.size() == 1) {
        // 处理只有一个segment的PT情况
        segment *a = nullptr;
        
        // 在每个进程上都定位segment
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)
            a = &m.digits[m.FindDigit(pt.content[0])];
        else
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        
        int total_items = pt.max_indices[0];
        
        // 计算每个MPI进程的工作量
        int items_per_proc = total_items / size;
        int remainder = total_items % size;
        
        // 确定本进程的工作范围
        int start_idx = rank * items_per_proc + min(rank, remainder);
        int end_idx = start_idx + items_per_proc + (rank < remainder ? 1 : 0);
        int local_items = end_idx - start_idx;
        
        // 本进程生成的猜测
        vector<string> local_guesses;
        local_guesses.resize(local_items); // 预分配空间提高效率
        
        // 使用OpenMP在进程内并行生成猜测
        #pragma omp parallel
        {
            // 计算每个线程的工作范围
            int thread_id = omp_get_thread_num();
            int thread_count = omp_get_num_threads();
            int items_per_thread = local_items / thread_count;
            int thread_remainder = local_items % thread_count;
            
            int thread_start = thread_id * items_per_thread + min(thread_id, thread_remainder);
            int thread_end = thread_start + items_per_thread + (thread_id < thread_remainder ? 1 : 0);
            
            // 每个线程处理自己的部分
            for (int i = thread_start; i < thread_end; i++) {
                local_guesses[i] = a->ordered_values[start_idx + i];
            }
        }
        
        // 确保所有线程都完成了工作
        #pragma omp barrier
        
        // 收集所有进程的结果
        int local_count = local_guesses.size();
        vector<int> all_counts(size);
        
        // 收集每个进程生成的猜测数量
        MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            // 只在主进程中更新total_guesses
            total_guesses = 0; // 重置计数
            for (int count : all_counts)
                total_guesses += count;
        }
        
        // 序列化本地猜测
        vector<char> send_buffer;
        vector<int> send_sizes;
        serialize_strings(local_guesses, send_buffer, send_sizes);
        
        // 收集各进程的序列化数据大小
        int local_buffer_size = send_buffer.size();
        vector<int> buffer_sizes(size);
        MPI_Gather(&local_buffer_size, 1, MPI_INT, buffer_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // 计算位移
        vector<int> displacements(size);
        if (rank == 0) {
            displacements[0] = 0;
            for (int i = 1; i < size; i++) {
                displacements[i] = displacements[i-1] + buffer_sizes[i-1];
            }
        }
        
        // 准备接收缓冲区 - 修改：只在rank 0上分配内存
        int total_buffer_size = 0;
        if (rank == 0) {
            for (int s : buffer_sizes)
                total_buffer_size += s;
        }
        vector<char> recv_buffer;
        if (rank == 0) {
            recv_buffer.resize(total_buffer_size);
        }
        
        // 收集所有序列化的猜测 - 修改：区分rank 0和其他进程
        if (rank == 0) {
            MPI_Gatherv(send_buffer.data(), send_buffer.size(), MPI_CHAR,
                      recv_buffer.data(), buffer_sizes.data(), displacements.data(),
                      MPI_CHAR, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(send_buffer.data(), send_buffer.size(), MPI_CHAR,
                      NULL, NULL, NULL,
                      MPI_CHAR, 0, MPI_COMM_WORLD);
        }
        
        // 收集所有字符串大小信息
        vector<int> all_sizes_flat;
        vector<int> sizes_counts(size);
        vector<int> sizes_displacements(size);
        
        if (rank == 0) {
            for (int i = 0; i < size; i++) {
                sizes_counts[i] = all_counts[i];
            }
            
            sizes_displacements[0] = 0;
            for (int i = 1; i < size; i++) {
                sizes_displacements[i] = sizes_displacements[i-1] + sizes_counts[i-1];
            }
            
            all_sizes_flat.resize(total_guesses);
        }
        
        // 修改：区分rank 0和其他进程
        if (rank == 0) {
            MPI_Gatherv(send_sizes.data(), send_sizes.size(), MPI_INT,
                      all_sizes_flat.data(), sizes_counts.data(), sizes_displacements.data(),
                      MPI_INT, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(send_sizes.data(), send_sizes.size(), MPI_INT,
                      NULL, NULL, NULL,
                      MPI_INT, 0, MPI_COMM_WORLD);
        }
        
        // 只在主进程中进行反序列化和结果合并
        if (rank == 0) {
            vector<string> all_guesses = deserialize_strings(recv_buffer, all_sizes_flat);
            guesses.insert(guesses.end(), all_guesses.begin(), all_guesses.end());
        }
    }
    else {
        // 处理有多个segment的PT情况
        // 构建除最后一个segment外的前缀
        string prefix;
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (seg_idx == (int)pt.content.size() - 1) break;

            const segment &segInfo = pt.content[seg_idx];
            if (segInfo.type == 1)
                prefix += m.letters[m.FindLetter(segInfo)].ordered_values[idx];
            else if (segInfo.type == 2)
                prefix += m.digits[m.FindDigit(segInfo)].ordered_values[idx];
            else
                prefix += m.symbols[m.FindSymbol(segInfo)].ordered_values[idx];

            ++seg_idx;
        }

        // 定位最后一个segment
        segment *a;
        const segment &lastSegInfo = pt.content.back();
        if (lastSegInfo.type == 1)
            a = &m.letters[m.FindLetter(lastSegInfo)];
        else if (lastSegInfo.type == 2)
            a = &m.digits[m.FindDigit(lastSegInfo)];
        else
            a = &m.symbols[m.FindSymbol(lastSegInfo)];
        
        int total_items = pt.max_indices.back();
        
        // 计算每个MPI进程的工作量
        int items_per_proc = total_items / size;
        int remainder = total_items % size;
        
        // 确定本进程的工作范围
        int start_idx = rank * items_per_proc + min(rank, remainder);
        int end_idx = start_idx + items_per_proc + (rank < remainder ? 1 : 0);
        int local_items = end_idx - start_idx;
        
        // 本进程生成的猜测
        vector<string> local_guesses;
        local_guesses.resize(local_items); // 预分配空间提高效率
        
        // 使用OpenMP在进程内并行生成猜测
        #pragma omp parallel
        {
            // 计算每个线程的工作范围
            int thread_id = omp_get_thread_num();
            int thread_count = omp_get_num_threads();
            int items_per_thread = local_items / thread_count;
            int thread_remainder = local_items % thread_count;
            
            int thread_start = thread_id * items_per_thread + min(thread_id, thread_remainder);
            int thread_end = thread_start + items_per_thread + (thread_id < thread_remainder ? 1 : 0);
            
            // 每个线程处理自己的部分
            for (int i = thread_start; i < thread_end; i++) {
                local_guesses[i] = prefix + a->ordered_values[start_idx + i];
            }
        }
        
        // 确保所有线程都完成了工作
        #pragma omp barrier
        
        // 收集所有进程的结果
        int local_count = local_guesses.size();
        vector<int> all_counts(size);
        
        // 收集每个进程生成的猜测数量
        MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            // 只在主进程中更新total_guesses
            total_guesses = 0; // 重置计数
            for (int count : all_counts)
                total_guesses += count;
        }
        
        // 序列化本地猜测
        vector<char> send_buffer;
        vector<int> send_sizes;
        serialize_strings(local_guesses, send_buffer, send_sizes);
        
        // 收集各进程的序列化数据大小
        int local_buffer_size = send_buffer.size();
        vector<int> buffer_sizes(size);
        MPI_Gather(&local_buffer_size, 1, MPI_INT, buffer_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // 计算位移
        vector<int> displacements(size);
        if (rank == 0) {
            displacements[0] = 0;
            for (int i = 1; i < size; i++) {
                displacements[i] = displacements[i-1] + buffer_sizes[i-1];
            }
        }
        
        // 准备接收缓冲区 - 修改：只在rank 0上分配内存
        int total_buffer_size = 0;
        if (rank == 0) {
            for (int s : buffer_sizes)
                total_buffer_size += s;
        }
        vector<char> recv_buffer;
        if (rank == 0) {
            recv_buffer.resize(total_buffer_size);
        }
        
        // 收集所有序列化的猜测 - 修改：区分rank 0和其他进程
        if (rank == 0) {
            MPI_Gatherv(send_buffer.data(), send_buffer.size(), MPI_CHAR,
                      recv_buffer.data(), buffer_sizes.data(), displacements.data(),
                      MPI_CHAR, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(send_buffer.data(), send_buffer.size(), MPI_CHAR,
                      NULL, NULL, NULL,
                      MPI_CHAR, 0, MPI_COMM_WORLD);
        }
        
        // 收集所有字符串大小信息
        vector<int> all_sizes_flat;
        vector<int> sizes_counts(size);
        vector<int> sizes_displacements(size);
        
        if (rank == 0) {
            for (int i = 0; i < size; i++) {
                sizes_counts[i] = all_counts[i];
            }
            
            sizes_displacements[0] = 0;
            for (int i = 1; i < size; i++) {
                sizes_displacements[i] = sizes_displacements[i-1] + sizes_counts[i-1];
            }
            
            all_sizes_flat.resize(total_guesses);
        }
        
        // 修改：区分rank 0和其他进程
        if (rank == 0) {
            MPI_Gatherv(send_sizes.data(), send_sizes.size(), MPI_INT,
                      all_sizes_flat.data(), sizes_counts.data(), sizes_displacements.data(),
                      MPI_INT, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(send_sizes.data(), send_sizes.size(), MPI_INT,
                      NULL, NULL, NULL,
                      MPI_INT, 0, MPI_COMM_WORLD);
        }
        
        // 只在主进程中进行反序列化和结果合并
        if (rank == 0) {
            vector<string> all_guesses = deserialize_strings(recv_buffer, all_sizes_flat);
            guesses.insert(guesses.end(), all_guesses.begin(), all_guesses.end());
        }
    }
    
    // 同步所有进程，确保所有结果已经被正确处理
    MPI_Barrier(MPI_COMM_WORLD);
}

// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
            string guess = a->ordered_values[i];
            // cout << guess << endl;
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
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

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            // cout << temp << endl;
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
    }
}

// OpenMP优化版本的Generate函数
void PriorityQueue::GenerateParallelOMP(PT pt)
{
    // 设置使用16个线程
    omp_set_num_threads(16);
    
    // 禁用嵌套并行，避免外层已在并行时这里再拉出新线程组
    omp_set_nested(0);

    // 计算 PT 的基础概率
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        // 定位最后一个 segment
        segment *a;
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)
            a = &m.digits [m.FindDigit(pt.content[0])];
        else
            a = &m.symbols[m.FindSymbol(pt.content[0])];

        // 并行之前预分配空间
        int N = pt.max_indices[0];
        size_t old_size = guesses.size();
        guesses.resize(old_size + N);

        // 并行填充
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            guesses[old_size + i] = a->ordered_values[i];
        }

        // 串行更新计数
        total_guesses += N;
    }
    else
    {
        // 构建除最后一个 segment 外的前缀
        std::string prefix;
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (seg_idx == (int)pt.content.size() - 1) break;

            const segment &segInfo = pt.content[seg_idx];
            if (segInfo.type == 1)
                prefix += m.letters[m.FindLetter(segInfo)].ordered_values[idx];
            else if (segInfo.type == 2)
                prefix += m.digits [m.FindDigit(segInfo)].ordered_values[idx];
            else
                prefix += m.symbols[m.FindSymbol(segInfo)].ordered_values[idx];

            ++seg_idx;
        }

        // 定位最后一个 segment
        segment *a;
        const segment &lastSegInfo = pt.content.back();
        if (lastSegInfo.type == 1)
            a = &m.letters[m.FindLetter(lastSegInfo)];
        else if (lastSegInfo.type == 2)
            a = &m.digits [m.FindDigit(lastSegInfo)];
        else
            a = &m.symbols[m.FindSymbol(lastSegInfo)];

        // 并行之前预分配空间
        int N = pt.max_indices.back();
        size_t old_size = guesses.size();
        guesses.resize(old_size + N);

        // 并行填充
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            guesses[old_size + i] = prefix + a->ordered_values[i];
        }

        // 串行更新计数
        total_guesses += N;
    }
}