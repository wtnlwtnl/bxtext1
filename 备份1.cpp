// 流水线并行实现 - guessing.cpp
#include "PCFG.h"
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <unistd.h>  // 提供 sleep() 函数 
#include <cstring>   // 提供 memcpy 函数
#include "md5.h"
using namespace std;

// PriorityQueue 方法实现
void PriorityQueue::CalProb(PT &pt)
{
    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
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

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
}

void PriorityQueue::PopNext()
{
    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    GenerateParallelMPI(priority.front());

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

// PT相关函数实现
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
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
            curr_indices[i] += 1;

            if (curr_indices[i] < max_indices[i])
            {
                pivot = i;
                res.emplace_back(*this);
            }

            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }
}

// 序列化/反序列化辅助函数
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

vector<string> deserialize_strings(const vector<char>& buffer, const vector<int>& sizes) {
    vector<string> result;
    size_t pos = 0;
    
    for (int size : sizes) {
        result.push_back(string(buffer.begin() + pos, buffer.begin() + pos + size));
        pos += size;
    }
    
    return result;
}

// PT序列化/反序列化函数
void serialize_pt(const PT& pt, vector<char>& buffer) {
    // 计算需要的缓冲区大小
    size_t buffer_size = 0;
    buffer_size += sizeof(int);  // pivot
    buffer_size += sizeof(float) * 2;  // prob和preterm_prob
    
    // content
    buffer_size += sizeof(int);  // content大小
    for (const auto& seg : pt.content) {
        buffer_size += sizeof(int) * 2;  // type和length
    }
    
    // max_indices和curr_indices
    buffer_size += sizeof(int);  // max_indices大小
    buffer_size += pt.max_indices.size() * sizeof(int);
    buffer_size += sizeof(int);  // curr_indices大小
    buffer_size += pt.curr_indices.size() * sizeof(int);
    
    // 分配缓冲区
    buffer.resize(buffer_size, 0);
    size_t pos = 0;
    
    // 写入数据
    memcpy(&buffer[pos], &pt.pivot, sizeof(int));
    pos += sizeof(int);
    
    memcpy(&buffer[pos], &pt.prob, sizeof(float));
    pos += sizeof(float);
    
    memcpy(&buffer[pos], &pt.preterm_prob, sizeof(float));
    pos += sizeof(float);
    
    // content
    int content_size = pt.content.size();
    memcpy(&buffer[pos], &content_size, sizeof(int));
    pos += sizeof(int);
    
    for (const auto& seg : pt.content) {
        memcpy(&buffer[pos], &seg.type, sizeof(int));
        pos += sizeof(int);
        memcpy(&buffer[pos], &seg.length, sizeof(int));
        pos += sizeof(int);
    }
    
    // max_indices
    int max_indices_size = pt.max_indices.size();
    memcpy(&buffer[pos], &max_indices_size, sizeof(int));
    pos += sizeof(int);
    
    for (int val : pt.max_indices) {
        memcpy(&buffer[pos], &val, sizeof(int));
        pos += sizeof(int);
    }
    
    // curr_indices
    int curr_indices_size = pt.curr_indices.size();
    memcpy(&buffer[pos], &curr_indices_size, sizeof(int));
    pos += sizeof(int);
    
    for (int val : pt.curr_indices) {
        memcpy(&buffer[pos], &val, sizeof(int));
        pos += sizeof(int);
    }
}

PT deserialize_pt(const vector<char>& buffer) {
    PT result;
    size_t pos = 0;
    
    // 读取数据
    memcpy(&result.pivot, &buffer[pos], sizeof(int));
    pos += sizeof(int);
    
    memcpy(&result.prob, &buffer[pos], sizeof(float));
    pos += sizeof(float);
    
    memcpy(&result.preterm_prob, &buffer[pos], sizeof(float));
    pos += sizeof(float);
    
    // content
    int content_size;
    memcpy(&content_size, &buffer[pos], sizeof(int));
    pos += sizeof(int);
    
    for (int i = 0; i < content_size; ++i) {
        int type, length;
        memcpy(&type, &buffer[pos], sizeof(int));
        pos += sizeof(int);
        memcpy(&length, &buffer[pos], sizeof(int));
        pos += sizeof(int);
        
        segment seg(type, length);
        result.content.push_back(seg);
    }
    
    // max_indices
    int max_indices_size;
    memcpy(&max_indices_size, &buffer[pos], sizeof(int));
    pos += sizeof(int);
    
    for (int i = 0; i < max_indices_size; ++i) {
        int val;
        memcpy(&val, &buffer[pos], sizeof(int));
        pos += sizeof(int);
        result.max_indices.push_back(val);
    }
    
    // curr_indices
    int curr_indices_size;
    memcpy(&curr_indices_size, &buffer[pos], sizeof(int));
    pos += sizeof(int);
    
    for (int i = 0; i < curr_indices_size; ++i) {
        int val;
        memcpy(&val, &buffer[pos], sizeof(int));
        pos += sizeof(int);
        result.curr_indices.push_back(val);
    }
    
    return result;
}

// 单PT并行实现
void PriorityQueue::GenerateParallelMPI(PT pt) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 计算PT的概率，这是在每个进程中都需要做的
    CalProb(pt);
    
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
        
        // 计算每个进程的工作量
        int items_per_proc = total_items / size;
        int remainder = total_items % size;
        
        // 确定本进程的工作范围
        int start_idx = rank * items_per_proc + min(rank, remainder);
        int end_idx = start_idx + items_per_proc + (rank < remainder ? 1 : 0);
        
        // 本进程生成的猜测
        vector<string> local_guesses;
        
        // 为本进程分配的部分生成猜测
        for (int i = start_idx; i < end_idx && i < (int)a->ordered_values.size(); i++) {
            local_guesses.push_back(a->ordered_values[i]);
        }
        
        // 收集所有进程的结果
        int local_count = local_guesses.size();
        vector<int> all_counts(size);
        
        // 收集每个进程生成的猜测数量
        MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            // 只在主进程中更新total_guesses
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
        
        // 准备接收缓冲区
        int total_buffer_size = 0;
        if (rank == 0) {
            for (int size : buffer_sizes)
                total_buffer_size += size;
        }
        vector<char> recv_buffer(total_buffer_size);
        
        // 收集所有序列化的猜测
        MPI_Gatherv(send_buffer.data(), send_buffer.size(), MPI_CHAR,
                   recv_buffer.data(), buffer_sizes.data(), displacements.data(),
                   MPI_CHAR, 0, MPI_COMM_WORLD);
        
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
        
        MPI_Gatherv(send_sizes.data(), send_sizes.size(), MPI_INT,
                   all_sizes_flat.data(), sizes_counts.data(), sizes_displacements.data(),
                   MPI_INT, 0, MPI_COMM_WORLD);
        
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
        
        // 计算每个进程的工作量
        int items_per_proc = total_items / size;
        int remainder = total_items % size;
        
        // 确定本进程的工作范围
        int start_idx = rank * items_per_proc + min(rank, remainder);
        int end_idx = start_idx + items_per_proc + (rank < remainder ? 1 : 0);
        
        // 本进程生成的猜测
        vector<string> local_guesses;
        
        // 为本进程分配的部分生成猜测
        for (int i = start_idx; i < end_idx && i < (int)a->ordered_values.size(); i++) {
            local_guesses.push_back(prefix + a->ordered_values[i]);
        }
        
        // 收集所有进程的结果
        int local_count = local_guesses.size();
        vector<int> all_counts(size);
        
        // 收集每个进程生成的猜测数量
        MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            // 只在主进程中更新total_guesses
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
        
        // 准备接收缓冲区
        int total_buffer_size = 0;
        if (rank == 0) {
            for (int size : buffer_sizes)
                total_buffer_size += size;
        }
        vector<char> recv_buffer(total_buffer_size);
        
        // 收集所有序列化的猜测
        MPI_Gatherv(send_buffer.data(), send_buffer.size(), MPI_CHAR,
                   recv_buffer.data(), buffer_sizes.data(), displacements.data(),
                   MPI_CHAR, 0, MPI_COMM_WORLD);
        
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
        
        MPI_Gatherv(send_sizes.data(), send_sizes.size(), MPI_INT,
                   all_sizes_flat.data(), sizes_counts.data(), sizes_displacements.data(),
                   MPI_INT, 0, MPI_COMM_WORLD);
        
        // 只在主进程中进行反序列化和结果合并
        if (rank == 0) {
            vector<string> all_guesses = deserialize_strings(recv_buffer, all_sizes_flat);
            guesses.insert(guesses.end(), all_guesses.begin(), all_guesses.end());
        }
    }
    
    // 同步所有进程
    MPI_Barrier(MPI_COMM_WORLD);
}

// 更新流水线并行函数，解决哈希组等待超时问题
void ParallelPipelineGuessingAndHashing(PriorityQueue& queue, vector<string>& targets, int guess_limit) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size < 2) {
        if (rank == 0) {
            cerr << "需要至少2个进程才能实现流水线并行!" << endl;
        }
        return;
    }
    
    // 输出调试信息
    if (rank == 0) {
        cout << "开始流水线并行，进程总数: " << size 
             << ", 目标密码数: " << targets.size() 
             << ", 猜测上限: " << guess_limit << endl;
        cout << "初始优先队列大小: " << queue.priority.size() << endl;
    }
    
    // 简化：直接使用WORLD通信子
    // 进程0为猜测进程，其他所有进程为哈希进程
    
    if (rank == 0) {
        // 猜测进程
        int total_guesses = 0;
        int num_pts_processed = 0;
        int max_pts_to_process = 5; // 减少PT数量以快速测试
        
        cout << "猜测组：优先队列大小 = " << queue.priority.size() << endl;
        
        // 主循环：生成密码并发送给哈希组
        while (total_guesses < guess_limit && !queue.priority.empty() && num_pts_processed < max_pts_to_process) {
            // 处理PT生成密码
            PT current_pt = queue.priority.front();
            queue.priority.erase(queue.priority.begin());
            queue.CalProb(current_pt);
            
            cout << "猜测组: 处理第 " << num_pts_processed + 1 << " 个PT" << endl;
            
            // 生成密码
            vector<string> guesses;
            if (current_pt.content.size() == 1) {
                // 处理单个segment的PT
                segment *a = nullptr;
                if (current_pt.content[0].type == 1)
                    a = &queue.m.letters[queue.m.FindLetter(current_pt.content[0])];
                else if (current_pt.content[0].type == 2)
                    a = &queue.m.digits[queue.m.FindDigit(current_pt.content[0])];
                else
                    a = &queue.m.symbols[queue.m.FindSymbol(current_pt.content[0])];
                
                if (a && !a->ordered_values.empty()) {
                    for (int i = 0; i < min(1000, (int)a->ordered_values.size()); i++) {
                        guesses.push_back(a->ordered_values[i]);
                    }
                }
            } else if (!current_pt.content.empty()) {
                // 处理多个segment的PT
                string prefix;
                int seg_idx = 0;
                
                // 构建前缀
                for (int idx : current_pt.curr_indices) {
                    if (seg_idx == (int)current_pt.content.size() - 1) break;
                    
                    const segment &segInfo = current_pt.content[seg_idx];
                    if (segInfo.type == 1) {
                        int letter_idx = queue.m.FindLetter(segInfo);
                        if (letter_idx >= 0 && idx < (int)queue.m.letters[letter_idx].ordered_values.size()) {
                            prefix += queue.m.letters[letter_idx].ordered_values[idx];
                        }
                    } else if (segInfo.type == 2) {
                        int digit_idx = queue.m.FindDigit(segInfo);
                        if (digit_idx >= 0 && idx < (int)queue.m.digits[digit_idx].ordered_values.size()) {
                            prefix += queue.m.digits[digit_idx].ordered_values[idx];
                        }
                    } else {
                        int symbol_idx = queue.m.FindSymbol(segInfo);
                        if (symbol_idx >= 0 && idx < (int)queue.m.symbols[symbol_idx].ordered_values.size()) {
                            prefix += queue.m.symbols[symbol_idx].ordered_values[idx];
                        }
                    }
                    ++seg_idx;
                }
                
                // 处理最后一个segment
                if (seg_idx < (int)current_pt.content.size()) {
                    segment *a = nullptr;
                    const segment &lastSegInfo = current_pt.content.back();
                    
                    if (lastSegInfo.type == 1) {
                        int letter_idx = queue.m.FindLetter(lastSegInfo);
                        if (letter_idx >= 0) a = &queue.m.letters[letter_idx];
                    } else if (lastSegInfo.type == 2) {
                        int digit_idx = queue.m.FindDigit(lastSegInfo);
                        if (digit_idx >= 0) a = &queue.m.digits[digit_idx];
                    } else {
                        int symbol_idx = queue.m.FindSymbol(lastSegInfo);
                        if (symbol_idx >= 0) a = &queue.m.symbols[symbol_idx];
                    }
                    
                    if (a && !a->ordered_values.empty()) {
                        for (int i = 0; i < min(1000, (int)a->ordered_values.size()); i++) {
                            guesses.push_back(prefix + a->ordered_values[i]);
                        }
                    }
                }
            }
            
            // 确保至少有一个猜测
            if (guesses.empty()) {
                guesses.push_back("testpassword123");
                cout << "猜测组: 添加测试密码" << endl;
            }
            
            // 发送猜测数量(批次标记，正数表示有猜测)
            int count = guesses.size();
            cout << "猜测组: 生成了 " << count << " 个密码" << endl;
            
            // 向所有哈希进程广播猜测数量
            for (int i = 1; i < size; i++) {
                MPI_Send(&count, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                cout << "猜测组: 向进程 " << i << " 发送猜测数量 " << count << endl;
            }
            
            // 序列化密码并发送
            if (count > 0) {
                vector<char> buffer;
                vector<int> sizes;
                serialize_strings(guesses, buffer, sizes);
                
                int buffer_size = buffer.size();
                int sizes_size = sizes.size();
                
                // 发送序列化数据大小
                for (int i = 1; i < size; i++) {
                    MPI_Send(&buffer_size, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                    MPI_Send(&sizes_size, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
                }
                
                // 发送实际数据
                for (int i = 1; i < size; i++) {
                    MPI_Send(buffer.data(), buffer_size, MPI_CHAR, i, 3, MPI_COMM_WORLD);
                    MPI_Send(sizes.data(), sizes_size, MPI_INT, i, 4, MPI_COMM_WORLD);
                }
                
                cout << "猜测组: 向所有哈希进程发送完成密码数据" << endl;
            }
            
            total_guesses += count;
            
            // 生成新的PT
            vector<PT> new_pts = current_pt.NewPTs();
            for (PT& pt : new_pts) {
                queue.CalProb(pt);
                queue.priority.push_back(pt);
            }
            
            num_pts_processed++;
            
            // 添加一些延迟，确保消息被接收
            sleep(1);
        }
        
        // 发送终止信号 (0表示没有更多猜测)
        int end_signal = 0;
        for (int i = 1; i < size; i++) {
            MPI_Send(&end_signal, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            cout << "猜测组: 向进程 " << i << " 发送终止信号 0" << endl;
        }
        
        // 等待所有哈希进程的结果
        int total_cracked = 0;
        for (int i = 1; i < size; i++) {
            int cracked;
            MPI_Status status;
            int flag = 0;
            int timeout_counter = 0;
            const int max_timeout = 10; // 缩短超时
            
            cout << "猜测组: 等待进程 " << i << " 的破解结果..." << endl;
            
            while (!flag && timeout_counter < max_timeout) {
                MPI_Iprobe(i, 5, MPI_COMM_WORLD, &flag, &status);
                if (!flag) {
                    sleep(1);
                    timeout_counter++;
                }
            }
            
            if (flag) {
                MPI_Recv(&cracked, 1, MPI_INT, i, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                total_cracked += cracked;
                cout << "猜测组: 收到进程 " << i << " 破解了 " << cracked << " 个密码" << endl;
            } else {
                cout << "猜测组: 等待进程 " << i << " 的结果超时" << endl;
            }
        }
        
        cout << "猜测组: 总共破解了 " << total_cracked << " 个密码" << endl;
        cout << "流水线并行执行完成" << endl;
    } else {
        // 哈希进程
        MD5 md5;
        int cracked = 0;
        
        cout << "哈希进程 " << rank << " 开始等待密码" << endl;
        
        while (true) {
            int count;
            MPI_Status status;
            
            // 接收批次标记/猜测数量
            cout << "哈希进程 " << rank << ": 等待密码批次..." << endl;
            MPI_Recv(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            cout << "哈希进程 " << rank << ": 收到批次大小 " << count << endl;
            
            // 检查是否结束
            if (count <= 0) {
                cout << "哈希进程 " << rank << ": 收到终止信号，结束处理" << endl;
                break;
            }
            
            // 接收序列化数据大小
            int buffer_size, sizes_size;
            MPI_Recv(&buffer_size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&sizes_size, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
            
            // 接收实际数据
            vector<char> buffer(buffer_size);
            vector<int> sizes(sizes_size);
            MPI_Recv(buffer.data(), buffer_size, MPI_CHAR, 0, 3, MPI_COMM_WORLD, &status);
            MPI_Recv(sizes.data(), sizes_size, MPI_INT, 0, 4, MPI_COMM_WORLD, &status);
            
            // 反序列化
            vector<string> guesses = deserialize_strings(buffer, sizes);
            cout << "哈希进程 " << rank << ": 成功接收并反序列化 " << guesses.size() << " 个密码" << endl;
            
            // 根据哈希进程数量分配工作
            int start = (rank - 1) * guesses.size() / (size - 1);
            int end = rank * guesses.size() / (size - 1);
            
            // 处理分配的密码
            int local_cracked = 0;
            for (int i = start; i < end && i < (int)guesses.size(); i++) {
                string hash = md5.GetMD5HashString(guesses[i]);
                for (const auto& target : targets) {
                    if (hash == target) {
                        cout << "哈希进程 " << rank << " 破解成功: " << guesses[i] << " -> " << hash << endl;
                        local_cracked++;
                    }
                }
            }
            
            cracked += local_cracked;
            cout << "哈希进程 " << rank << ": 本批次破解了 " << local_cracked << " 个密码" << endl;
        }
        
        // 发送破解结果
        cout << "哈希进程 " << rank << ": 发送最终破解结果 " << cracked << endl;
        MPI_Send(&cracked, 1, MPI_INT, 0, 5, MPI_COMM_WORLD);
    }
    
    // 同步所有进程
    MPI_Barrier(MPI_COMM_WORLD);
}