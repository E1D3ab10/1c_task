#include <iostream>
#include <filesystem>
#include <vector>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <string>
#include <thread>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <atomic>
#include <functional>
#include <tuple>
#include <unordered_set>

template <typename T>
class UnboundedBlockingQueue {
public:
    void Push(T item) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (closed_) return;
        buffer_.push_back(std::move(item));
        empty_sleep_.notify_one();
    }

    std::optional<T> Pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (buffer_.empty() && !closed_) {
            empty_sleep_.wait(lock);
        }
        if (closed_ && buffer_.empty()) {
            return std::nullopt;
        }
        T front = std::move(buffer_.front());
        buffer_.pop_front();
        return front;
    }

    void Close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        empty_sleep_.notify_all();
    }

private:
    bool closed_{false};
    std::deque<T> buffer_;
    std::mutex mutex_;
    std::condition_variable empty_sleep_;
};

class ThreadPool {
public:
    using Task = std::function<void()>;

    explicit ThreadPool(size_t threads)
        : num_threads_(threads), stopped_(false) {}

    ~ThreadPool() {
        if (!stopped_.load()) {
            Stop();
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    void Start() {
        for (size_t i = 0; i < num_threads_; ++i) {
            workers_.emplace_back([this] {
                Work();
            });
        }
    }

    void Submit(Task task) {
        tasks_.Push(std::move(task));
    }

    void Stop() {
        tasks_.Close();
        stopped_.store(true);
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    static ThreadPool* Current() {
        return curr_thread_pool;
    }

private:
    void Work() {
        curr_thread_pool = this;
        while (auto task = tasks_.Pop()) {
            (*task)();
        }
    }

private:
    static thread_local ThreadPool* curr_thread_pool;
    std::atomic<bool> stopped_;
    const size_t num_threads_;
    std::vector<std::thread> workers_;
    UnboundedBlockingQueue<Task> tasks_;
};

thread_local ThreadPool* ThreadPool::curr_thread_pool = nullptr;

struct FileInfo {
    std::string filename;
    std::vector<unsigned char> content;

    explicit FileInfo(const std::filesystem::path& p)
        : filename(p.filename().generic_string()) {
        std::ifstream file(p, std::ios::binary);
        file.seekg(0, std::ios::end);
        std::size_t size = file.tellg();
        file.seekg(0);
        content.resize(size);
        file.read(reinterpret_cast<char*>(content.data()), size);
    }
};

std::vector<FileInfo> read_all_files(const std::string& folder) {
    std::vector<FileInfo> result;
    for (const auto& entry : std::filesystem::directory_iterator(folder)) {
        result.emplace_back(entry.path());
    }
    return result;
}

double compute_lcs_similarity(const std::vector<unsigned char>& x, const std::vector<unsigned char>& y) {
    if (x.empty() && y.empty()) return 100.0;
    if (x.empty() || y.empty()) return 0.0;

    std::size_t n = x.size();
    std::size_t m = y.size();
    std::vector<std::vector<std::size_t>> dp(n + 1, std::vector<std::size_t>(m + 1, 0));

    for (std::size_t i = 1; i <= n; ++i) {
        for (std::size_t j = 1; j <= m; ++j) {
            if (x[i-1] == y[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = std::max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }

    std::size_t lcs_length = dp[n][m];
    return (static_cast<double>(lcs_length) / std::max(n, m)) * 100.0;
}

double compute_block_similarity(const std::vector<unsigned char>& x, const std::vector<unsigned char>& y) {
    if (x.empty() && y.empty()) return 100.0;
    if (x.empty() || y.empty()) return 0.0;

    const std::size_t block_size = 64;

    auto extract_block_hashes = [&](const std::vector<unsigned char>& data) {
        std::unordered_set<std::size_t> hashes;
        std::hash<std::string_view> hasher;
        for (std::size_t i = 0; i < data.size(); i += block_size) {
            std::size_t len = std::min(block_size, data.size() - i);
            std::string_view chunk(reinterpret_cast<const char*>(&data[i]), len);
            hashes.insert(hasher(chunk));
        }
        return hashes;
    };

    auto blocks_a = extract_block_hashes(x);
    auto blocks_b = extract_block_hashes(y);

    std::size_t common = 0;
    for (std::size_t h : blocks_a) {
        if (blocks_b.find(h) != blocks_b.end()) {
            ++common;
        }
    }

    std::size_t max_block_count = std::max(blocks_a.size(), blocks_b.size());
    return (static_cast<double>(common) / max_block_count) * 100.0;
}

double calculate_file_similarity(const FileInfo& a, const FileInfo& b) {
    if (a.content == b.content) {
        return 100.0;
    }

    const std::size_t small_file_limit = 4096;
    if (a.content.size() <= small_file_limit && b.content.size() <= small_file_limit) {
        return compute_lcs_similarity(a.content, b.content);
    }

    return compute_block_similarity(a.content, b.content);
}

void run_comparison(const std::string& dir_a, const std::string& dir_b, double min_percent) {
    auto files_a = read_all_files(dir_a);
    auto files_b = read_all_files(dir_b);

    std::size_t total = files_a.size() * files_b.size();
    std::vector<std::tuple<std::size_t, std::size_t, double>> matches;
    std::mutex output_lock;

    if (total >= 100) {
        ThreadPool pool(std::thread::hardware_concurrency());
        pool.Start();

        for (std::size_t idx = 0; idx < total; ++idx) {
            pool.Submit([&, idx]() {
                std::size_t i = idx / files_b.size();
                std::size_t j = idx % files_b.size();
                double score = calculate_file_similarity(files_a[i], files_b[j]);
                if (score >= min_percent) {
                    std::lock_guard<std::mutex> guard(output_lock);
                    matches.emplace_back(i, j, score);
                }
            });
        }

        pool.Stop();
    } else {
        for (std::size_t i = 0; i < files_a.size(); ++i) {
            for (std::size_t j = 0; j < files_b.size(); ++j) {
                double score = calculate_file_similarity(files_a[i], files_b[j]);
                if (score >= min_percent) {
                    matches.emplace_back(i, j, score);
                }
            }
        }
    }

    std::vector<bool> used_a(files_a.size(), false);
    std::vector<bool> used_b(files_b.size(), false);
    std::vector<std::tuple<std::string, std::string, double>> exact, similar;

    std::sort(matches.begin(), matches.end(),
        [](const auto& left, const auto& right) {
            return std::get<2>(left) > std::get<2>(right);
        });

    for (const auto& [i, j, score] : matches) {
        if (used_a[i] || used_b[j]) continue;

        std::string path_a = "Директория1/" + files_a[i].filename;
        std::string path_b = "Директория2/" + files_b[j].filename;

        if (score == 100.0) {
            exact.emplace_back(path_a, path_b, score);
        } else {
            similar.emplace_back(path_a, path_b, score);
        }

        used_a[i] = true;
        used_b[j] = true;
    }

    if (!exact.empty()) {
        std::cout << "=== Идентичные файлы ===\n";
        for (const auto& [p1, p2, _] : exact) {
            std::cout << p1 << " - " << p2 << "\n";
        }
    }

    if (!similar.empty()) {
        std::cout << "\n=== Похожие файлы ===\n";
        for (const auto& [p1, p2, score] : similar) {
            std::cout << p1 << " - " << p2 << " - "
                      << std::fixed << std::setprecision(0) << score << "%\n";
        }
    }

    bool found = false;
    for (std::size_t i = 0; i < files_a.size(); ++i) {
        if (!used_a[i]) {
            if (!found) {
                std::cout << "\n=== Только в директории 1 ===\n";
                found = true;
            }
            std::cout << files_a[i].filename << "\n";
        }
    }

    found = false;
    for (std::size_t i = 0; i < files_b.size(); ++i) {
        if (!used_b[i]) {
            if (!found) {
                std::cout << "\n=== Только в директории 2 ===\n";
                found = true;
            }
            std::cout << files_b[i].filename << "\n";
        }
    }
}

int main(int argc, char* argv[]) {
    std::string folder1 = argv[1];
    std::string folder2 = argv[2];
    double threshold = std::stod(argv[3]);

    run_comparison(folder1, folder2, threshold);
    return 0;
}