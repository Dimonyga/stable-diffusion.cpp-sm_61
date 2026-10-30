#ifndef __SD_RUNTIME_IMATRIX_H__
#define __SD_RUNTIME_IMATRIX_H__

#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/* Adapted from llama.cpp (credits: Kawrakow). */

struct ggml_tensor;
struct gguf_context;
struct ggml_context;

struct IMatrixStats {
    std::vector<float> values{};
    std::vector<int> counts{};
    int ncall = 0;
    int n_mat = 1;
};

class IMatrixCollector {
private:
    std::unordered_map<std::string, IMatrixStats> stats_ = {};
    std::unordered_set<std::string> current_step_tensors_ = {};
    mutable std::mutex mutex_;
    int last_call_ = 0;
    int step_count_ = 0;
    int save_frequency_ = 0;
    std::string out_path_ = "";
    std::vector<float> src1_data_;
    std::vector<char> ids_;  // the expert ids from ggml_mul_mat_id

    void save_imatrix_legacy_unlocked(const std::string& fname, int ncall = -1) const;
    void save_imatrix_gguf_unlocked(const std::string& fname, int ncall = -1) const;
    void save_imatrix_unlocked(const std::string& fname, int ncall = -1) const;
    bool load_imatrix_legacy_locked(const char* fname);
    bool load_imatrix_gguf_locked(struct gguf_context* ctx_gguf, struct ggml_context* ctx, const char* fname);

public:
    IMatrixCollector() = default;
    bool collect_imatrix(struct ggml_tensor* t, bool ask, void* user_data);
    void save_imatrix(std::string fname, int ncall = -1) const;
    bool load_imatrix(const char* fname);
    void set_save_params(const std::string& out_path, int save_frequency);
    void flush() const;
    void step();

    std::vector<float> get_values(const std::string& key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = stats_.find(key);
        if (it == stats_.end()) {
            return {};
        }
        const auto& stat = it->second;
        size_t nval = stat.values.size();
        if (nval == 0) {
            return {};
        }
        std::vector<float> res(nval);
        size_t ncounts = stat.counts.size();
        if (ncounts == nval) {
            for (size_t i = 0; i < nval; ++i) {
                float c = static_cast<float>(stat.counts[i]);
                res[i] = (c > 0.0f) ? (stat.values[i] / c) : 1.0f;
            }
        } else if (ncounts > 0 && nval % ncounts == 0) {
            size_t ne0 = nval / ncounts;
            for (size_t ex = 0; ex < ncounts; ++ex) {
                float c = static_cast<float>(stat.counts[ex]);
                for (size_t j = 0; j < ne0; ++j) {
                    size_t idx = ex * ne0 + j;
                    res[idx] = (c > 0.0f) ? (stat.values[idx] / c) : 1.0f;
                }
            }
        } else if (stat.ncall > 0) {
            float c = static_cast<float>(stat.ncall);
            for (size_t i = 0; i < nval; ++i) {
                res[i] = stat.values[i] / c;
            }
        } else {
            for (size_t i = 0; i < nval; ++i) {
                res[i] = 1.0f;
            }
        }
        return res;
    }
};

IMatrixCollector& get_imatrix_collector();

#endif  // __SD_RUNTIME_IMATRIX_H__
