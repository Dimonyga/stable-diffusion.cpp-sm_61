#include "runtime/imatrix.h"

/* Adapted from llama.cpp (credits: Kawrakow). */

#include "core/util.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "stable-diffusion.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>

static IMatrixCollector imatrix_collector;

IMatrixCollector& get_imatrix_collector() {
    return imatrix_collector;
}

// remove any prefix and suffixes from the name
// CUDA0#blk.0.attn_k.weight#0 => blk.0.attn_k.weight
static std::string filter_tensor_name(const char* name) {
    std::string wname;
    const char* p = strchr(name, '#');
    if (p != NULL) {
        p             = p + 1;
        const char* q = strchr(p, '#');
        if (q != NULL) {
            wname = std::string(p, q - p);
        } else {
            wname = p;
        }
    } else {
        wname = name;
    }
    return wname;
}

void IMatrixCollector::set_save_params(const std::string& out_path, int save_frequency) {
    std::lock_guard<std::mutex> lock(mutex_);
    out_path_       = out_path;
    save_frequency_ = save_frequency;
}

void IMatrixCollector::flush() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!out_path_.empty()) {
        save_imatrix_unlocked(out_path_);
    }
}

void IMatrixCollector::step() {
    std::lock_guard<std::mutex> lock(mutex_);
    step_count_++;
    last_call_ = step_count_;
    if (save_frequency_ > 0 && !out_path_.empty() && (step_count_ % save_frequency_ == 0)) {
        LOG_INFO("imatrix: periodically saving imatrix (step %d) to %s\n", step_count_, out_path_.c_str());
        save_imatrix_unlocked(out_path_);
    }
}

bool IMatrixCollector::collect_imatrix(struct ggml_tensor* t, bool ask, void* user_data) {
    GGML_UNUSED(user_data);
    if (t == nullptr) {
        return false;
    }
    if (t->op != GGML_OP_MUL_MAT && t->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }

    const struct ggml_tensor* src0 = t->src[0];
    const struct ggml_tensor* src1 = t->src[1];
    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }
    std::string wname = filter_tensor_name(src0->name);

    // when ask is true, the scheduler wants to know if we are interested in data from this tensor
    // if we return true, a follow-up call will be made with ask=false in which we can do the actual collection
    if (ask) {
        if (t->op == GGML_OP_MUL_MAT_ID) {
            return true;  // collect all indirect matrix multiplications
        }
        if (!(wname.substr(0, 6) == "model." || wname.substr(0, 17) == "cond_stage_model." || wname.substr(0, 14) == "text_encoders.")) {
            return false;
        }
        return true;
    }
    std::lock_guard<std::mutex> lock(mutex_);

    // Detect step/iteration boundaries: when a tensor is seen again in the current pass
    if (current_step_tensors_.find(wname) != current_step_tensors_.end()) {
        current_step_tensors_.clear();
        step_count_++;
        last_call_ = step_count_;
        if (save_frequency_ > 0 && !out_path_.empty() && (step_count_ % save_frequency_ == 0)) {
            LOG_INFO("imatrix: periodically saving imatrix (step %d) to %s\n", step_count_, out_path_.c_str());
            save_imatrix_unlocked(out_path_);
        }
    }
    current_step_tensors_.insert(wname);

    // copy the data from the GPU memory if needed
    const bool is_host = src1->buffer == NULL || ggml_backend_buffer_is_host(src1->buffer);

    if (!is_host) {
        src1_data_.resize(ggml_nelements(src1));
        ggml_backend_tensor_get(src1, src1_data_.data(), 0, ggml_nbytes(src1));
    }

    const float* data = is_host ? (const float*)src1->data : src1_data_.data();

    // this has been adapted to the new format of storing merged experts in a single 3d tensor
    // ref: https://github.com/ggml-org/llama.cpp/pull/6387
    if (t->op == GGML_OP_MUL_MAT_ID) {
        //   ids  -> [n_experts_used, n_tokens]
        //   src1 -> [cols, n_expert_used, n_tokens]
        const ggml_tensor* ids = t->src[2];
        const int n_as         = static_cast<int>(src0->ne[2]);
        const int n_ids        = static_cast<int>(ids->ne[0]);

        // the top-k selected expert ids are stored in the ids tensor
        // for simplicity, always copy ids to host, because it is small
        // take into account that ids is not contiguous!

        GGML_ASSERT(ids->ne[1] == src1->ne[2]);

        ids_.resize(ggml_nbytes(ids));
        ggml_backend_tensor_get(ids, ids_.data(), 0, ggml_nbytes(ids));

        auto& e = stats_[wname];

        ++e.ncall;

        if (e.values.empty()) {
            e.values.resize(src1->ne[0] * n_as, 0);
            e.counts.resize(src1->ne[0] * n_as, 0);
            e.n_mat = n_as;
        } else if (e.values.size() != (size_t)src1->ne[0] * n_as) {
            LOG_ERROR("inconsistent size for %s (%d vs %d)\n", wname.c_str(), (int)e.values.size(), (int)src1->ne[0] * n_as);
            exit(1);  // GGML_ABORT("fatal error");
        }
        // loop over all possible experts, regardless if they are used or not in the batch
        for (int ex = 0; ex < n_as; ++ex) {
            size_t e_start = ex * src1->ne[0];

            for (int idx = 0; idx < n_ids; ++idx) {
                for (int row = 0; row < (int)src1->ne[2]; ++row) {
                    const int excur = *(const int32_t*)(ids_.data() + row * ids->nb[1] + idx * ids->nb[0]);

                    GGML_ASSERT(excur >= 0 && excur < n_as);  // sanity check

                    if (excur != ex)
                        continue;

                    const int64_t i11 = idx % src1->ne[1];
                    const int64_t i12 = row;
                    const float* x    = (const float*)((const char*)data + i11 * src1->nb[1] + i12 * src1->nb[2]);

                    for (int j = 0; j < (int)src1->ne[0]; ++j) {
                        e.values[e_start + j] += x[j] * x[j];
                        e.counts[e_start + j]++;
                        if (!std::isfinite(e.values[e_start + j])) {
                            LOG_ERROR("%f detected in %s\n", e.values[e_start + j], wname.c_str());
                            exit(1);
                        }
                    }
                }
            }
        }
    } else {
        auto& e = stats_[wname];
        if (e.values.empty()) {
            e.values.resize(src1->ne[0], 0);
            e.counts.resize(src1->ne[0], 0);
            e.n_mat = 1;
        } else if (e.values.size() != (size_t)src1->ne[0]) {
            LOG_WARN("inconsistent size for %s (%d vs %d)\n", wname.c_str(), (int)e.values.size(), (int)src1->ne[0]);
            exit(1);  // GGML_ABORT("fatal error");
        }

        ++e.ncall;
        for (int row = 0; row < (int)src1->ne[1]; ++row) {
            const float* x = data + row * src1->ne[0];
            for (int j = 0; j < (int)src1->ne[0]; ++j) {
                if (std::isfinite(x[j])) {
                    e.values[j] += x[j] * x[j];
                    e.counts[j]++;
                    if (!std::isfinite(e.values[j])) {
                        LOG_WARN("%f detected in %s\n", e.values[j], wname.c_str());
                        exit(1);
                    }
                } else {
                    // Likely something from an attention mask?
                }
            }
        }
    }
    return true;
}

bool load_imatrix(const char* imatrix_path) {
    return imatrix_collector.load_imatrix(imatrix_path);
}

void save_imatrix(const char* imatrix_path) {
    imatrix_collector.save_imatrix(imatrix_path);
}

static bool collect_imatrix(struct ggml_tensor* t, bool ask, void* user_data) {
    return imatrix_collector.collect_imatrix(t, ask, user_data);
}

void enable_imatrix_collection() {
    sd_set_backend_eval_callback(collect_imatrix, nullptr);
}

void enable_imatrix_collection_ext(const char* out_path, int save_frequency) {
    imatrix_collector.set_save_params(out_path ? out_path : "", save_frequency);
    enable_imatrix_collection();
}

void disable_imatrix_collection() {
    sd_set_backend_eval_callback(nullptr, nullptr);
}

void flush_imatrix() {
    imatrix_collector.flush();
}

void IMatrixCollector::save_imatrix(std::string fname, int ncall) const {
    std::lock_guard<std::mutex> lock(mutex_);
    save_imatrix_unlocked(fname, ncall);
}

void IMatrixCollector::save_imatrix_unlocked(const std::string& fname, int ncall) const {
    if (fname.empty()) {
        return;
    }
    bool is_dat = (fname.size() >= 4 && fname.compare(fname.size() - 4, 4, ".dat") == 0);
    if (is_dat) {
        save_imatrix_legacy_unlocked(fname, ncall);
    } else {
        save_imatrix_gguf_unlocked(fname, ncall);
    }
}

void IMatrixCollector::save_imatrix_legacy_unlocked(const std::string& fname, int ncall) const {
    std::string out_name = fname;
    if (ncall > 0) {
        out_name += ".at_";
        out_name += std::to_string(ncall);
    }

    int n_entries = 0;
    std::vector<std::string> to_store;
    to_store.reserve(stats_.size());

    for (const auto& kv : stats_) {
        const int n_all = static_cast<int>(kv.second.counts.size());
        if (n_all == 0) {
            continue;
        }

        int n_zeros = 0;
        for (const int c : kv.second.counts) {
            if (c == 0) {
                n_zeros++;
            }
        }

        if (n_zeros == n_all) {
            LOG_WARN("entry %40s has no data - skipping\n", kv.first.c_str());
            continue;
        }

        if (n_zeros > 0) {
            LOG_WARN("entry %40s has partial data (%.2f%%) - skipping\n", kv.first.c_str(), 100.0f * (n_all - n_zeros) / n_all);
            continue;
        }

        n_entries++;
        to_store.push_back(kv.first);
    }

    std::sort(to_store.begin(), to_store.end());

    if (to_store.size() < stats_.size()) {
        LOG_WARN("storing only %zu out of %zu entries\n", to_store.size(), stats_.size());
    }

    std::string tmp_name = out_name + ".tmp";
    {
        std::ofstream out(std::filesystem::u8path(tmp_name), std::ios::binary);
        if (!out) {
            LOG_ERROR("imatrix: failed to open temporary file %s for writing\n", tmp_name.c_str());
            return;
        }
        out.write((const char*)&n_entries, sizeof(n_entries));
        for (const auto& name : to_store) {
            const auto& stat = stats_.at(name);
            int len          = static_cast<int>(name.size());
            out.write((const char*)&len, sizeof(len));
            out.write(name.c_str(), len);
            out.write((const char*)&stat.ncall, sizeof(stat.ncall));
            int nval = static_cast<int>(stat.values.size());
            out.write((const char*)&nval, sizeof(nval));
            if (nval > 0) {
                std::vector<float> tmp(nval);
                for (int i = 0; i < nval; i++) {
                    float count = (i < (int)stat.counts.size()) ? static_cast<float>(stat.counts[i]) : 1.0f;
                    tmp[i]      = (count > 0.0f) ? (stat.values[i] / count) * static_cast<float>(stat.ncall) : 1.0f;
                }
                out.write((const char*)tmp.data(), nval * sizeof(float));
            }
        }
        out.write((const char*)&last_call_, sizeof(last_call_));
    }

    std::error_code ec;
#if defined(_WIN32)
    std::filesystem::remove(std::filesystem::u8path(out_name), ec);
#endif
    std::filesystem::rename(std::filesystem::u8path(tmp_name), std::filesystem::u8path(out_name), ec);
    if (ec) {
        LOG_WARN("imatrix: failed to rename %s to %s: %s\n", tmp_name.c_str(), out_name.c_str(), ec.message().c_str());
    }
}

void IMatrixCollector::save_imatrix_gguf_unlocked(const std::string& fname, int ncall) const {
    std::string out_name = fname;
    if (ncall > 0) {
        out_name += ".at_";
        out_name += std::to_string(ncall);
    }

    std::vector<std::string> to_store;
    to_store.reserve(stats_.size());

    size_t data_size = 0;
    for (const auto& kv : stats_) {
        const auto& stat = kv.second;
        const int n_all  = static_cast<int>(stat.counts.size());
        if (n_all == 0) {
            continue;
        }

        int n_zeros = 0;
        for (const int c : stat.counts) {
            if (c == 0) {
                n_zeros++;
            }
        }

        if (n_zeros == n_all) {
            LOG_WARN("entry %40s has no data - skipping\n", kv.first.c_str());
            continue;
        }

        if (n_zeros > 0) {
            LOG_WARN("entry %40s has partial data (%.2f%%)\n", kv.first.c_str(), 100.0f * (n_all - n_zeros) / n_all);
        }

        to_store.push_back(kv.first);
        int n_mat = (stat.n_mat > 0 && stat.values.size() % stat.n_mat == 0) ? stat.n_mat : 1;
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * stat.values.size(), GGML_MEM_ALIGN);
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * n_mat, GGML_MEM_ALIGN);
    }

    std::sort(to_store.begin(), to_store.end());

    data_size += ggml_tensor_overhead() * 32 + 1024 * 1024;  // headroom for ggml memory pool

    struct ggml_init_params params = {
        /* .mem_size   = */ data_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        LOG_ERROR("imatrix: failed to initialize ggml context for GGUF writing\n");
        return;
    }
    struct gguf_context* gguf = gguf_init_empty();
    if (!gguf) {
        LOG_ERROR("imatrix: failed to initialize GGUF context\n");
        ggml_free(ctx);
        return;
    }

    const char* dataset   = "stable-diffusion";
    uint32_t total_chunks = (step_count_ > 0) ? static_cast<uint32_t>(step_count_) : static_cast<uint32_t>(last_call_);
    gguf_set_val_str(gguf, "general.type", "imatrix");
    gguf_set_arr_str(gguf, "imatrix.datasets", &dataset, 1);
    gguf_set_val_u32(gguf, "imatrix.chunk_count", total_chunks);
    gguf_set_val_u32(gguf, "imatrix.chunk_size", 1);

    for (const auto& name : to_store) {
        const auto& stat = stats_.at(name);
        const int nval   = static_cast<int>(stat.values.size());
        const int n_mat  = (stat.n_mat > 0 && nval % stat.n_mat == 0) ? stat.n_mat : 1;
        const int ne0    = nval / n_mat;

        struct ggml_tensor* in_sum2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, n_mat);
        struct ggml_tensor* counts  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_mat);

        ggml_format_name(in_sum2, "%s.in_sum2", name.c_str());
        ggml_format_name(counts, "%s.counts", name.c_str());

        std::memcpy(in_sum2->data, stat.values.data(), nval * sizeof(float));

        float* cnt_data = (float*)counts->data;
        for (int ex = 0; ex < n_mat; ++ex) {
            int idx      = ex * ne0;
            cnt_data[ex] = (idx < (int)stat.counts.size()) ? static_cast<float>(stat.counts[idx]) : static_cast<float>(stat.ncall);
        }

        gguf_add_tensor(gguf, in_sum2);
        gguf_add_tensor(gguf, counts);
    }

    std::string tmp_name = out_name + ".tmp";
    bool ok              = gguf_write_to_file(gguf, tmp_name.c_str(), false);
    gguf_free(gguf);
    ggml_free(ctx);

    if (ok) {
        std::error_code ec;
#if defined(_WIN32)
        std::filesystem::remove(std::filesystem::u8path(out_name), ec);
#endif
        std::filesystem::rename(std::filesystem::u8path(tmp_name), std::filesystem::u8path(out_name), ec);
        if (ec) {
            LOG_WARN("imatrix: failed to rename %s to %s: %s\n", tmp_name.c_str(), out_name.c_str(), ec.message().c_str());
        }
    } else {
        LOG_ERROR("imatrix: failed to write GGUF to %s\n", tmp_name.c_str());
    }
}

bool IMatrixCollector::load_imatrix(const char* fname) {
    std::lock_guard<std::mutex> lock(mutex_);

    struct ggml_context* ctx                 = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ false,
        /* .ctx      = */ &ctx,
    };
    struct gguf_context* ctx_gguf = gguf_init_from_file(fname, meta_gguf_params);
    if (ctx_gguf) {
        bool ok = load_imatrix_gguf_locked(ctx_gguf, ctx, fname);
        gguf_free(ctx_gguf);
        if (ctx) {
            ggml_free(ctx);
        }
        return ok;
    }

    return load_imatrix_legacy_locked(fname);
}

bool IMatrixCollector::load_imatrix_gguf_locked(struct gguf_context* ctx_gguf, struct ggml_context* ctx, const char* fname) {
    const int32_t n_entries = gguf_get_n_tensors(ctx_gguf);
    if (n_entries < 1) {
        LOG_ERROR("no data in file %s\n", fname);
        return false;
    }

    int64_t chunk_count_key = gguf_find_key(ctx_gguf, "imatrix.chunk_count");
    if (chunk_count_key != -1) {
        int loaded_chunks = gguf_get_val_u32(ctx_gguf, chunk_count_key);
        last_call_ += loaded_chunks;
        step_count_ += loaded_chunks;
    }

    const std::string in_sum2_suffix{".in_sum2"};
    const std::string counts_suffix{".counts"};

    struct sum_tensors {
        struct ggml_tensor* in_sum2 = nullptr;
        struct ggml_tensor* counts  = nullptr;
    };

    std::unordered_map<std::string, sum_tensors> sums_counts_for;
    for (struct ggml_tensor* cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        std::string name = cur->name;
        if (name.empty()) {
            continue;
        }

        if (name.size() > in_sum2_suffix.size() &&
            name.compare(name.size() - in_sum2_suffix.size(), in_sum2_suffix.size(), in_sum2_suffix) == 0) {
            std::string base_name              = name.substr(0, name.size() - in_sum2_suffix.size());
            sums_counts_for[base_name].in_sum2 = cur;
        } else if (name.size() > counts_suffix.size() &&
                   name.compare(name.size() - counts_suffix.size(), counts_suffix.size(), counts_suffix) == 0) {
            std::string base_name              = name.substr(0, name.size() - counts_suffix.size());
            sums_counts_for[base_name].counts  = cur;
        }
    }

    for (const auto& sc : sums_counts_for) {
        const std::string& name           = sc.first;
        const struct ggml_tensor* in_sum2 = sc.second.in_sum2;
        const struct ggml_tensor* counts  = sc.second.counts;

        if (!in_sum2 || !counts) {
            LOG_WARN("missing sums or counts for %s in %s\n", name.c_str(), fname);
            continue;
        }

        if (in_sum2->type != GGML_TYPE_F32 || counts->type != GGML_TYPE_F32) {
            LOG_ERROR("sums and counts for %s must be F32\n", name.c_str());
            return false;
        }

        auto& e = stats_[name];

        const int64_t nval    = ggml_nelements(in_sum2);
        const int64_t ncounts = ggml_nelements(counts);

        if (ncounts <= 0 || nval % ncounts != 0) {
            LOG_ERROR("inconsistent tensor sizes for %s (nval=%lld, ncounts=%lld)\n", name.c_str(), (long long)nval, (long long)ncounts);
            return false;
        }

        if (e.values.empty()) {
            e.values.resize(nval, 0.0f);
            e.counts.resize(nval, 0);
            e.n_mat = static_cast<int>(ncounts);
        } else if (e.values.size() != (size_t)nval) {
            LOG_ERROR("inconsistent size for repeated entry %s (%zu vs %lld)\n", name.c_str(), e.values.size(), (long long)nval);
            return false;
        }

        const int64_t ne0     = nval / ncounts;
        const float* sum_data = (const float*)in_sum2->data;
        const float* cnt_data = (const float*)counts->data;

        for (int64_t ex = 0; ex < ncounts; ++ex) {
            const int count_val = static_cast<int>(std::lround(cnt_data[ex]));
            for (int64_t j = 0; j < ne0; ++j) {
                int64_t idx = ex * ne0 + j;
                e.values[idx] += sum_data[idx];
                e.counts[idx] += count_val;
            }
        }
        e.ncall += (ncounts > 0) ? static_cast<int>(std::lround(cnt_data[0])) : 1;
    }

    return true;
}

bool IMatrixCollector::load_imatrix_legacy_locked(const char* fname) {
    std::ifstream in(std::filesystem::u8path(fname), std::ios::binary);
    if (!in) {
        LOG_ERROR("failed to open %s\n", fname);
        return false;
    }
    int n_entries;
    in.read((char*)&n_entries, sizeof(n_entries));
    if (in.fail() || n_entries < 1) {
        LOG_ERROR("no data in file %s\n", fname);
        return false;
    }
    for (int i = 0; i < n_entries; ++i) {
        int len;
        in.read((char*)&len, sizeof(len));
        if (in.fail() || len < 0 || len > 1024 * 1024) {
            LOG_ERROR("failed reading name len for entry %d from %s\n", i + 1, fname);
            return false;
        }
        std::vector<char> name_as_vec(len + 1);
        in.read((char*)name_as_vec.data(), len);
        if (in.fail()) {
            LOG_ERROR("failed reading name for entry %d from %s\n", i + 1, fname);
            return false;
        }
        name_as_vec[len] = 0;
        std::string name{name_as_vec.data()};
        auto& e = stats_[std::move(name)];
        int ncall;
        in.read((char*)&ncall, sizeof(ncall));
        int nval;
        in.read((char*)&nval, sizeof(nval));
        if (in.fail() || nval < 1) {
            LOG_ERROR("failed reading number of values for entry %d\n", i);
            return false;
        }

        if (e.values.empty()) {
            e.values.resize(nval, 0);
            e.counts.resize(nval, 0);
            e.n_mat = 1;
        } else if (e.values.size() != (size_t)nval) {
            LOG_ERROR("inconsistent size for a repeated entry (%d vs %d)\n", (int)e.values.size(), nval);
            return false;
        }

        std::vector<float> tmp(nval);
        in.read((char*)tmp.data(), nval * sizeof(float));
        if (in.fail()) {
            LOG_ERROR("failed reading data for entry %d\n", i);
            return false;
        }

        // Recreate the state as expected by save_imatrix(), and correct for weighted sum.
        for (int j = 0; j < nval; j++) {
            e.values[j] += tmp[j];
            e.counts[j] += ncall;
        }
        e.ncall += ncall;
    }

    if (in.peek() != EOF) {
        int loaded_last_call = 0;
        in.read((char*)&loaded_last_call, sizeof(loaded_last_call));
        if (!in.fail()) {
            last_call_ += loaded_last_call;
            step_count_ += loaded_last_call;
        }
    }
    return true;
}
