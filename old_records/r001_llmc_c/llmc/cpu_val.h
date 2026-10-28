/*
The validation rows built on CPUs (HINT_CPU_VAL=1), instead of hint_stats.cuh's GPU build: the Rust library cpuval/
(C ABI below; build.sh links it) runs in rank 0's process, in two thread pools of its own. From step 0, once the
training rows are done, it builds whatever its engines count from the training shards alone, on the CPUs the ranks
leave (HINT_CPU_VAL_CPUS). After the last step, on the clock, every rank waits for that build (training-side work);
then the clock stops. Only then, like the validation forward passes, does rank 0 read the validation tokens and the
library look them up (on HINT_CPU_VAL_LOOKUP_CPUS: every CPU, the ranks wait asleep), writing the rows (llm.c's Rows
layout: tot / dist / cnt [N, NLEVELS], cell [N], tokens / weights [N, ENTRIES]) into a shared /dev/shm mapping that all
the ranks map and pin before the clock. Every rank waits for rank 0's ready flag, copies the rows of its own
validation batches to its GPU, and meets the others (a barrier).

Env: HINT_CPU_VAL_SHM (the mapping's path, default /dev/shm/llmc_cpuval_<uid>_<parent pid>; unlinked once mapped),
HINT_CPU_VAL_CHECK=1 (the GPU build too, after the CPU rows, compared at every position: for a configuration where
every level counts over all the shards); the build's own settings (HINT_CPU_VAL_LEVEL_SHARDS, _ENGINES, _CPUS, ...)
are read by the library (cpuval/src/lib.rs).
*/
#ifndef LLMC_CPU_VAL_H
#define LLMC_CPU_VAL_H

#include <atomic>
#include <string>
#include <fcntl.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "hint_stats.cuh"

extern "C" {
struct CpuValOut { int16_t* cell; uint16_t* tokens; uint16_t* weights; uint32_t* tot; uint16_t* dist; uint32_t* cnt; };
void* cpuval_new(const char* shard_list, char* err, size_t err_len);
void cpuval_describe(const void* h, char* buf, size_t len);
int cpuval_scans(const void* h);
int cpuval_start(void* h);
int cpuval_wait(void* h, char* report, size_t len);
int cpuval_lookup(void* h, const uint16_t* val, size_t n, CpuValOut out, char* report, size_t len);
void cpuval_free(void* h);
}

namespace hint_stats {

struct CpuValBuild {
    // the mapping's header: the build's and the rows' flags (0 not yet, 1 done, 2 failed) and rank 0's times after
    // the clock
    struct Header { std::atomic<int> built, ready; double read_s, lookup_s; };
    int rank = 0, world = 1;
    i64 N = VAL_TOKENS, BT = 0;
    void* lib = nullptr;  // rank 0's build
    std::string val_path;  // read after the clock only
    std::vector<uint16_t> val_tokens;  // rank 0: their buffer, allocated (pages faulted) before the clock
    char* base = nullptr; size_t bytes = 0;
    Header* head = nullptr;
    CpuValOut host = {};
    bool pinned = false, copy_all = false;
    std::string describe, report;
    // timings (s): the build's start (wall); after the last step, on the clock: the wait for the build, its end (wall,
    // since the build's start); after the clock: the wait for the rows (rank 0: reading the validation tokens, the
    // lookup), their copies, in all
    double t_start = 0, t_built = 0, t_built_end = 0, t_wait = 0, t_copy = 0, t_total = 0;

    static size_t up(size_t b) { return (b + 4095) / 4096 * 4096; }

    // before the clock: rank 0 creates the mapping (zeroed, its pages faulted) and its build; every rank maps and pins it
    void setup(const std::string& shard_list, const std::string& val, int rank_, int world_, i64 BT_, bool check) {
        rank = rank_; world = world_; BT = BT_; copy_all = check; val_path = val;
        std::string path = getenv("HINT_CPU_VAL_SHM") ? getenv("HINT_CPU_VAL_SHM")
                         : "/dev/shm/llmc_cpuval_" + std::to_string(getuid()) + "_" + std::to_string(getppid());
        const size_t o_tot = 4096, o_cnt = o_tot + up(N * NLEVELS * 4), o_dist = o_cnt + up(N * NLEVELS * 4);
        const size_t o_tok = o_dist + up(N * NLEVELS * 2), o_w = o_tok + up(N * ENTRIES * 2), o_cell = o_w + up(N * ENTRIES * 2);
        bytes = o_cell + up(N * 2);
        int fd = -1;
        if (rank == 0) {
            char err[4096] = {0};
            lib = cpuval_new(shard_list.c_str(), err, sizeof(err));
            if (!lib) { fprintf(stderr, "cpu val: %s\n", err); exit(EXIT_FAILURE); }
            char buf[4096];
            cpuval_describe(lib, buf, sizeof(buf));
            describe = buf;
            if (cpuval_scans(lib) && !check) {  // its pass over the shards would run after the clock
                fprintf(stderr, "cpu val: the scan engine is for checks only (HINT_CPU_VAL_CHECK=1)\n");
                exit(EXIT_FAILURE);
            }
            val_tokens.assign(N + 1, 0);
            fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
            if (fd < 0 || ftruncate(fd, bytes) != 0) { fprintf(stderr, "cpu val: cannot create %s\n", path.c_str()); exit(EXIT_FAILURE); }
        }
        multi_gpu_barrier(&multi_gpu_config);
        if (rank != 0) fd = open(path.c_str(), O_RDWR);
        if (fd < 0) { fprintf(stderr, "cpu val: cannot open %s\n", path.c_str()); exit(EXIT_FAILURE); }
        base = (char*)mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) { fprintf(stderr, "cpu val: mmap of %s failed\n", path.c_str()); exit(EXIT_FAILURE); }
        close(fd);
        if (rank == 0) memset(base, 0, bytes);  // the pages allocated before the clock
        multi_gpu_barrier(&multi_gpu_config);
        // pinned for the copies (each rank its own view of the shared pages)
        pinned = cudaHostRegister(base, bytes, cudaHostRegisterDefault) == cudaSuccess;
        if (!pinned) (void)cudaGetLastError();
        multi_gpu_barrier(&multi_gpu_config);
        if (rank == 0) unlink(path.c_str());  // mapped everywhere: nothing is left behind
        head = (Header*)base;
        host = {(int16_t*)(base + o_cell), (uint16_t*)(base + o_tok), (uint16_t*)(base + o_w), (uint32_t*)(base + o_tot),
                (uint16_t*)(base + o_dist), (uint32_t*)(base + o_cnt)};
    }

    // at step 0, after the training rows (on the clock, beside the training)
    void start() {
        t_start = now_s();
        if (rank == 0 && cpuval_start(lib) != 0) { fprintf(stderr, "cpu val: the build did not start\n"); exit(EXIT_FAILURE); }
    }

    // a flag of the header set by rank 0: every rank waits for it (asleep), and all exit if it says failed
    void await_flag(std::atomic<int>& flag, const char* what) {
        const timespec nap = {0, 20000};
        while (flag.load(std::memory_order_acquire) == 0) { nanosleep(&nap, nullptr); }
        if (flag.load(std::memory_order_acquire) != 1) {
            fprintf(stderr, "cpu val: %s failed: %s\n", what, rank == 0 ? report.c_str() : "(see rank 0)");
            exit(EXIT_FAILURE);
        }
    }

    // after the last step, on the clock: every rank waits for rank 0's build (training-side work)
    void wait() {
        const double t0 = now_s();
        if (rank == 0) {
            static char buf[1 << 12];
            const int rc = cpuval_wait(lib, buf, sizeof(buf));
            report = buf;
            head->built.store(rc == 0 ? 1 : 2, std::memory_order_release);
        }
        await_flag(head->built, "the build");
        t_built = now_s() - t0;
        t_built_end = now_s() - t_start;
    }

    // the first n + 1 validation tokens into `tokens` (after the clock)
    static void read_val(const std::string& path, i64 n, std::vector<uint16_t>& tokens) {
        int header[256];
        FILE* f = fopenCheck(path.c_str(), "rb");
        freadCheck(header, sizeof(int), 256, f);
        if (header[0] != 20240520 || header[2] < n + 1) { fprintf(stderr, "cpu val: %s is no validation shard of %lld tokens\n", path.c_str(), (long long)n + 1); exit(EXIT_FAILURE); }
        freadCheck(tokens.data(), sizeof(uint16_t), n + 1, f);
        fcloseCheck(f);
    }

    // after the clock: rank 0 reads the validation tokens and looks them up; every rank waits for the rows, copies its
    // validation batches' rows (all with copy_all) to `out`
    void finish(Rows& out, cudaStream_t st) {
        const double t0 = now_s();
        if (rank == 0) {
            read_val(val_path, N, val_tokens);
            head->read_s = now_s() - t0;
            static char buf[1 << 16];
            const int rc = cpuval_lookup(lib, val_tokens.data(), (size_t)N, host, buf, sizeof(buf));
            report = buf;
            head->lookup_s = now_s() - t0 - head->read_s;
            head->ready.store(rc == 0 ? 1 : 2, std::memory_order_release);
        }
        await_flag(head->ready, "the lookup");  // the others asleep: the lookup has every CPU
        t_wait = now_s() - t0;
        for (i64 c = copy_all ? 0 : rank; c * BT < N; c += copy_all ? 1 : world) {
            const i64 a = c * BT, n = std::min(BT, N - a);
            cudaCheck(cudaMemcpyAsync(out.tot + a * NLEVELS, host.tot + a * NLEVELS, n * NLEVELS * 4, cudaMemcpyHostToDevice, st));
            cudaCheck(cudaMemcpyAsync(out.cnt + a * NLEVELS, host.cnt + a * NLEVELS, n * NLEVELS * 4, cudaMemcpyHostToDevice, st));
            cudaCheck(cudaMemcpyAsync(out.dist + a * NLEVELS, host.dist + a * NLEVELS, n * NLEVELS * 2, cudaMemcpyHostToDevice, st));
            cudaCheck(cudaMemcpyAsync(out.tokens + a * ENTRIES, host.tokens + a * ENTRIES, n * ENTRIES * 2, cudaMemcpyHostToDevice, st));
            cudaCheck(cudaMemcpyAsync(out.weights + a * ENTRIES, host.weights + a * ENTRIES, n * ENTRIES * 2, cudaMemcpyHostToDevice, st));
            cudaCheck(cudaMemcpyAsync(out.cell + a, host.cell + a, n * 2, cudaMemcpyHostToDevice, st));
        }
        cudaCheck(cudaStreamSynchronize(st));
        t_copy = now_s() - t0 - t_wait;
        multi_gpu_barrier(&multi_gpu_config);
        t_total = now_s() - t0;
    }

    void release() {
        if (base) {
            if (pinned) cudaCheck(cudaHostUnregister(base));
            munmap(base, bytes);
            base = nullptr;
        }
        if (lib) { cpuval_free(lib); lib = nullptr; }
    }
};

}  // namespace hint_stats

#endif  // LLMC_CPU_VAL_H
