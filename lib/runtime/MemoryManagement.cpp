#include "marco/runtime/MemoryManagement.h"
#include <cstdlib>

#ifdef MARCO_PROFILING

#include "marco/runtime/Profiling.h"
#include <chrono>
#include <iostream>
#include <map>

class MemoryProfiler : public Profiler
{
  public:
  MemoryProfiler() : Profiler("Memory management")
  {
    registerProfiler(*this);
  }

  void reset() override
  {
    mallocCalls = 0;
    freeCalls = 0;
    totalHeapMemory = 0;
    currentHeapMemory = 0;
    peakHeapMemory = 0;
    accumulatedTime = std::chrono::duration_values<std::chrono::nanoseconds>::zero();
  }

  void print() const override
  {
    std::cerr << "Number of 'malloc' invocations: " << mallocCalls << "\n";
    std::cerr << "Number of 'free' invocations: " << freeCalls << "\n";

    if (mallocCalls > freeCalls) {
      std::cerr << "[Warning] Possible memory leak detected\n";
    } else if (mallocCalls < freeCalls) {
      std::cerr << "[Warning] Possible double 'free' detected\n";
    }

    std::cerr << "Total amount of heap allocated memory: " << totalHeapMemory << " bytes\n";
    std::cerr << "Peak of heap memory usage: " << peakHeapMemory << " bytes\n";
    std::cerr << "Time spent in heap memory management: " << time() << " ms\n";
  }

  void malloc(void* address, int64_t bytes)
  {
    ++mallocCalls;

    totalHeapMemory += bytes;
    currentHeapMemory += bytes;
    sizes[address] = bytes;

    if (currentHeapMemory > peakHeapMemory) {
      peakHeapMemory = currentHeapMemory;
    }
  }

  void free(void* address)
  {
    ++freeCalls;

    if (auto it = sizes.find(address); it != sizes.end()) {
      currentHeapMemory -= it->second;
      sizes.erase(it);
    }
  }

  void startTimer()
  {
    start = std::chrono::steady_clock::now();
  }

  void stopTimer()
  {
    accumulatedTime += (std::chrono::steady_clock::now() - start);
  }

  double time() const
  {
    return static_cast<double>(accumulatedTime.count()) / 1e6;
  }

  private:
  size_t mallocCalls;
  size_t freeCalls;
  int64_t totalHeapMemory;
  int64_t currentHeapMemory;
  int64_t peakHeapMemory;
  std::map<void*, int64_t> sizes;
  std::chrono::steady_clock::time_point start;
  std::chrono::nanoseconds accumulatedTime;
};

MemoryProfiler& profiler()
{
  static MemoryProfiler obj;
  return obj;
}

#endif

inline void* heapAlloc_pvoid(int64_t sizeInBytes)
{
#ifdef MARCO_PROFILING
  profiler().startTimer();
#endif

	void* result = sizeInBytes == 0 ? nullptr : std::malloc(sizeInBytes);

#ifdef MARCO_PROFILING
	profiler().stopTimer();
	profiler().malloc(result, sizeInBytes);
#endif

	return result;
}

RUNTIME_FUNC_DEF(heapAlloc, PTR(void), int64_t)

inline void heapFree_void(void* ptr)
{
#ifdef MARCO_PROFILING
  profiler().free(ptr);
  profiler().startTimer();
#endif

  if (ptr != nullptr) {
    std::free(ptr);
  }

#ifdef MARCO_PROFILING
  profiler().stopTimer();
#endif
}

RUNTIME_FUNC_DEF(heapFree, void, PTR(void))
