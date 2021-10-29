#include <marco/runtime/MemoryManagement.h>

#ifdef MARCO_PROFILING

#include <chrono>
#include <iostream>
#include <map>
#include <marco/runtime/Profiling.h>

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
    maxHeapMemory = 0;
    accumulatedTime = std::chrono::duration_values<std::chrono::nanoseconds>::zero();
  }

  void print() const override
  {
    std::cout << "Number of 'malloc' invocations: " << mallocCalls << "\n";
    std::cout << "Number of 'free' invocations: " << freeCalls << "\n";

    if (mallocCalls > freeCalls)
      std::cout << "[Warning] Possible memory leak detected\n";
    else if (mallocCalls < freeCalls)
      std::cout << "[Warning] Possible double 'free' detected\n";

    std::cout << "Total amount of heap allocated memory: " << totalHeapMemory << " bytes\n";
    std::cout << "Maximum amount of heap allocated memory at the same time: " << maxHeapMemory << " bytes\n";
    std::cout << "Time spent in heap memory management: " << time() << " ms\n";
  }

  void malloc(void* address, int64_t bytes)
  {
    ++mallocCalls;

    totalHeapMemory += bytes;
    currentHeapMemory += bytes;
    sizes[address] = bytes;

    if (currentHeapMemory > maxHeapMemory)
      maxHeapMemory = currentHeapMemory;
  }

  void free(void* address)
  {
    ++freeCalls;

    if (auto it = sizes.find(address); it != sizes.end())
    {
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
  int64_t maxHeapMemory;
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

inline void* heapAlloc(int64_t sizeInBytes)
{
#ifdef MARCO_PROFILING
  profiler().startTimer();
#endif

	void* result = malloc(sizeInBytes);

#ifdef MARCO_PROFILING
	profiler().stopTimer();
	profiler().malloc(result, sizeInBytes);
#endif

	return result;
}

RUNTIME_FUNC_DEF(heapAlloc, PTR(void), int64_t)

inline void heapFree(void* ptr)
{
#ifdef MARCO_PROFILING
  profiler().free(ptr);
  profiler().startTimer();
#endif

	free(ptr);

#ifdef MARCO_PROFILING
  profiler().stopTimer();
#endif
}

RUNTIME_FUNC_DEF(heapFree, void, PTR(void))
