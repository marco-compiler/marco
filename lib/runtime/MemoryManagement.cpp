#include "marco/runtime/MemoryManagement.h"
#ifndef WINDOWS_NOSTDLIB
#include <cstdlib>
#else
#include <Windows.h>
#endif

#ifdef MARCO_PROFILING

#include "marco/runtime/Profiling.h"
#ifndef WINDOWS_NOSTDLIB
#include <chrono>
#include <iostream>
#include <map>
#else
#include "marco/runtime/Printing.h"
#endif

#ifdef WINDOWS_NOSTDLIB
// This is needed because static objects are thread safe

inline int __cxa_guard_acquire(uint64_t* guard_object)
{
  return 1;
}

inline void __cxa_guard_release(uint64_t* guard_object)
{
  return;
}

inline int atexit (void (*func)(void))
{
  return 0;
}
#endif

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
    #ifndef WINDOWS_NOSTDLIB
    accumulatedTime = std::chrono::duration_values<std::chrono::nanoseconds>::zero();
    #else
    accumulatedTime = 0;
    #endif
  }

  void print() const override
  {
    #ifndef WINDOWS_NOSTDLIB
    std::cout << "Number of 'malloc' invocations: " << mallocCalls << "\n";
    std::cout << "Number of 'free' invocations: " << freeCalls << "\n";

    if (mallocCalls > freeCalls) {
      std::cout << "[Warning] Possible memory leak detected\n";
    } else if (mallocCalls < freeCalls) {
      std::cout << "[Warning] Possible double 'free' detected\n";
    }

    std::cout << "Total amount of heap allocated memory: " << totalHeapMemory << " bytes\n";
    std::cout << "Peak of heap memory usage: " << peakHeapMemory << " bytes\n";
    std::cout << "Time spent in heap memory management: " << time() << " ms\n";
    #else
    ryuPrintf("Number of 'malloc' invocations: %d\n", mallocCalls);
    ryuPrintf("Number of 'free' invocations: %d\n", freeCalls);

    if (mallocCalls > freeCalls) {
      ryuPrintf("[Warning] Possible memory leak detected\n");
    } else if (mallocCalls < freeCalls) {
      ryuPrintf("[Warning] Possible double 'free' detected\n");
    }

    ryuPrintf("Total amount of heap allocated memory: %d bytes\n", totalHeapMemory);
    ryuPrintf("Peak of heap memory usage: %d bytes\n", peakHeapMemory);
    ryuPrintf("Time spent in heap memory management: %d ms\n", time());
    #endif
  }

  void malloc(void* address, int64_t bytes)
  {
    ++mallocCalls;

    totalHeapMemory += bytes;
    currentHeapMemory += bytes;
    #ifndef WINDOWS_NOSTDLIB
    sizes[address] = bytes;
    #endif

    if (currentHeapMemory > peakHeapMemory) {
      peakHeapMemory = currentHeapMemory;
    }
  }

  void free(void* address)
  {
    ++freeCalls;

    #ifndef WINDOWS_NOSTDLIB
    if (auto it = sizes.find(address); it != sizes.end()) {
      currentHeapMemory -= it->second;
      sizes.erase(it);
    }
    #endif
  }

  void startTimer()
  {
    #ifndef WINDOWS_NOSTDLIB
    start = std::chrono::steady_clock::now();
    #else
    LARGE_INTEGER li;
    if(!QueryPerformanceFrequency(&li))
      ryuPrintf("QueryPerformanceFrequency failed!\n");
    freq = double(li.QuadPart)/1000000; // nanoseconds
    QueryPerformanceCounter(&li);
    start = li.QuadPart;
    #endif
  }

  void stopTimer()
  {
    #ifndef WINDOWS_NOSTDLIB
    accumulatedTime += (std::chrono::steady_clock::now() - start);
    #else
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    accumulatedTime += double(li.QuadPart - start)/freq;
    #endif
  }

  double time() const
  {
    #ifndef WINDOWS_NOSTDLIB
    return static_cast<double>(accumulatedTime.count()) / 1e6;
    #else
    return accumulatedTime;
    #endif
  }

  private:
  size_t mallocCalls;
  size_t freeCalls;
  int64_t totalHeapMemory;
  int64_t currentHeapMemory;
  int64_t peakHeapMemory;
  #ifndef WINDOWS_NOSTDLIB
  std::map<void*, int64_t> sizes;
  std::chrono::steady_clock::time_point start;
  std::chrono::nanoseconds accumulatedTime;
  #else
  __int64 start;
  double accumulatedTime;
  double freq;
  #endif
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

  #ifndef WINDOWS_NOSTDLIB
	void* result = sizeInBytes == 0 ? nullptr : std::malloc(sizeInBytes);
  #else
	void* result = sizeInBytes == 0 ? nullptr : HeapAlloc(GetProcessHeap(), 0x0, sizeInBytes);
  #endif

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
    #ifndef WINDOWS_NOSTDLIB
    std::free(ptr);
    #else
    HeapFree(GetProcessHeap(), 0x0, ptr);
    #endif
  }

#ifdef MARCO_PROFILING
  profiler().stopTimer();
#endif
}

RUNTIME_FUNC_DEF(heapFree, void, PTR(void))
