#include "../../include/marco/lib/MemoryManagement.h"
#include "../../include/marco/lib/StdFunctions.h"
#include "../../include/marco/driver/heap.h"
#include "../../include/marco/lib/Print.h"
//#include <cstdlib>

#ifdef MARCO_PROFILING

#include "../../include/marco/lib/Profiling.h"
#include <chrono>
//#include <iostream>
//#include <map>



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
    print_char("Number of 'malloc' invocations: ");
    print_integer(malloCalls);
    print_char("\n\r");
    print_char("Number of 'free' invocations: ");
    print_integer(freeCalls);
    print_char("\n\r");

    if (mallocCalls > freeCalls) {
      print_char("[Warning] Possible memory leak detected\n\r");
    } else if (mallocCalls < freeCalls) {

      print_char("[Warning] Possible double 'free' detected\n");
    }

    print_char( "Total amount of heap allocated memory: " );
    print_integer(totalHeapMemory);
    print_char(" bytes\n\r");

    print_char( "Peak of heap memory usage: " );
    print_integer(totalHeapory);
    print_char(" bytes\n\r");

  
    print_char( "Time spent in heap memory management: ");
    print_integer(time());
    print_char(" ms\n\r");
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
  stde::map<void*, int64_t> sizes;
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

	//void* result = sizeInBytes == 0 ? nullptr : std::malloc(sizeInBytes);
  void* result = sizeInBytes == 0 ? nullptr : malloc(sizeInBytes);

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
    //std::free(ptr);
    free(ptr);
  }

#ifdef MARCO_PROFILING
  profiler().stopTimer();
#endif
}

RUNTIME_FUNC_DEF(heapFree, void, PTR(void))
