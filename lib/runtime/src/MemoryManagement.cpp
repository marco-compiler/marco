#include <iostream>
#include <marco/runtime/MemoryManagement.h>
#include <marco/runtime/Profiling.h>

#ifdef MARCO_PROFILING

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
  }

  void print() override
  {
    std::cout << "Number of 'malloc' invocations: " << mallocCalls << "\n";
    std::cout << "Number of 'free' invocations: " << freeCalls << "\n";
  }

  void incrementMallocCalls()
  {
    ++mallocCalls;
  }

  void incrementFreeCalls()
  {
    ++freeCalls;
  }

  private:
  size_t mallocCalls;
  size_t freeCalls;
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
  profiler().incrementMallocCalls();
#endif

	return malloc(sizeInBytes);
}

RUNTIME_FUNC_DEF(heapAlloc, PTR(void), int64_t)

inline void heapFree(void* ptr)
{
#ifdef MARCO_PROFILING
  profiler().incrementFreeCalls();
#endif

	free(ptr);
}

RUNTIME_FUNC_DEF(heapFree, void, PTR(void))
