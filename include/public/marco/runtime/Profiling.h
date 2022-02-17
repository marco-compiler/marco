#ifndef MARCO_RUNTIME_PROFILING_H
#define MARCO_RUNTIME_PROFILING_H

#ifdef MARCO_PROFILING

#ifndef WINDOWS_NOSTDLIB
#include <string>
#endif

class Profiler
{
  public:
  Profiler(const char* name);

  Profiler(const Profiler& other);

  virtual ~Profiler();

  const char* getName() const;

  virtual void reset() = 0;
  virtual void print() const = 0;

  private:
  const char* name;
};

void registerProfiler(Profiler& profiler);

void profilingInit();
void printProfilingStats();

#endif

#endif //MARCO_RUNTIME_PROFILING_H
