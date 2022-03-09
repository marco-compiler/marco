#ifndef MARCO_RUNTIME_PROFILING_H
#define MARCO_RUNTIME_PROFILING_H

#include <string>

#ifdef MARCO_PROFILING

class Profiler
{
  public:
  Profiler(const std::string& name);

  Profiler(const Profiler& other);

  virtual ~Profiler();

  const std::string& getName() const;

  virtual void reset() = 0;
  virtual void print() const = 0;

  private:
  std::string name;
};

void registerProfiler(Profiler& profiler);

void profilingInit();
void printProfilingStats();

#endif

#endif //MARCO_RUNTIME_PROFILING_H
