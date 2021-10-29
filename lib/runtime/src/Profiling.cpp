#ifdef MARCO_PROFILING

#include <iostream>
#include <llvm/ADT/SmallVector.h>
#include <marco/runtime/Profiling.h>

Profiler::Profiler(llvm::StringRef name) : name(name.str())
{
}

llvm::StringRef Profiler::getName() const
{
  return name;
}

class Statistics
{
  public:
  void registerProfiler(Profiler& profiler)
  {
    profilers.push_back(&profiler);
  }

  void reset()
  {
    for (const auto& profiler : profilers)
      profiler->reset();
  }

  void print() const
  {
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "|                         Runtime statistics                         |\n";
    std::cout << "----------------------------------------------------------------------\n";

    for (const auto& profiler : profilers)
    {
      auto name = profiler->getName();

      size_t width = 70;
      size_t left = (width - name.size() - 2) / 2;

      for (size_t i = 0; i < left; ++i)
        std::cout << "=";

      std::cout << " " << name.data() << " ";

      for (size_t i = left + name.size() + 2; i < width; ++i)
        std::cout << "=";

      std::cout << "\n";
      profiler->print();
      std::cout << "\n";
    }
  }

  private:
  llvm::SmallVector<Profiler*, 3> profilers;
};

Statistics& statistics()
{
  static Statistics obj;
  return obj;
};

void profilingInit()
{
  statistics().reset();
}

void printProfilingStats()
{
  statistics().print();
}

void registerProfiler(Profiler& profiler)
{
  statistics().registerProfiler(profiler);
}

#endif
