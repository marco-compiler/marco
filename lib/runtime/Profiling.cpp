#ifdef MARCO_PROFILING

#include "llvm/ADT/SmallVector.h"
#include "marco/runtime/Profiling.h"
#include <iostream>

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
    constexpr size_t lineWidth = 80;

    std::cout << "\n";
    printHeader(lineWidth, "Runtime statistics");

    for (const auto& profiler : profilers)
    {
      printProfilerTitle(lineWidth, profiler->getName());
      profiler->print();
      std::cout << "\n";
    }
  }

  private:
  void printHeaderLine(size_t width) const
  {
    for (size_t i = 0; i < width; ++i)
      std::cout << "-";

    std::cout << "\n";
  }

  void printHeaderTitle(size_t width, llvm::StringRef title) const
  {
    size_t spaces = width - 2 - title.size();
    size_t left = spaces / 2;
    size_t right = spaces - left;

    std::cout << "|";

    for (size_t i = 0; i < left; ++i)
      std::cout << " ";

    std::cout << title.data();

    for (size_t i = 0; i < right; ++i)
      std::cout << " ";

    std::cout << "|\n";
  }

  void printHeader(size_t width, llvm::StringRef title) const
  {
    printHeaderLine(width);
    printHeaderTitle(width, title);
    printHeaderLine(width);
  }

  void printProfilerTitle(size_t width, llvm::StringRef title) const
  {
    size_t symbols = width - 2 - title.size();
    size_t left = symbols / 2;
    size_t right = symbols - left;

    for (size_t i = 0; i < left; ++i)
      std::cout << "=";

    std::cout << " " << title.data() << " ";

    for (size_t i = 0; i < right; ++i)
      std::cout << "=";

    std::cout << "\n";
  }

  llvm::SmallVector<Profiler*, 3> profilers;
};

Statistics& statistics()
{
  static Statistics obj;
  return obj;
}

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
