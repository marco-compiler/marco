#ifdef MARCO_PROFILING

#include "marco/runtime/Profiling.h"
#ifndef WINDOWS_NOSTDLIB
#include <iostream>
#else
#include "marco/runtime/Printing.h"
#endif
#include <vector>

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

int stringLength(const char* str)
{
  int i;
  for(i = 0; str[i] != '\0'; ++i)
    ;
  return i;
}

Profiler::Profiler(const char* name) : name(name)
{
}

Profiler::Profiler(const Profiler& other) = default;

Profiler::~Profiler() = default;

const char* Profiler::getName() const
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
    for (const auto& profiler : profilers) {
      profiler->reset();
    }
  }

  void print() const
  {
    constexpr size_t lineWidth = 80;

    #ifndef WINDOWS_NOSTDLIB
    std::cout << "\n";
    #else
    printString("\n");
    #endif
    printHeader(lineWidth, "Runtime statistics");

    for (const auto& profiler : profilers) {
      printProfilerTitle(lineWidth, profiler->getName());
      profiler->print();
      #ifndef WINDOWS_NOSTDLIB
      std::cout << "\n";
      #else
      printString("\n");
      #endif
    }
  }

  private:
  void printHeaderLine(size_t width) const
  {
    for (size_t i = 0; i < width; ++i) {
      #ifndef WINDOWS_NOSTDLIB
      std::cout << "-";
      #else
      printString("-");
      #endif
    }

    #ifndef WINDOWS_NOSTDLIB
    std::cout << "\n";
    #else
    printString("\n");
    #endif
  }

  void printHeaderTitle(size_t width, const char* title) const
  {
    size_t spaces = width - 2 - stringLength(title);
    size_t left = spaces / 2;
    size_t right = spaces - left;

    #ifndef WINDOWS_NOSTDLIB
    std::cout << "|";
    #else
    printString("|");
    #endif

    for (size_t i = 0; i < left; ++i) {
      #ifndef WINDOWS_NOSTDLIB
      std::cout << " ";
      #else
      printString(" ");
      #endif
    }

    #ifndef WINDOWS_NOSTDLIB
    std::cout << title;
    #else
    printString(title);
    #endif

    for (size_t i = 0; i < right; ++i) {
      #ifndef WINDOWS_NOSTDLIB
      std::cout << " ";
      #else
      printString(" ");
      #endif
    }

    #ifndef WINDOWS_NOSTDLIB
    std::cout << "|\n";
    #else
    printString("|\n");
    #endif
  }

  void printHeader(size_t width, const char* title) const
  {
    printHeaderLine(width);
    printHeaderTitle(width, title);
    printHeaderLine(width);
  }

  void printProfilerTitle(size_t width, const char* title) const
  {
    size_t symbols = width - 2 - stringLength(title);
    size_t left = symbols / 2;
    size_t right = symbols - left;

    for (size_t i = 0; i < left; ++i) {
      #ifndef WINDOWS_NOSTDLIB
      std::cout << "=";
      #else
      printString("=");
      #endif
    }

    #ifndef WINDOWS_NOSTDLIB
    std::cout << " " << title << " ";
    #else
    printString(" ");
    printString(title);
    printString(" ");
    #endif
    

    for (size_t i = 0; i < right; ++i) {
      #ifndef WINDOWS_NOSTDLIB
      std::cout << "=";
      #else
      printString("=");
      #endif
    }

    #ifndef WINDOWS_NOSTDLIB
    std::cout << "\n";
    #else
    printString("\n");
    #endif
  }

  std::vector<Profiler*> profilers;
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
