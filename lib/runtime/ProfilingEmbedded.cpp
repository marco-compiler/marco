#ifdef MARCO_PROFILING

#include "marco/runtime/Profiling.h"
#include <vector>

Profiler::Profiler(const char*& name) : name(name)
{
}

Profiler::Profiler(const char*& other) = default;

Profiler::~Profiler() = default;

const char*& Profiler::getName() const
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

    print.print_char("\n\r");
    //printHeader(lineWidth, "Runtime statistics");
    print.print_char("Runtime statistics");

    for (const auto& profiler : profilers) {
      //printProfilerTitle(lineWidth, profiler->getName());
      print.print_char(profiler->getName());
      profiler->print();
      print.print_char("\n\r");
    }
  }

  private:
  void printHeaderLine(size_t width) const
  {
    for (size_t i = 0; i < width; ++i) {
      //std::cout << "-";
      print.print_char("-");
    }

    print.print_char("\n\r");
  }

  void printHeaderTitle(size_t width, const std::string& title) const
  {
    size_t spaces = width - 2 - title.size();
    size_t left = spaces / 2;
    size_t right = spaces - left;

    print.print_char("|");

    for (size_t i = 0; i < left; ++i) {
      print.print_char(" ");
    }

    std::cout << title.data();

    for (size_t i = 0; i < right; ++i) {
      print.print_char(" ");
    }

    print.print_char("|\n\r");
  }

  void printHeader(size_t width, const char*& title) const
  {
    printHeaderLine(width);
    printHeaderTitle(width, title);
    printHeaderLine(width);
  }

  void printProfilerTitle(size_t width, const char*& title) const
  {
    size_t symbols = width - 2 - title.size();
    size_t left = symbols / 2;
    size_t right = symbols - left;

    for (size_t i = 0; i < left; ++i) {
      print.print_char("=");
    }

    //std::cout << " " << title.data() << " ";
    print.print_char(" ");
    print.print_char(title.data());
    print.print_char(" ");

    for (size_t i = 0; i < right; ++i) {
      print.print_char("=");
    }

    print.print_char("\n");
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
