#include "marco/Runtime/Profiling/Statistics.h"
#include <iostream>

namespace marco::runtime::profiling
{
  void Statistics::registerProfiler(Profiler& profiler)
  {
    profilers.push_back(&profiler);
  }

  void Statistics::registerProfiler(std::shared_ptr<Profiler> profiler)
  {
    sharedProfilers.push_back(profiler);
  }

  void Statistics::reset()
  {
    for (const auto& profiler : profilers) {
      profiler->reset();
    }
  }

  void Statistics::print() const
  {
    constexpr size_t lineWidth = 80;

    std::cerr << "\n";
    printHeader(lineWidth, "Runtime statistics");

    for (const auto& profiler : profilers) {
      printProfilerTitle(lineWidth, profiler->getName());
      profiler->print();
      std::cerr << "\n";
    }

    for (const auto& profiler : sharedProfilers) {
      printProfilerTitle(lineWidth, profiler->getName());
      profiler->print();
      std::cerr << "\n";
    }
  }

  void Statistics::printHeaderLine(size_t width) const
  {
    for (size_t i = 0; i < width; ++i) {
      std::cerr << "-";
    }

    std::cerr << "\n";
  }

  void Statistics::printHeaderTitle(
      size_t width, const std::string& title) const
  {
    size_t spaces = width - 2 - title.size();
    size_t left = spaces / 2;
    size_t right = spaces - left;

    std::cerr << "|";

    for (size_t i = 0; i < left; ++i) {
      std::cerr << " ";
    }

    std::cerr << title.data();

    for (size_t i = 0; i < right; ++i) {
      std::cerr << " ";
    }

    std::cerr << "|\n";
  }

  void Statistics::printHeader(
      size_t width, const std::string& title) const
  {
    printHeaderLine(width);
    printHeaderTitle(width, title);
    printHeaderLine(width);
  }

  void Statistics::printProfilerTitle(
      size_t width, const std::string& title) const
  {
    size_t symbols = width - 2 - title.size();
    size_t left = symbols / 2;
    size_t right = symbols - left;

    for (size_t i = 0; i < left; ++i) {
      std::cerr << "=";
    }

    std::cerr << " " << title.data() << " ";

    for (size_t i = 0; i < right; ++i) {
      std::cerr << "=";
    }

    std::cerr << "\n";
  }
}
