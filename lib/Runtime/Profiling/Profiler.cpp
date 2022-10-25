#include "marco/Runtime/Profiling/Profiler.h"
#include <iostream>
#include <vector>

namespace marco::runtime::profiling
{
  Profiler::Profiler(const std::string& name)
      : name(name)
  {
  }

  Profiler::~Profiler() = default;

  const std::string& Profiler::getName() const
  {
    return name;
  }
}
