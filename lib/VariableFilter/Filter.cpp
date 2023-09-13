#include "marco/VariableFilter/Filter.h"

using namespace ::marco;
using namespace ::marco::vf;

namespace marco::vf
{
  Filter::Filter(bool visibility, llvm::ArrayRef<Range> ranges)
    : visibility(visibility), ranges(ranges.begin(), ranges.end())
  {
  }

  Filter Filter::visibleScalar()
  {
    return Filter(true, std::nullopt);
  }

  Filter Filter::visibleArray(llvm::ArrayRef<long> shape)
  {
    llvm::SmallVector<Range, 3> ranges;

    for (const auto& dimension : shape) {
      long start = 0;
      long end = dimension == Range::kUnbounded ? Range::kUnbounded : dimension - 1;
      ranges.emplace_back(start, end);
    }

    return Filter(true, ranges);
  }

  bool Filter::isVisible() const
  {
    return visibility;
  }

  llvm::ArrayRef<Range> Filter::getRanges() const
  {
    return ranges;
  }
}
