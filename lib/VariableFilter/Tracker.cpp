#include "marco/VariableFilter/Tracker.h"

using namespace ::marco;
using namespace ::marco::vf;

namespace marco::vf
{
  Tracker::Tracker()
  {
  }

  Tracker::Tracker(llvm::StringRef name)
    : Tracker(name, std::nullopt)
  {
  }

  Tracker::Tracker(llvm::StringRef name, llvm::ArrayRef<Range> ranges)
    : name(name.str()),
      ranges(ranges.begin(), ranges.end())
  {
  }

  void Tracker::setRanges(llvm::ArrayRef<Range> newRanges)
  {
    this->ranges.clear();

    for (const auto& range : newRanges) {
      this->ranges.push_back(range);
    }
  }

  llvm::StringRef Tracker::getName() const
  {
    return name;
  }

  llvm::ArrayRef<Range> Tracker::getRanges() const
  {
    return ranges;
  }

  Range Tracker::getRangeOfDimension(unsigned int dimensionIndex) const
  {
    assert(dimensionIndex < ranges.size());
    return *(ranges.begin() + dimensionIndex);
  }
}
