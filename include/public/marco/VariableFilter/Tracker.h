#ifndef MARCO_VARIABLEFILTER_TRACKER_H
#define MARCO_VARIABLEFILTER_TRACKER_H

#include "marco/VariableFilter/Range.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace marco::vf {
/// Keeps tracks of a single variable, array or derivative that has been
/// specified by command line argument.
class Tracker {
public:
  Tracker();
  Tracker(llvm::StringRef name);
  Tracker(llvm::StringRef name, llvm::ArrayRef<Range> ranges);

  void setRanges(llvm::ArrayRef<Range> ranges);

  llvm::StringRef getName() const;

  llvm::ArrayRef<Range> getRanges() const;

  Range getRangeOfDimension(unsigned int dimensionIndex) const;

private:
  std::string name;
  llvm::SmallVector<Range, 3> ranges;
};
} // namespace marco::vf

#endif // MARCO_VARIABLEFILTER_TRACKER_H
