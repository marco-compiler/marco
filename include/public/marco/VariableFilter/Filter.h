#ifndef MARCO_VARIABLEFILTER_FILTER_H
#define MARCO_VARIABLEFILTER_FILTER_H

#include "marco/VariableFilter/Range.h"
#include "llvm/ADT/ArrayRef.h"

namespace marco::vf
{
  class Filter
  {
    public:
      Filter(bool visibility, llvm::ArrayRef<Range> ranges);

      static Filter visibleScalar();
      static Filter visibleArray(llvm::ArrayRef<long> shape);

      bool isVisible() const;
      llvm::ArrayRef<Range> getRanges() const;

    private:
      bool visibility;
      llvm::SmallVector<Range, 3> ranges;
  };
}

#endif // MARCO_VARIABLEFILTER_FILTER_H
