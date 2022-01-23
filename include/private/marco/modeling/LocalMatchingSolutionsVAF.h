#ifndef MARCO_MODELING_LOCALMATCHINGSOLUTIONSVAF_H
#define MARCO_MODELING_LOCALMATCHINGSOLUTIONSVAF_H

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <marco/modeling/AccessFunction.h>
#include <marco/modeling/MultidimensionalRange.h>
#include <memory>

#include "LocalMatchingSolutionsImpl.h"

namespace marco::modeling::internal
{
  /// Compute the local matching solution starting from a given set of variable access functions (VAF).
  /// The computation is done in a lazy way, that is each result is computed only when requested.
  class VAFSolutions : public LocalMatchingSolutions::ImplInterface
  {
    public:
      VAFSolutions(
          llvm::ArrayRef<AccessFunction> accessFunctions,
          MultidimensionalRange equationRanges,
          MultidimensionalRange variableRanges);

      MCIM& operator[](size_t index) override;

      size_t size() const override;

    private:
      void fetchNext();

      void getInductionVariablesUsage(
          llvm::SmallVectorImpl<size_t>& usages,
          const AccessFunction& accessFunction) const;

      llvm::SmallVector<AccessFunction, 3> accessFunctions;
      MultidimensionalRange equationRanges;
      MultidimensionalRange variableRanges;

      // Total number of possible match matrices
      size_t solutionsAmount;

      // List of the computed match matrices
      llvm::SmallVector<MCIM, 3> matrices;

      size_t currentAccessFunction = 0;
      size_t groupSize;
      llvm::SmallVector<Range, 3> reorderedRanges;
      std::unique_ptr<MultidimensionalRange> range;
      llvm::SmallVector<size_t, 3> ordering;
      std::unique_ptr<MultidimensionalRange::const_iterator> rangeIt;
      std::unique_ptr<MultidimensionalRange::const_iterator> rangeEnd;
  };
}

#endif // MARCO_MODELING_LOCALMATCHINGSOLUTIONSVAF_H
