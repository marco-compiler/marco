#ifndef MARCO_MODELING_LOCALMATCHINGSOLUTIONSMCIM_H
#define MARCO_MODELING_LOCALMATCHINGSOLUTIONSMCIM_H

#include "marco/Modeling/LocalMatchingSolutionsImpl.h"
#include "llvm/ADT/SmallVector.h"

namespace marco::modeling::internal {
/// Compute the local matching solutions starting from an incidence matrix.
/// Differently from the VAF case, the computation is done in an eager way.
class MCIMSolutions : public LocalMatchingSolutions::ImplInterface {
public:
  MCIMSolutions(const MCIM &obj);

  MCIM &operator[](size_t index) override;

  size_t size() const override;

private:
  void compute(const MCIM &obj);

  llvm::SmallVector<MCIM, 3> solutions;
};
} // namespace marco::modeling::internal

#endif // MARCO_MODELING_LOCALMATCHINGSOLUTIONSMCIM_H
