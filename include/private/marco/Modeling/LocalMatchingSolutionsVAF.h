#ifndef MARCO_MODELING_LOCALMATCHINGSOLUTIONSVAF_H
#define MARCO_MODELING_LOCALMATCHINGSOLUTIONSVAF_H

#include "marco/Modeling/AccessFunction.h"
#include "marco/Modeling/AccessFunctionConstant.h"
#include "marco/Modeling/AccessFunctionRotoTranslation.h"
#include "marco/Modeling/LocalMatchingSolutionsImpl.h"
#include "marco/Modeling/MultidimensionalRange.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace marco::modeling::internal {
/// Compute the local matching solution starting from a given set of variable
/// access functions (VAF). The computation is done in a lazy way, that is
/// each result is computed only when requested.
class VAFSolutions : public LocalMatchingSolutions::ImplInterface {
public:
  class Generator {
  public:
    virtual ~Generator() = default;

    virtual bool hasValue() const = 0;

    virtual MCIM getValue() const = 0;

    virtual void fetchNext() = 0;
  };

  VAFSolutions(llvm::ArrayRef<std::unique_ptr<AccessFunction>> accessFunctions,
               IndexSet equationIndices, IndexSet variableIndices);

  MCIM &operator[](size_t index) override;

  size_t size() const override;

private:
  void initialize();

  void fetchNext();

  bool compareAccessFunctions(const AccessFunction &lhs,
                              const AccessFunction &rhs) const;

  bool compareAccessFunctions(const AccessFunctionRotoTranslation &lhs,
                              const AccessFunction &rhs) const;

  bool compareAccessFunctions(const AccessFunctionRotoTranslation &lhs,
                              const AccessFunctionRotoTranslation &rhs) const;

  size_t getSolutionsAmount(const AccessFunction &accessFunction) const;

  size_t getSolutionsAmount(const AccessFunctionConstant &accessFunction) const;

  size_t
  getSolutionsAmount(const AccessFunctionRotoTranslation &accessFunction) const;

  std::unique_ptr<Generator>
  getGenerator(const AccessFunction &accessFunction) const;

private:
  llvm::SmallVector<std::unique_ptr<AccessFunction>, 3> accessFunctions;
  IndexSet equationIndices;
  IndexSet variableIndices;

  // Total number of possible match matrices.
  size_t solutionsAmount;

  // List of the computed match matrices.
  llvm::SmallVector<MCIM, 3> matrices;

  // The access function being processed.
  size_t currentAccessFunction = 0;

  // The current generator.
  std::unique_ptr<Generator> generator;
};
} // namespace marco::modeling::internal

#endif // MARCO_MODELING_LOCALMATCHINGSOLUTIONSVAF_H
