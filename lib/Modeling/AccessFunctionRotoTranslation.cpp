#include "marco/Modeling/AccessFunctionRotoTranslation.h"
#include "marco/Modeling/DimensionAccessAdd.h"
#include "marco/Modeling/DimensionAccessConstant.h"
#include "marco/Modeling/DimensionAccessDimension.h"

using namespace ::marco::modeling;

namespace marco::modeling
{
  AccessFunctionRotoTranslation::AccessFunctionRotoTranslation(
      mlir::MLIRContext* context,
      unsigned int numOfDimensions,
      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results)
      : AccessFunction(
          AccessFunction::Kind::RotoTranslation,
          context, numOfDimensions, results)
  {
    assert(canBeBuilt(numOfDimensions, results));
  }

  AccessFunctionRotoTranslation::AccessFunctionRotoTranslation(
      mlir::AffineMap affineMap)
      : AccessFunctionRotoTranslation(
          affineMap.getContext(),
          affineMap.getNumDims(),
          convertAffineExpressions(affineMap.getResults()))
  {
  }

  AccessFunctionRotoTranslation
      ::~AccessFunctionRotoTranslation() = default;

  bool AccessFunctionRotoTranslation::canBeBuilt(
      unsigned int numOfDimensions,
      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results)
  {
    if (numOfDimensions == 0) {
      return false;
    }

    if (results.empty()) {
      return false;
    }

    return llvm::all_of(
        results, [](const std::unique_ptr<DimensionAccess>& result) {
          if (result->isa<DimensionAccessDimension>() ||
              result->isa<DimensionAccessConstant>()) {
            return true;
          }

          if (auto addExpr = result->dyn_cast<DimensionAccessAdd>()) {
            if (addExpr->getFirst().isa<DimensionAccessDimension>() &&
                addExpr->getSecond().isa<DimensionAccessConstant>()) {
              return true;
            }

            if (addExpr->getFirst().isa<DimensionAccessConstant>() &&
                addExpr->getSecond().isa<DimensionAccessDimension>()) {
              return true;
            }
          }

          return false;
        });
  }

  bool AccessFunctionRotoTranslation::canBeBuilt(mlir::AffineMap affineMap)
  {
    return AccessFunctionRotoTranslation::canBeBuilt(
        affineMap.getNumDims(),
        AccessFunction::convertAffineExpressions(affineMap.getResults()));
  }

  std::unique_ptr<AccessFunction> AccessFunctionRotoTranslation::clone() const
  {
    return std::make_unique<AccessFunctionRotoTranslation>(*this);
  }

  bool AccessFunctionRotoTranslation::isInvertible() const
  {
    if (getNumOfDims() != getNumOfResults()) {
      return false;
    }

    llvm::BitVector usedDimensions(getNumOfDims(), false);

    auto processDim = [&](std::optional<uint64_t> dimension) -> bool {
      if (!dimension) {
        return false;
      }

      if (*dimension > getNumOfDims()) {
        return false;
      }

      usedDimensions[*dimension] = true;
      return true;
    };

    for (size_t i = 0, e = getNumOfResults(); i < e; ++i) {
      if (!processDim(getInductionVariableIndex(i))) {
        return false;
      }
    }

    return usedDimensions.all();
  }

  std::unique_ptr<AccessFunction>
  AccessFunctionRotoTranslation::inverse() const
  {
    if (!isInvertible()) {
      return nullptr;
    }

    llvm::SmallVector<mlir::AffineExpr, 3> remapped;

    llvm::SmallVector<size_t, 3> positionsMap;
    positionsMap.resize(getNumOfResults());

    for (size_t i = 0, e = getNumOfResults(); i < e; ++i) {
      std::optional<uint64_t> inductionVar = getInductionVariableIndex(i);

      if (!inductionVar) {
        return nullptr;
      }

      int64_t offset = getOffset(i);

      remapped.push_back(
          mlir::getAffineDimExpr(i, getContext()) -
          mlir::getAffineConstantExpr(offset, getContext()));

      positionsMap[*inductionVar] = i;
    }

    llvm::SmallVector<mlir::AffineExpr, 3> reordered;

    for (const auto& position : positionsMap) {
      reordered.push_back(remapped[position]);
    }

    return AccessFunction::build(mlir::AffineMap::get(
        getNumOfDims(), 0, reordered, getContext()));
  }

  MultidimensionalRange AccessFunctionRotoTranslation::map(
      const MultidimensionalRange& indices) const
  {
    llvm::SmallVector<int64_t, 3> lowerBounds;
    llvm::SmallVector<int64_t, 3> upperBounds;
    llvm::SmallVector<Range, 3> ranges;

    for (size_t i = 0, e = indices.rank(); i < e; ++i) {
      lowerBounds.push_back(indices[i].getBegin());
      upperBounds.push_back(indices[i].getEnd());
    }

    llvm::SmallVector<mlir::AffineExpr> mapExpressions;
    DimensionAccess::FakeDimensionsMap fakeDimensionsMap;

    for (const auto& result : getResults()) {
      mapExpressions.push_back(
          result->getAffineExpr(getNumOfDims(), fakeDimensionsMap));

      assert(fakeDimensionsMap.empty());
    }

    auto map = mlir::AffineMap::get(
        std::max(getNumOfDims(), static_cast<size_t>(indices.rank())), 0,
        mapExpressions, getContext());

    auto mappedLowerBounds = map.compose(lowerBounds);
    auto mappedUpperBounds = map.compose(upperBounds);

    for (size_t i = 0, e = getNumOfResults(); i < e; ++i) {
      int64_t mappedLowerBound = mappedLowerBounds[i];
      int64_t mappedUpperBound = mappedUpperBounds[i];

      if (mappedLowerBound == mappedUpperBound) {
        // Constant access.
        ++mappedUpperBound;
      }

      ranges.push_back(Range(mappedLowerBound, mappedUpperBound));
    }

    assert(ranges.size() == getNumOfResults());
    return {ranges};
  }

  IndexSet AccessFunctionRotoTranslation::map(const IndexSet& indices) const
  {
    IndexSet result;

    for (const MultidimensionalRange& range :
         llvm::make_range(indices.rangesBegin(), indices.rangesEnd())) {
      result += map(range);
    }

    return result;
  }

  IndexSet AccessFunctionRotoTranslation::inverseMap(
      const IndexSet& accessedIndices,
      const IndexSet& parentIndices) const
  {
    if (auto inverseAccessFunction = inverse()) {
      if (!accessedIndices.empty() &&
          !parentIndices.empty() &&
          accessedIndices.rank() == parentIndices.rank()) {
        IndexSet mapped = inverseAccessFunction->map(accessedIndices);
        assert(map(mapped).contains(accessedIndices));
        return mapped;
      }
    }

    // If the access function is not invertible, then not all the iteration
    // variables are used. This loss of information doesn't allow to
    // reconstruct the equation ranges that leads to the dependency loop. Thus,
    // we need to iterate on all the original equation points and determine
    // which of them lead to a loop. This is highly expensive but also
    // inevitable, and confined only to very few cases within real scenarios.

    return AccessFunction::inverseMap(accessedIndices, parentIndices);
  }

  bool AccessFunctionRotoTranslation::isIdentityLike() const
  {
    for (size_t i = 0, e = getNumOfResults(); i < e; ++i) {
      auto inductionIndex = getInductionVariableIndex(i);

      if (!inductionIndex) {
        return false;
      }

      if (*inductionIndex != i) {
        return false;
      }
    }

    return true;
  }

  void AccessFunctionRotoTranslation::countVariablesUsages(
      llvm::SmallVectorImpl<size_t>& usages) const
  {
    for (size_t& usage : usages) {
      usage = 0;
    }

    usages.resize(getNumOfDims(), 0);

    for (size_t i = 0, e = getNumOfResults(); i < e; ++i) {
      if (auto inductionVariable = getInductionVariableIndex(i)) {
        ++usages[*inductionVariable];
      }
    }
  }

  std::optional<uint64_t>
  AccessFunctionRotoTranslation::getInductionVariableIndex(
      unsigned int expressionIndex) const
  {
    const DimensionAccess* access = getResults()[expressionIndex].get();

    if (auto dimExpr = access->dyn_cast<DimensionAccessDimension>()) {
      return dimExpr->getDimension();
    }

    if (auto addOp = access->dyn_cast<DimensionAccessAdd>()) {
      if (auto dimExpr =
              addOp->getFirst().dyn_cast<DimensionAccessDimension>()) {
        return dimExpr->getDimension();
      }

      if (auto dimExpr =
              addOp->getSecond().dyn_cast<DimensionAccessDimension>()) {
        return dimExpr->getDimension();
      }
    }

    return std::nullopt;
  }

  int64_t AccessFunctionRotoTranslation::getOffset(
      unsigned int expressionIndex) const
  {
    const DimensionAccess* access = getResults()[expressionIndex].get();

    if (access->isa<DimensionAccessDimension>()) {
      return 0;
    }

    if (auto addOp = access->dyn_cast<DimensionAccessAdd>()) {
      if (auto constantExpr =
              addOp->getFirst().dyn_cast<DimensionAccessConstant>()) {
        return constantExpr->getValue();
      }

      if (auto constantExpr =
              addOp->getSecond().dyn_cast<DimensionAccessConstant>()) {
        return constantExpr->getValue();
      }
    }

    llvm_unreachable("Incompatible access expression");
    return 0;
  }
}
