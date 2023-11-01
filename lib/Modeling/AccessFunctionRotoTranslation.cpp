#include "marco/Modeling/AccessFunctionRotoTranslation.h"
#include "marco/Modeling/DimensionAccessAdd.h"
#include "marco/Modeling/DimensionAccessConstant.h"
#include "marco/Modeling/DimensionAccessDimension.h"

using namespace ::marco::modeling;

namespace marco::modeling
{
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

  AccessFunctionRotoTranslation::AccessFunctionRotoTranslation(
      mlir::AffineMap affineMap)
      : AccessFunctionAffineMap(Kind::RotoTranslation, affineMap)
  {
    assert(canBeBuilt(affineMap));
  }

  AccessFunctionRotoTranslation::AccessFunctionRotoTranslation(
      mlir::MLIRContext* context,
      uint64_t numOfDimensions,
      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results)
      : AccessFunctionRotoTranslation(
          buildAffineMap(context, numOfDimensions, results))
  {
  }

  AccessFunctionRotoTranslation
      ::~AccessFunctionRotoTranslation() = default;

  std::unique_ptr<AccessFunction> AccessFunctionRotoTranslation::clone() const
  {
    return std::make_unique<AccessFunctionRotoTranslation>(*this);
  }

  bool AccessFunctionRotoTranslation::operator==(
      const AccessFunction& other) const
  {
    if (auto otherCasted = other.dyn_cast<AccessFunctionRotoTranslation>()) {
      return *this == *otherCasted;
    }

    return false;
  }

  bool AccessFunctionRotoTranslation::operator==(
      const AccessFunctionRotoTranslation& other) const
  {
    return getAffineMap() == other.getAffineMap();
  }

  bool AccessFunctionRotoTranslation::operator!=(
      const AccessFunction& other) const
  {
    return !(*this == other);
  }

  bool AccessFunctionRotoTranslation::operator!=(
      const AccessFunctionRotoTranslation& other) const
  {
    return !(*this == other);
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

    mlir::AffineMap map = getAffineMapWithGivenDimensions(indices.rank());
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
      uint64_t expressionIndex) const
  {
    return getInductionVariableIndex(
        getAffineMap().getResult(expressionIndex));
  }

  std::optional<uint64_t>
  AccessFunctionRotoTranslation::getInductionVariableIndex(
      mlir::AffineExpr expression) const
  {
    if (auto dimExpr = expression.dyn_cast<mlir::AffineDimExpr>()) {
      return dimExpr.getPosition();
    }

    if (auto binaryOp = expression.dyn_cast<mlir::AffineBinaryOpExpr>()) {
      if (auto dimExpr = binaryOp.getLHS().dyn_cast<mlir::AffineDimExpr>()) {
        return dimExpr.getPosition();
      }

      if (auto dimExpr = binaryOp.getRHS().dyn_cast<mlir::AffineDimExpr>()) {
        return dimExpr.getPosition();
      }
    }

    return std::nullopt;
  }

  int64_t AccessFunctionRotoTranslation::getOffset(
      uint64_t expressionIndex) const
  {
    return getOffset(getAffineMap().getResult(expressionIndex));
  }

  int64_t AccessFunctionRotoTranslation::getOffset(
      mlir::AffineExpr expression) const
  {
    if (auto dimExpr = expression.dyn_cast<mlir::AffineDimExpr>()) {
      return 0;
    }

    if (auto binaryOp = expression.dyn_cast<mlir::AffineBinaryOpExpr>()) {
      if (auto constExpr =
              binaryOp.getLHS().dyn_cast<mlir::AffineConstantExpr>()) {
        return constExpr.getValue();
      }

      if (auto constExpr =
              binaryOp.getRHS().dyn_cast<mlir::AffineConstantExpr>()) {
        return constExpr.getValue();
      }
    }

    llvm_unreachable("Incompatible access expression");
    return 0;
  }
}
