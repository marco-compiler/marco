#include "marco/Modeling/AccessFunctionRotoTranslation.h"

using namespace ::marco::modeling;

namespace marco::modeling
{
  AccessFunctionRotoTranslation::AccessFunctionRotoTranslation(
      mlir::AffineMap affineMap)
      : AccessFunction(
          AccessFunction::Kind::RotoTranslation,
          affineMap)
  {
    assert(canBeBuilt(affineMap));
  }

  AccessFunctionRotoTranslation
      ::~AccessFunctionRotoTranslation() = default;

  bool AccessFunctionRotoTranslation::canBeBuilt(
      mlir::AffineMap affineMap)
  {
    if (affineMap.getNumResults() == 0) {
      return false;
    }

    for (mlir::AffineExpr result : affineMap.getResults()) {
      if (result.isa<mlir::AffineDimExpr>() ||
          result.isa<mlir::AffineConstantExpr>()) {
        continue;
      }

      if (auto binaryOp = result.dyn_cast<mlir::AffineBinaryOpExpr>()) {
        if (binaryOp.getKind() != mlir::AffineExprKind::Add) {
          return false;
        }

        if (!((binaryOp.getLHS().isa<mlir::AffineDimExpr>() &&
            binaryOp.getRHS().isa<mlir::AffineConstantExpr>()) ||
              (binaryOp.getLHS().isa<mlir::AffineConstantExpr>() &&
               binaryOp.getRHS().isa<mlir::AffineDimExpr>()))) {
          return false;
        }

        return true;
      }

      return false;
    }

    return true;
  }

  std::unique_ptr<AccessFunction> AccessFunctionRotoTranslation::clone() const
  {
    return std::make_unique<AccessFunctionRotoTranslation>(*this);
  }

  bool AccessFunctionRotoTranslation::isInvertible() const
  {
    if (getAffineMap().getNumDims() != getAffineMap().getNumResults()) {
      return false;
    }

    llvm::BitVector usedDimensions(getAffineMap().getNumDims(), false);

    auto processDim = [&](llvm::Optional<unsigned int> dimension) -> bool {
      if (!dimension) {
        return false;
      }

      if (*dimension > getAffineMap().getNumDims()) {
        return false;
      }

      usedDimensions[*dimension] = true;
      return true;
    };

    for (mlir::AffineExpr result : getAffineMap().getResults()) {
      if (!processDim(getInductionVariableIndex(result))) {
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
      mlir::AffineExpr result = getAffineMap().getResult(i);

      llvm::Optional<unsigned int> inductionVar =
          getInductionVariableIndex(result);

      if (!inductionVar) {
        return nullptr;
      }

      int64_t offset = getOffset(result);

      remapped.push_back(
          mlir::getAffineDimExpr(i, getAffineMap().getContext()) -
          mlir::getAffineConstantExpr(offset, getAffineMap().getContext()));

      positionsMap[*inductionVar] = i;
    }

    llvm::SmallVector<mlir::AffineExpr, 3> reordered;

    for (const auto& position : positionsMap) {
      reordered.push_back(remapped[position]);
    }

    return AccessFunction::build(mlir::AffineMap::get(
        getAffineMap().getNumDims(), 0,
        reordered, getAffineMap().getContext()));
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

    mlir::AffineMap map = getAffineMapWithAtLeastNDimensions(indices.rank());
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
    for (unsigned int i = 0, e = getAffineMap().getNumResults(); i < e; ++i) {
      auto inductionIndex =
          getInductionVariableIndex(getAffineMap().getResult(i));

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
    unsigned int numOfDims = getAffineMap().getNumDims();

    for (size_t i = 0, e = usages.size(); i < e; ++i) {
      usages[i] = 0;
    }

    usages.resize(numOfDims, 0);

    for (mlir::AffineExpr expression : getAffineMap().getResults()) {
      if (auto inductionVariable = getInductionVariableIndex(expression)) {
        ++usages[*inductionVariable];
      }
    }
  }

  llvm::Optional<unsigned int>
  AccessFunctionRotoTranslation::getInductionVariableIndex(
      unsigned int expressionIndex) const
  {
    return getInductionVariableIndex(
        getAffineMap().getResult(expressionIndex));
  }

  llvm::Optional<unsigned int>
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

    return llvm::None;
  }

  int64_t AccessFunctionRotoTranslation::getOffset(
      unsigned int expressionIndex) const
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
