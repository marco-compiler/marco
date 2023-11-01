#include "marco/Modeling/AccessFunctionConstant.h"
#include "marco/Modeling/DimensionAccessConstant.h"
#include "marco/Modeling/DimensionAccessIndices.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco::modeling;

namespace marco::modeling
{
  bool AccessFunctionConstant::canBeBuilt(
      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results)
  {
    if (results.empty()) {
      return false;
    }

    return llvm::all_of(results, [](const auto& result) {
      return result->template dyn_cast<DimensionAccessConstant>() ||
          result->template dyn_cast<DimensionAccessIndices>();
    });
  }

  bool AccessFunctionConstant::canBeBuilt(mlir::AffineMap affineMap)
  {
    return llvm::all_of(
        affineMap.getResults(), [](mlir::AffineExpr expression) {
          return expression.isa<mlir::AffineConstantExpr>();
        });
  }

  AccessFunctionConstant::AccessFunctionConstant(
      mlir::MLIRContext* context,
      uint64_t numOfDimensions,
      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results)
      : AccessFunctionGeneric(
          Kind::Constant, context, numOfDimensions, results,
          DimensionAccess::FakeDimensionsMap())
  {
    assert(canBeBuilt(results));
  }

  AccessFunctionConstant::AccessFunctionConstant(mlir::AffineMap affineMap)
      : AccessFunctionConstant(
            affineMap.getContext(),
            affineMap.getNumDims(),
            convertAffineExpressions(affineMap.getResults()))
  {
  }

  AccessFunctionConstant::~AccessFunctionConstant() = default;

  std::unique_ptr<AccessFunction> AccessFunctionConstant::clone() const
  {
    return std::make_unique<AccessFunctionConstant>(*this);
  }

  IndexSet AccessFunctionConstant::map(const IndexSet& indices) const
  {
    llvm::SmallVector<Point::data_type, 6> dummyCoordinates(getNumOfDims(), 0);
    Point dummyPoint(dummyCoordinates);
    return map(dummyPoint);
  }

  IndexSet AccessFunctionConstant::inverseMap(
      const IndexSet& accessedIndices,
      const IndexSet& parentIndices) const
  {
    return parentIndices;
  }
}
