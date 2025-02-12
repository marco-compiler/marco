#include "marco/Codegen/Lowering/SubscriptLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering {
SubscriptLowerer::SubscriptLowerer(BridgeInterface *bridge) : Lowerer(bridge) {}

std::optional<Results>
SubscriptLowerer::lower(const ast::Subscript &subscript) {
  mlir::Location location = loc(subscript.getLocation());

  if (subscript.isUnbounded()) {
    mlir::Value result = builder().create<UnboundedRangeOp>(location);
    return Reference::ssa(builder(), result);
  }

  auto loweredSubscript = lower(*subscript.getExpression());
  if (!loweredSubscript) {
    return std::nullopt;
  }
  mlir::Value index = (*loweredSubscript)[0].get(location);

  // Indices in Modelica are 1-based, while in the MLIR dialect are
  // 0-based. Thus, we need to shift them by one. In doing so, we also
  // force the result to be of index type.

  mlir::Value one =
      builder().create<ConstantOp>(index.getLoc(), builder().getIndexAttr(-1));

  mlir::Type resultType = builder().getIndexType();

  if (mlir::Type indexType = index.getType(); indexType.isa<RangeType>()) {
    resultType = indexType;
  }

  mlir::Value zeroBasedIndex =
      builder().create<AddOp>(index.getLoc(), resultType, index, one);

  return Reference::ssa(builder(), zeroBasedIndex);
}
} // namespace marco::codegen::lowering
