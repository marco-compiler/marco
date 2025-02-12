#include "marco/Codegen/Lowering/ConstantLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering {
ConstantLowerer::ConstantLowerer(BridgeInterface *bridge) : Lowerer(bridge) {}

std::optional<Results> ConstantLowerer::lower(const ast::Constant &constant) {
  mlir::Location location = loc(constant.getLocation());
  mlir::TypedAttr attribute = constant.visit(*this);
  auto result = builder().create<ConstantOp>(location, attribute);
  return Reference::ssa(builder(), result);
}

mlir::TypedAttr ConstantLowerer::operator()(bool value) {
  return BooleanAttr::get(builder().getContext(), value);
}

mlir::TypedAttr ConstantLowerer::operator()(int64_t value) {
  return IntegerAttr::get(builder().getContext(), value);
}

mlir::TypedAttr ConstantLowerer::operator()(double value) {
  return RealAttr::get(builder().getContext(), value);
}

mlir::TypedAttr ConstantLowerer::operator()(std::string value) {
  llvm_unreachable("Unsupported constant type");
  return nullptr;
}
} // namespace marco::codegen::lowering
