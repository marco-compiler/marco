#include "marco/Codegen/Lowering/BaseModelica/ExpressionLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering::bmodelica {
ExpressionLowerer::ExpressionLowerer(BridgeInterface *bridge)
    : Lowerer(bridge) {}

std::optional<Results>
ExpressionLowerer::lower(const ast::bmodelica::Expression &expression) {
  if (auto array = expression.dyn_cast<ast::bmodelica::ArrayGenerator>()) {
    return lower(*array);
  }

  if (auto call = expression.dyn_cast<ast::bmodelica::Call>()) {
    return lower(*call);
  }

  if (auto constant = expression.dyn_cast<ast::bmodelica::Constant>()) {
    return lower(*constant);
  }

  if (auto operation = expression.dyn_cast<ast::bmodelica::Operation>()) {
    return lower(*operation);
  }

  if (auto componentReference =
          expression.dyn_cast<ast::bmodelica::ComponentReference>()) {
    return lower(*componentReference);
  }

  if (auto tuple = expression.dyn_cast<ast::bmodelica::Tuple>()) {
    return lower(*tuple);
  }

  if (auto subscript = expression.dyn_cast<ast::bmodelica::Subscript>()) {
    return lower(*subscript);
  }

  llvm_unreachable("Unknown expression type");
  return {};
}
} // namespace marco::codegen::lowering::bmodelica
