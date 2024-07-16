#include "marco/Codegen/Lowering/ExpressionLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering
{
  ExpressionLowerer::ExpressionLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  std::optional<Results> ExpressionLowerer::lower(const ast::Expression& expression)
  {
    if (const auto *array = expression.dyn_cast<ast::ArrayGenerator>()) {
      return lower(*array);
    }

    if (const auto *call = expression.dyn_cast<ast::Call>()) {
      return lower(*call);
    }

    if (const auto *constant = expression.dyn_cast<ast::Constant>()) {
      return lower(*constant);
    }

    if (const auto *operation = expression.dyn_cast<ast::Operation>()) {
      return lower(*operation);
    }

    if (const auto *componentReference =
            expression.dyn_cast<ast::ComponentReference>()) {
      return lower(*componentReference);
    }

    if (const auto *tuple = expression.dyn_cast<ast::Tuple>()) {
      return lower(*tuple);
    }

    if (const auto *subscript = expression.dyn_cast<ast::Subscript>()) {
      return lower(*subscript);
    }

    llvm_unreachable("Unknown expression type");
    return {};
  }
} // namespace marco::codegen::lowering
