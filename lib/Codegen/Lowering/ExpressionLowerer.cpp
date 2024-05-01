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

  Results ExpressionLowerer::lower(const ast::Expression& expression)
  {
    if (auto array = expression.dyn_cast<ast::ArrayGenerator>()) {
      return lower(*array);
    }

    if (auto call = expression.dyn_cast<ast::Call>()) {
      return lower(*call);
    }

    if (auto constant = expression.dyn_cast<ast::Constant>()) {
      return lower(*constant);
    }

    if (auto operation = expression.dyn_cast<ast::Operation>()) {
      return lower(*operation);
    }

    if (auto componentReference =
            expression.dyn_cast<ast::ComponentReference>()) {
      return lower(*componentReference);
    }

    if (auto tuple = expression.dyn_cast<ast::Tuple>()) {
      return lower(*tuple);
    }

    if (auto subscript = expression.dyn_cast<ast::Subscript>()) {
      return lower(*subscript);
    }

    llvm_unreachable("Unknown expression type");
    return {};
  }
}
