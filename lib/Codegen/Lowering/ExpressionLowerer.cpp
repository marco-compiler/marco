#include "marco/Codegen/Lowering/ExpressionLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  ExpressionLowerer::ExpressionLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  Results ExpressionLowerer::lower(const ast::Expression& expression)
  {
    if (auto array = expression.dyn_cast<ast::Array>()) {
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

    if (auto referenceAccess = expression.dyn_cast<ast::ReferenceAccess>()) {
      return lower(*referenceAccess);
    }

    if (auto tuple = expression.dyn_cast<ast::Tuple>()) {
      return lower(*tuple);
    }

    llvm_unreachable("Unknown expression type");
    return {};
  }
}
