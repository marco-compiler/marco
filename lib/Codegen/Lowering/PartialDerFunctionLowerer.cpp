#include "marco/Codegen/Lowering/PartialDerFunctionLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  PartialDerFunctionLowerer::PartialDerFunctionLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  void PartialDerFunctionLowerer::declare(
      const ast::PartialDerFunction& function)
  {
    mlir::Location location = loc(function.getLocation());

    llvm::StringRef derivedFunctionName =
        function.getDerivedFunction()->cast<ast::ReferenceAccess>()->getName();

    llvm::SmallVector<mlir::Attribute, 3> independentVariables;

    for (const auto& independentVariable :
         function.getIndependentVariables()) {
      auto independentVariableName =
          independentVariable->cast<ast::ReferenceAccess>()->getName();

      independentVariables.push_back(
          builder().getStringAttr(independentVariableName));
    }

    builder().create<DerFunctionOp>(
        location, function.getName(), derivedFunctionName,
        builder().getArrayAttr(independentVariables));
  }

  void PartialDerFunctionLowerer::declareVariables(
      const ast::PartialDerFunction& function)
  {
    // Nothing to do.
  }

  void PartialDerFunctionLowerer::lower(
      const ast::PartialDerFunction& function)
  {
    // Nothing to do.
  }
}
