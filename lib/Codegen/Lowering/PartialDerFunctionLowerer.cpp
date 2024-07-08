#include "marco/Codegen/Lowering/PartialDerFunctionLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

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

    std::string derivedFunctionName =
        function.getDerivedFunction()
            ->cast<ast::ComponentReference>()->getName();

    llvm::SmallVector<mlir::Attribute, 3> independentVariables;

    for (const auto& independentVariable :
         function.getIndependentVariables()) {
      auto independentVariableName =
          independentVariable->cast<ast::ComponentReference>()->getName();

      independentVariables.push_back(
          builder().getStringAttr(independentVariableName));
    }

    builder().create<DerFunctionOp>(
        location, function.getName(),
        mlir::SymbolRefAttr::get(builder().getContext(), derivedFunctionName),
        builder().getArrayAttr(independentVariables));
  }

  bool PartialDerFunctionLowerer::declareVariables(
      const ast::PartialDerFunction& function)
  {
    // Nothing to do.
    return true;
  }

  bool PartialDerFunctionLowerer::lower(
      const ast::PartialDerFunction& function)
  {
    // Nothing to do.
    return true;
  }
}
