#include "marco/Codegen/Lowering/ClassLowerer.h"
#include "marco/Codegen/Lowering/ModelLowerer.h"
#include "marco/Codegen/Lowering/StandardFunctionLowerer.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  ClassLowerer::ClassLowerer(LoweringContext* context, BridgeInterface* bridge)
      : Lowerer(context, bridge),
        modelLowerer(std::make_unique<ModelLowerer>(context, bridge)),
        recordLowerer(std::make_unique<RecordLowerer>(context, bridge)),
        standardFunctionLowerer(
            std::make_unique<StandardFunctionLowerer>(context, bridge))
  {
    assert(modelLowerer != nullptr);
    assert(recordLowerer != nullptr);
    assert(standardFunctionLowerer != nullptr);
  }

  std::vector<mlir::Operation*> ClassLowerer::operator()(
      const ast::PartialDerFunction& function)
  {
    std::vector<mlir::Operation*> result;

    mlir::OpBuilder::InsertionGuard guard(builder());
    auto location = loc(function.getLocation());

    llvm::StringRef derivedFunctionName =
        function.getDerivedFunction()->get<ReferenceAccess>()->getName();

    llvm::SmallVector<mlir::Attribute, 3> independentVariables;

    for (const auto& independentVariable :
         function.getIndependentVariables()) {
      auto independentVariableName =
          independentVariable->get<ReferenceAccess>()->getName();

      independentVariables.push_back(
          builder().getStringAttr(independentVariableName));
    }

    auto derFunctionOp = builder().create<DerFunctionOp>(
        location, function.getName(), derivedFunctionName,
        builder().getArrayAttr(independentVariables));

    result.push_back(derFunctionOp);
    return result;
  }

  std::vector<mlir::Operation*> ClassLowerer::operator()(
      const ast::StandardFunction& function)
  {
    // inlined functions are already handled in the frontend's InliningPass.
    if(function.shouldBeInlined())
      return {};

    return standardFunctionLowerer->lower(function);
  }

  std::vector<mlir::Operation*> ClassLowerer::operator()(
      const ast::Model& model)
  {
    return modelLowerer->lower(model);
  }

  std::vector<mlir::Operation*> ClassLowerer::operator()(
      const ast::Package& package)
  {
    std::vector<mlir::Operation*> result;

    for (const auto& cls : package.getInnerClasses()) {
      for (const auto& op : lower(*cls)) {
        result.push_back(op);
      }
    }

    return result;
  }

  std::vector<mlir::Operation*> ClassLowerer::operator()(const ast::Record& record)
  {
    return recordLowerer->lower(record);
  }
}
