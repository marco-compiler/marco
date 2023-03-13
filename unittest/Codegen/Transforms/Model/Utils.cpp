#include "mlir/IR/Builders.h"

#include "Utils.h"

using namespace ::mlir::modelica;

namespace marco::codegen::test
{
  ModelOp createModel(mlir::OpBuilder& builder, mlir::TypeRange varTypes)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    llvm::SmallVector<std::string, 3> names;
    llvm::SmallVector<mlir::Location, 3> varLocations;

    for (size_t i = 0; i < varTypes.size(); ++i) {
      names.push_back("var" + std::to_string(i));
      varLocations.push_back(builder.getUnknownLoc());
    }

    auto modelOp = builder.create<ModelOp>(builder.getUnknownLoc(), "Test");
    builder.setInsertionPointToStart(modelOp.bodyBlock());

    for (const auto& [name, type] : llvm::zip(names, varTypes)) {
      auto variableType = VariableType::wrap(type);

      builder.create<VariableOp>(
          builder.getUnknownLoc(), name, variableType);
    }

    return modelOp;
  }

  mlir::modelica::EquationOp createEquation(
      mlir::OpBuilder& builder,
      mlir::modelica::ModelOp model,
      llvm::ArrayRef<std::pair<long, long>> iterationRanges,
      std::function<void(mlir::OpBuilder&, mlir::ValueRange)> bodyFn)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(model.bodyBlock());
    auto loc = builder.getUnknownLoc();

    // Create the iteration ranges
    std::vector<mlir::Value> inductionVariables;

    for (const auto& [begin, end] : iterationRanges) {
      assert(begin <= end);
      auto forEquationOp = builder.create<ForEquationOp>(loc, begin, end, 1);
      inductionVariables.push_back(forEquationOp.induction());
      builder.setInsertionPointToStart(forEquationOp.bodyBlock());
    }

    // Create the equation body
    auto equationOp = builder.create<EquationOp>(loc);

    mlir::Block* equationBodyBlock = builder.createBlock(&equationOp.getBodyRegion());
    builder.setInsertionPointToStart(equationBodyBlock);

    bodyFn(builder, inductionVariables);

    return equationOp;
  }

  void createEquationSides(mlir::OpBuilder& builder, mlir::ValueRange lhs, mlir::ValueRange rhs)
  {
    auto loc = builder.getUnknownLoc();
    auto lhsOp = builder.create<EquationSideOp>(loc, lhs);
    auto rhsOp = builder.create<EquationSideOp>(loc, rhs);
    builder.create<EquationSidesOp>(loc, lhsOp, rhsOp);
  }
}
