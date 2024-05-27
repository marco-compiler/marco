#include "marco/Dialect/BaseModelica/Transforms/ExplicitInitialEquationsInsertion.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_EXPLICITINITIALEQUATIONSINSERTIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

namespace
{
  class ExplicitInitialEquationsInsertionPass
      : public mlir::bmodelica::impl::ExplicitInitialEquationsInsertionPassBase<
            ExplicitInitialEquationsInsertionPass>
  {
    public:
      using ExplicitInitialEquationsInsertionPassBase<
          ExplicitInitialEquationsInsertionPass>
          ::ExplicitInitialEquationsInsertionPassBase;

      void runOnOperation() override;

      void cloneEquationsAsInitialEquations(
          mlir::OpBuilder& builder, ModelOp modelOp);

      void createInitialEquationsFromStartOps(
          mlir::OpBuilder& builder, ModelOp modelOp);
  };
}

void ExplicitInitialEquationsInsertionPass::runOnOperation()
{
  ModelOp modelOp = getOperation();
  mlir::OpBuilder builder(modelOp);
  cloneEquationsAsInitialEquations(builder, modelOp);
  createInitialEquationsFromStartOps(builder, modelOp);
}

void ExplicitInitialEquationsInsertionPass::cloneEquationsAsInitialEquations(
    mlir::OpBuilder& builder, ModelOp modelOp)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  for (DynamicOp dynamicOp : modelOp.getOps<DynamicOp>()) {
    builder.setInsertionPoint(dynamicOp);

    auto initialOp = builder.create<InitialOp>(modelOp.getLoc());
    builder.createBlock(&initialOp.getBodyRegion());
    builder.setInsertionPointToStart(initialOp.getBody());

    for (auto& childOp : dynamicOp.getOps()) {
      if (mlir::isa<EquationInstanceInterface>(childOp)) {
        builder.clone(childOp);
      }
    }
  }
}

void ExplicitInitialEquationsInsertionPass::createInitialEquationsFromStartOps(
    mlir::OpBuilder& builder,
    ModelOp modelOp)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::SymbolTableCollection symbolTableCollection;
  llvm::SmallVector<StartOp> startOps;

  for (StartOp startOp : modelOp.getOps<StartOp>()) {
    startOps.push_back(startOp);
  }

  for (StartOp startOp : startOps) {
    if (!startOp.getFixed()) {
      continue;
    }

    builder.setInsertionPointAfter(startOp);

    // Create the equation template.
    mlir::Location loc = startOp.getLoc();

    auto variable = symbolTableCollection.lookupSymbolIn<VariableOp>(
        getOperation(), startOp.getVariableAttr());

    VariableType variableType = variable.getVariableType();
    int64_t expressionRank = 0;

    auto yieldOp =  mlir::cast<YieldOp>(
        startOp.getBodyRegion().back().getTerminator());

    mlir::Value expressionValue = yieldOp.getValues()[0];

    if (auto expressionShapedType =
            expressionValue.getType().dyn_cast<mlir::ShapedType>()) {
      expressionRank = expressionShapedType.getRank();
    }

    int64_t variableRank = variableType.getRank();
    assert(variableRank >= expressionRank);

    auto templateOp = builder.create<EquationTemplateOp>(startOp.getLoc());
    mlir::Block* templateBody = templateOp.createBody(variableRank);
    builder.setInsertionPointToStart(templateBody);

    auto inductions = templateOp.getInductionVariables();
    assert(startOp.getVariable().getNestedReferences().empty());

    mlir::Value lhs = builder.create<VariableGetOp>(
        startOp.getLoc(), variableType.unwrap(),
        startOp.getVariable().getRootReference());

    if (!inductions.empty()) {
      lhs = builder.create<TensorExtractOp>(lhs.getLoc(), lhs, inductions);
    }

    // Clone the operations.
    mlir::IRMapping mapping;

    for (auto& op : startOp.getOps()) {
      if (!mlir::isa<YieldOp>(op)) {
        builder.clone(op, mapping);
      }
    }

    // Right-hand side.
    mlir::Value rhs = mapping.lookup(expressionValue);
    builder.setInsertionPointToEnd(templateBody);

    if (expressionRank != 0) {
      rhs = builder.create<TensorExtractOp>(
          rhs.getLoc(), rhs, inductions.take_back(expressionRank));
    }

    // Create the assignment.
    mlir::Value lhsTuple = builder.create<EquationSideOp>(loc, lhs);
    mlir::Value rhsTuple = builder.create<EquationSideOp>(loc, rhs);
    builder.create<EquationSidesOp>(loc, lhsTuple, rhsTuple);

    // Create the equation instance.
    builder.setInsertionPointAfter(templateOp);

    auto initialOp = builder.create<InitialOp>(modelOp.getLoc());
    builder.createBlock(&initialOp.getBodyRegion());
    builder.setInsertionPointToStart(initialOp.getBody());

    if (variableType.isScalar()) {
      builder.create<EquationInstanceOp>(loc, templateOp);
    } else {
      IndexSet indices = variable.getIndices();

      for (const MultidimensionalRange& range : llvm::make_range(
               indices.rangesBegin(), indices.rangesEnd())) {
        auto instanceOp = builder.create<EquationInstanceOp>(loc, templateOp);

        instanceOp.setIndicesAttr(
            MultidimensionalRangeAttr::get(builder.getContext(), range));
      }
    }
  }
}

namespace mlir::bmodelica
{
  std::unique_ptr<mlir::Pass> createExplicitInitialEquationsInsertionPass()
  {
    return std::make_unique<ExplicitInitialEquationsInsertionPass>();
  }
}
