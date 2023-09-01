#include "marco/Codegen/Transforms/ExplicitInitialEquationsInsertion.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_EXPLICITINITIALEQUATIONSINSERTIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  class ExplicitInitialEquationsInsertionPass
      : public mlir::modelica::impl::ExplicitInitialEquationsInsertionPassBase<
            ExplicitInitialEquationsInsertionPass>
  {
    public:
      using ExplicitInitialEquationsInsertionPassBase
          ::ExplicitInitialEquationsInsertionPassBase;

      void runOnOperation() override;

      void cloneEquationAsInitialEquation(
          mlir::OpBuilder& builder,
          EquationInstanceOp equationOp);

      void createInitialEquationsFromStartOp(
          mlir::OpBuilder& builder,
          mlir::SymbolTableCollection& symbolTable,
          StartOp startOp);
  };
}

void ExplicitInitialEquationsInsertionPass::runOnOperation()
{
  ModelOp modelOp = getOperation();

  llvm::SmallVector<EquationInstanceOp> equationOps;
  llvm::SmallVector<StartOp> startOps;

  for (auto& op : modelOp.getOps()) {
    if (auto equationOp = mlir::dyn_cast<EquationInstanceOp>(op)) {
      equationOps.push_back(equationOp);
    } else if (auto startOp = mlir::dyn_cast<StartOp>(op)) {
      startOps.push_back(startOp);
    }
  }

  mlir::OpBuilder builder(modelOp);
  mlir::SymbolTableCollection symbolTableCollection;

  for (EquationInstanceOp equationOp : equationOps) {
    cloneEquationAsInitialEquation(builder, equationOp);
  }

  for (StartOp startOp : startOps) {
    createInitialEquationsFromStartOp(builder, symbolTableCollection, startOp);
  }
}

void ExplicitInitialEquationsInsertionPass::cloneEquationAsInitialEquation(
    mlir::OpBuilder& builder,
    EquationInstanceOp equationOp)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(equationOp);

  auto clonedOp = builder.cloneWithoutRegions(equationOp);
  clonedOp.setInitial(true);
}

void ExplicitInitialEquationsInsertionPass::createInitialEquationsFromStartOp(
    mlir::OpBuilder& builder,
    mlir::SymbolTableCollection& symbolTableCollection,
    StartOp startOp)
{
  if (!startOp.getFixed()) {
    return;
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(startOp);

  // Create the equation template.
  mlir::Location loc = startOp.getLoc();

  auto variable = symbolTableCollection.lookupSymbolIn<VariableOp>(
      getOperation(), startOp.getVariableAttr());

  auto variableType = variable.getVariableType();
  unsigned int expressionRank = 0;

  auto yieldOp =  mlir::cast<YieldOp>(
      startOp.getBodyRegion().back().getTerminator());

  mlir::Value expressionValue = yieldOp.getValues()[0];

  if (auto expressionArrayType =
          expressionValue.getType().dyn_cast<ArrayType>()) {
    expressionRank = expressionArrayType.getRank();
  }

  auto variableRank = variableType.getRank();
  assert(expressionRank == 0 || expressionRank == variableRank);

  auto equationTemplateOp =
      builder.create<EquationTemplateOp>(startOp.getLoc());

  builder.setInsertionPointToStart(
      equationTemplateOp.createBody(variableRank - expressionRank));

  auto inductionVariables = equationTemplateOp.getInductionVariables();

  // Left-hand side.
  mlir::Value lhsValue = builder.create<VariableGetOp>(
      startOp.getLoc(), variableType.unwrap(), startOp.getVariable());

  if (!inductionVariables.empty()) {
    if (inductionVariables.size() ==
        static_cast<size_t>(variableType.getRank())) {
      lhsValue = builder.create<LoadOp>(loc, lhsValue, inductionVariables);
    } else {
      lhsValue = builder.create<SubscriptionOp>(
          loc, lhsValue, inductionVariables);
    }
  }

  // Clone the operations.
  mlir::BlockAndValueMapping mapping;

  for (auto& op : startOp.getOps()) {
    if (!mlir::isa<YieldOp>(op)) {
      builder.clone(op, mapping);
    }
  }

  // Right-hand side.
  mlir::Value rhsValue = mapping.lookup(yieldOp.getValues()[0]);

  // Create the assignment.
  mlir::Value lhsTuple = builder.create<EquationSideOp>(loc, lhsValue);
  mlir::Value rhsTuple = builder.create<EquationSideOp>(loc, rhsValue);
  builder.create<EquationSidesOp>(loc, lhsTuple, rhsTuple);

  // Create the equation instance.
  builder.setInsertionPointAfter(equationTemplateOp);

  if (variableRank == expressionRank) {
    builder.create<EquationInstanceOp>(loc, equationTemplateOp, true);
  } else {
    llvm::SmallVector<Range, 3> ranges;

    for (unsigned int i = 0; i < variableRank - expressionRank; ++i) {
      ranges.push_back(Range(0, variableType.getShape()[i]));
    }

    auto instanceOp =
        builder.create<EquationInstanceOp>(loc, equationTemplateOp, true);

    instanceOp.setIndicesAttr(MultidimensionalRangeAttr::get(
        builder.getContext(), MultidimensionalRange(ranges)));
  }
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createExplicitInitialEquationsInsertionPass()
  {
    return std::make_unique<ExplicitInitialEquationsInsertionPass>();
  }
}
