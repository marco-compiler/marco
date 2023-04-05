#include "marco/Codegen/Lowering/ForStatementLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  ForStatementLowerer::ForStatementLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  void ForStatementLowerer::lower(const ast::ForStatement& statement)
  {
    Lowerer::VariablesScope varScope(getVariablesSymbolTable());
    mlir::Location location = loc(statement.getLocation());

    const auto& induction = statement.getInduction();

    mlir::Value inductionVar = builder().create<AllocaOp>(
        location,
        ArrayType::get(llvm::None, builder().getIndexType()), llvm::None);

    mlir::Location lowerBoundLoc = loc(induction->getBegin()->getLocation());

    mlir::Value lowerBound =
        lower(*induction->getBegin())[0].get(lowerBoundLoc);

    lowerBound = builder().create<CastOp>(
        lowerBound.getLoc(), builder().getIndexType(), lowerBound);

    builder().create<StoreOp>(location, lowerBound, inductionVar, llvm::None);

    auto forOp = builder().create<ForOp>(location, llvm::None);
    mlir::OpBuilder::InsertionGuard guard(builder());

    assert(forOp.getConditionRegion().getBlocks().empty());

    mlir::Block* conditionBlock =
        builder().createBlock(&forOp.getConditionRegion());

    assert(forOp.getBodyRegion().getBlocks().empty());
    mlir::Block* bodyBlock = builder().createBlock(&forOp.getBodyRegion());

    assert(forOp.getStepRegion().getBlocks().empty());
    mlir::Block* stepBlock = builder().createBlock(&forOp.getStepRegion());

    {
      // Check the loop condition.
      builder().setInsertionPointToStart(conditionBlock);

      mlir::Value inductionValue =
          builder().create<LoadOp>(location, inductionVar);

      mlir::Location upperBoundLoc = loc(induction->getEnd()->getLocation());

      mlir::Value upperBound =
          lower(*induction->getEnd())[0].get(upperBoundLoc);

      upperBound = builder().create<CastOp>(
          lowerBound.getLoc(), builder().getIndexType(), upperBound);

      mlir::Value condition = builder().create<LteOp>(
          location, BooleanType::get(
                        builder().getContext()), inductionValue, upperBound);

      builder().create<ConditionOp>(location, condition, llvm::None);
    }

    {
      // Body.
      Lowerer::VariablesScope scope(getVariablesSymbolTable());

      builder().setInsertionPointToStart(bodyBlock);

      mlir::Value inductionValue =
          builder().create<LoadOp>(location, inductionVar);

      getVariablesSymbolTable().insert(
          induction->getName(), Reference::ssa(builder(), inductionValue));

      for (size_t i = 0, e = statement.size(); i < e; ++i) {
        lower(*statement[i]);
      }

      if (bodyBlock->empty() ||
          !bodyBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        builder().setInsertionPointToEnd(bodyBlock);
        builder().create<YieldOp>(location, llvm::None);
      }
    }

    {
      // Step.
      builder().setInsertionPointToStart(stepBlock);

      mlir::Value inductionValue =
          builder().create<LoadOp>(location, inductionVar);

      mlir::Value step = builder().create<ConstantOp>(
          location, builder().getIndexAttr(1));

      mlir::Value incremented = builder().create<AddOp>(
          location, builder().getIndexType(), inductionValue, step);

      builder().create<StoreOp>(
          location, incremented, inductionVar, llvm::None);

      builder().create<YieldOp>(location, llvm::None);
    }
  }
}
