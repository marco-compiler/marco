#include "marco/Codegen/Lowering/ForStatementLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering {
ForStatementLowerer::ForStatementLowerer(BridgeInterface *bridge)
    : Lowerer(bridge) {}

bool ForStatementLowerer::lower(const ast::ForStatement &statement) {
  VariablesSymbolTable::VariablesScope varScope(getVariablesSymbolTable());
  mlir::Location location = loc(statement.getLocation());

  size_t numOfForIndices = statement.getNumOfForIndices();

  for (size_t forIndexCount = 0; forIndexCount < numOfForIndices;
       ++forIndexCount) {
    const ast::ForIndex *forIndex = statement.getForIndex(forIndexCount);
    assert(forIndex->hasExpression());
    const ast::Expression *forIndexExpression = forIndex->getExpression();
    assert(forIndexExpression->isa<ast::Operation>());

    const auto *forIndexRange = forIndexExpression->cast<ast::Operation>();

    assert(forIndexRange->getOperationKind() == ast::OperationKind::range);

    mlir::Value inductionVar = builder().create<AllocaOp>(
        location, ArrayType::get(std::nullopt, builder().getIndexType()),
        std::nullopt);

    const ast::Expression *lowerBoundExp = forIndexRange->getArgument(0);
    mlir::Location lowerBoundLoc = loc(lowerBoundExp->getLocation());
    auto loweredLowerBoundExp = lower(*lowerBoundExp);
    if (!loweredLowerBoundExp) {
      return false;
    }
    mlir::Value lowerBound = (*loweredLowerBoundExp)[0].get(lowerBoundLoc);

    lowerBound = builder().create<CastOp>(lowerBound.getLoc(),
                                          builder().getIndexType(), lowerBound);

    builder().create<StoreOp>(location, lowerBound, inductionVar, std::nullopt);

    auto forOp = builder().create<ForOp>(location, std::nullopt);
    mlir::OpBuilder::InsertionGuard guard(builder());

    assert(forOp.getConditionRegion().getBlocks().empty());

    mlir::Block *conditionBlock =
        builder().createBlock(&forOp.getConditionRegion());

    assert(forOp.getBodyRegion().getBlocks().empty());
    mlir::Block *bodyBlock = builder().createBlock(&forOp.getBodyRegion());

    assert(forOp.getStepRegion().getBlocks().empty());
    mlir::Block *stepBlock = builder().createBlock(&forOp.getStepRegion());

    {
      // Check the loop condition.
      builder().setInsertionPointToStart(conditionBlock);

      mlir::Value inductionValue =
          builder().create<LoadOp>(location, inductionVar);

      const ast::Expression *upperBoundExp = forIndexRange->getArgument(1);
      mlir::Location upperBoundLoc = loc(upperBoundExp->getLocation());
      auto loweredUpperBoundExp = lower(*upperBoundExp);
      if (!loweredUpperBoundExp) {
        return false;
      }
      mlir::Value upperBound = (*loweredUpperBoundExp)[0].get(upperBoundLoc);

      upperBound = builder().create<CastOp>(
          lowerBound.getLoc(), builder().getIndexType(), upperBound);

      mlir::Value condition = builder().create<LteOp>(
          location, BooleanType::get(builder().getContext()), inductionValue,
          upperBound);

      builder().create<ConditionOp>(location, condition, std::nullopt);
    }

    {
      // Body.
      VariablesSymbolTable::VariablesScope scope(getVariablesSymbolTable());

      builder().setInsertionPointToStart(bodyBlock);

      mlir::Value inductionValue =
          builder().create<LoadOp>(location, inductionVar);

      llvm::StringRef name = forIndex->getName();
      getVariablesSymbolTable().insert(
          name, Reference::ssa(builder(), inductionValue));

      for (size_t statementIndex = 0, e = statement.getNumOfStatements();
           statementIndex < e; ++statementIndex) {
        if (!lower(*statement.getStatement(statementIndex))) {
          return false;
        }
      }

      if (bodyBlock->empty() ||
          !bodyBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        builder().setInsertionPointToEnd(bodyBlock);
        builder().create<YieldOp>(location, std::nullopt);
      }
    }

    {
      // Step.
      builder().setInsertionPointToStart(stepBlock);

      mlir::Value inductionValue =
          builder().create<LoadOp>(location, inductionVar);

      mlir::Value step;

      if (forIndexRange->getNumOfArguments() == 3) {
        auto loweredExpression = lower(*forIndexRange->getArgument(2));
        if (!loweredExpression) {
          return false;
        }
        step = (*loweredExpression)[0].get(
            loc(forIndexRange->getArgument(2)->getLocation()));
      } else {
        step =
            builder().create<ConstantOp>(location, builder().getIndexAttr(1));
      }

      mlir::Value incremented = builder().create<AddOp>(
          location, builder().getIndexType(), inductionValue, step);

      builder().create<StoreOp>(location, incremented, inductionVar,
                                std::nullopt);

      builder().create<YieldOp>(location, std::nullopt);
    }
  }

  return true;
}
} // namespace marco::codegen::lowering
