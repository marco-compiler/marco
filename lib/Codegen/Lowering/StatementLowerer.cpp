#include "marco/Codegen/Lowering/StatementLowerer.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  StatementLowerer::StatementLowerer(LoweringContext* context, BridgeInterface* bridge)
      : Lowerer(context, bridge)
  {
  }

  void StatementLowerer::operator()(const ast::AssignmentStatement& statement)
  {
    const auto* destinations = statement.getDestinations();
    auto values = lower(*statement.getExpression());

    assert(destinations->isa<Tuple>());
    const auto* destinationsTuple = destinations->get<Tuple>();
    assert(values.size() == destinationsTuple->size() && "Unequal number of destinations and results");

    for (const auto& [ dest, value ] : llvm::zip(*destinationsTuple, values)) {
      auto destination = lower(*dest)[0];
      destination.set(*value);
    }
  }

  void StatementLowerer::operator()(const ast::IfStatement& statement)
  {
    // Each conditional blocks creates an "if" operation, but we need to keep
    // track of the first one in order to restore the insertion point right
    // after that when we have finished lowering all the blocks.
    mlir::Operation* firstOp = nullptr;

    for (size_t i = 0, e = statement.size(); i < e; ++i) {
      Lowerer::SymbolScope varScope(symbolTable());

      const auto& conditionalBlock = statement[i];
      auto condition = lower(*conditionalBlock.getCondition())[0];

      // The last conditional block can be at most an originally equivalent
      // "else" block, and thus doesn't need a lowered "else" block.
      bool elseBlock = i < e - 1;

      auto location = loc(statement.getLocation());
      auto ifOp = builder().create<IfOp>(location, *condition, elseBlock);

      if (firstOp == nullptr) {
        firstOp = ifOp;
      }

      // Populate the "then" block
      builder().setInsertionPointToStart(ifOp.thenBlock());

      for (const auto& bodyStatement : conditionalBlock) {
        lower(*bodyStatement);
      }

      if (i > 0) {
        builder().setInsertionPointAfter(ifOp);
      }

      // The next conditional blocks will be placed as new If operations
      // nested inside the "else" block.
      if (elseBlock) {
        builder().setInsertionPointToStart(ifOp.elseBlock());
      }
    }

    builder().setInsertionPointAfter(firstOp);
  }

  void StatementLowerer::operator()(const ast::ForStatement& statement)
  {
    Lowerer::SymbolScope varScope(symbolTable());
    auto location = loc(statement.getLocation());

    const auto& induction = statement.getInduction();

    mlir::Value inductionVar = builder().create<AllocaOp>(
        location, ArrayType::get(builder().getContext(), builder().getIndexType(), llvm::None), llvm::None);

    mlir::Value lowerBound = *lower(*induction->getBegin())[0];
    lowerBound = builder().create<CastOp>(lowerBound.getLoc(), builder().getIndexType(), lowerBound);

    builder().create<StoreOp>(location, lowerBound, inductionVar, llvm::None);

    auto forOp = builder().create<ForOp>(location, llvm::None);
    mlir::OpBuilder::InsertionGuard guard(builder());

    assert(forOp.getConditionRegion().getBlocks().empty());
    mlir::Block* conditionBlock = builder().createBlock(&forOp.getConditionRegion());

    assert(forOp.getBodyRegion().getBlocks().empty());
    mlir::Block* bodyBlock = builder().createBlock(&forOp.getBodyRegion());

    assert(forOp.getStepRegion().getBlocks().empty());
    mlir::Block* stepBlock = builder().createBlock(&forOp.getStepRegion());

    {
      // Check the loop condition
      builder().setInsertionPointToStart(conditionBlock);
      mlir::Value inductionValue = builder().create<LoadOp>(location, inductionVar);

      mlir::Value upperBound = *lower(*induction->getEnd())[0];
      upperBound = builder().create<CastOp>(lowerBound.getLoc(), builder().getIndexType(), upperBound);

      mlir::Value condition = builder().create<LteOp>(location, BooleanType::get(builder().getContext()), inductionValue, upperBound);
      builder().create<ConditionOp>(location, condition, llvm::None);
    }

    {
      // Body
      Lowerer::SymbolScope scope(symbolTable());

      builder().setInsertionPointToStart(bodyBlock);
      mlir::Value inductionValue = builder().create<LoadOp>(location, inductionVar);
      symbolTable().insert(induction->getName(), Reference::ssa(&builder(), inductionValue));

      for (const auto& stmnt : statement) {
        lower(*stmnt);
      }

      if (bodyBlock->empty() || !bodyBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        builder().setInsertionPointToEnd(bodyBlock);
        builder().create<YieldOp>(location, llvm::None);
      }
    }

    {
      // Step
      builder().setInsertionPointToStart(stepBlock);
      mlir::Value inductionValue = builder().create<LoadOp>(location, inductionVar);

      mlir::Value step = builder().create<ConstantOp>(location, builder().getIndexAttr(1));
      mlir::Value incremented = builder().create<AddOp>(location, builder().getIndexType(), inductionValue, step);
      builder().create<StoreOp>(location, incremented, inductionVar, llvm::None);
      builder().create<YieldOp>(location, llvm::None);
    }
  }

  void StatementLowerer::operator()(const ast::WhileStatement& statement)
  {
    auto location = loc(statement.getLocation());

    // Create the operation
    auto whileOp = builder().create<WhileOp>(location);
    mlir::OpBuilder::InsertionGuard guard(builder());

    {
      // Condition
      Lowerer::SymbolScope scope(symbolTable());
      assert(whileOp.getConditionRegion().empty());
      mlir::Block* conditionBlock = builder().createBlock(&whileOp.getConditionRegion());
      builder().setInsertionPointToStart(conditionBlock);
      const auto* condition = statement.getCondition();
      builder().create<ConditionOp>(loc(condition->getLocation()), *lower(*condition)[0]);
    }

    {
      // Body
      Lowerer::SymbolScope scope(symbolTable());
      assert(whileOp.getBodyRegion().empty());
      mlir::Block* bodyBlock = builder().createBlock(&whileOp.getBodyRegion());
      builder().setInsertionPointToStart(bodyBlock);

      for (const auto& stmnt : statement) {
        lower(*stmnt);
      }

      if (auto& body = whileOp.getBodyRegion().back(); body.empty() || !body.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        builder().setInsertionPointToEnd(&body);
        builder().create<YieldOp>(location, llvm::None);
      }
    }
  }

  void StatementLowerer::operator()(const ast::WhenStatement& statement)
  {
    llvm_unreachable("When statement is not implemented yet");
  }

  void StatementLowerer::operator()(const ast::BreakStatement& statement)
  {
    auto location = loc(statement.getLocation());
    builder().create<BreakOp>(location);
  }

  void StatementLowerer::operator()(const ast::ReturnStatement& statement)
  {
    auto location = loc(statement.getLocation());
    builder().create<ReturnOp>(location);
  }
}
