#include "marco/Codegen/Lowering/WhileStatementLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering
{
  WhileStatementLowerer::WhileStatementLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  bool WhileStatementLowerer::lower(const ast::WhileStatement& statement)
  {
    mlir::Location location = loc(statement.getLocation());

    // Create the operation.
    auto whileOp = builder().create<WhileOp>(location);
    mlir::OpBuilder::InsertionGuard guard(builder());

    {
      // Condition.
      Lowerer::VariablesScope scope(getVariablesSymbolTable());
      assert(whileOp.getConditionRegion().empty());

      mlir::Block* conditionBlock =
          builder().createBlock(&whileOp.getConditionRegion());

      builder().setInsertionPointToStart(conditionBlock);

      mlir::Location conditionLoc =
          loc(statement.getCondition()->getLocation());

      const auto* condition = statement.getCondition();

      auto optionalLoweredCondition = lower(*condition);
      if (!optionalLoweredCondition) {
        return false;
      }
      builder().create<ConditionOp>(
          loc(condition->getLocation()),
          optionalLoweredCondition.value()[0].get(conditionLoc));
    }

    {
      // Body.
      Lowerer::VariablesScope scope(getVariablesSymbolTable());
      assert(whileOp.getBodyRegion().empty());
      mlir::Block* bodyBlock = builder().createBlock(&whileOp.getBodyRegion());
      builder().setInsertionPointToStart(bodyBlock);

      for (size_t i = 0, e = statement.size(); i < e; ++i) {
        const bool outcome = lower(*statement[i]);
        if (!outcome) {
          return false;
        }
      }

      if (auto& body = whileOp.getBodyRegion().back(); body.empty() ||
          !body.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        builder().setInsertionPointToEnd(&body);
        builder().create<YieldOp>(location, std::nullopt);
      }
    }

    return true;
  }
}
