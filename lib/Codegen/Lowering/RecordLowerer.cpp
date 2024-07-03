#include "marco/Codegen/Lowering/RecordLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering
{
  RecordLowerer::RecordLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  void RecordLowerer::declare(const ast::Record& record)
  {
    mlir::Location location = loc(record.getLocation());

    // Create the record operation.
    auto recordOp = builder().create<RecordOp>(location, record.getName());

    mlir::OpBuilder::InsertionGuard guard(builder());
    mlir::Block* bodyBlock = builder().createBlock(&recordOp.getBodyRegion());
    builder().setInsertionPointToStart(bodyBlock);

    // Declare the inner classes.
    for (const auto& innerClassNode : record.getInnerClasses()) {
      declare(*innerClassNode->cast<ast::Class>());
    }
  }

  bool RecordLowerer::declareVariables(const ast::Record& record)
  {
    mlir::OpBuilder::InsertionGuard guard(builder());
    LookupScopeGuard lookupScopeGuard(&getContext());

    // Get the operation.
    auto recordOp = mlir::cast<RecordOp>(getClass(record));
    pushLookupScope(recordOp);
    builder().setInsertionPointToEnd(recordOp.getBody());

    // Declare the variables.
    for (const auto& variable : record.getVariables()) {
      if (!declare(*variable->cast<ast::Member>())) {
        return false;
      }
    }

    // Declare the variables of inner classes.
    for (const auto& innerClassNode : record.getInnerClasses()) {
      if (!declareVariables(*innerClassNode->cast<ast::Class>())) {
        return false;
      }
    }

    return true;
  }

  bool RecordLowerer::lower(const ast::Record& record)
  {
    mlir::OpBuilder::InsertionGuard guard(builder());

    Lowerer::VariablesScope varScope(getVariablesSymbolTable());
    LookupScopeGuard lookupScopeGuard(&getContext());

    // Get the operation.
    auto recordOp = mlir::cast<RecordOp>(getClass(record));
    pushLookupScope(recordOp);
    builder().setInsertionPointToEnd(recordOp.getBody());

    // Map the variables.
    insertVariable(
        "time",
        Reference::time(builder(), builder().getUnknownLoc()));

    for (VariableOp variableOp : recordOp.getVariables()) {
      insertVariable(
          variableOp.getSymName(),
          Reference::variable(
              builder(), variableOp->getLoc(),
              variableOp.getSymName(),
              variableOp.getVariableType().unwrap()));
    }

    // Lower the body.
    if (!lowerClassBody(record)) {
      return false;
    }

    // Create the algorithms.
    llvm::SmallVector<const ast::Algorithm*> initialAlgorithms;
    llvm::SmallVector<const ast::Algorithm*> algorithms;

    for (const auto& algorithm : record.getAlgorithms()) {
      if (algorithm->cast<ast::Algorithm>()->isInitial()) {
        initialAlgorithms.push_back(algorithm->cast<ast::Algorithm>());
      } else {
        algorithms.push_back(algorithm->cast<ast::Algorithm>());
      }
    }

    if (!initialAlgorithms.empty()) {
      auto initialOp =
          builder().create<InitialOp>(loc(record.getLocation()));

      mlir::OpBuilder::InsertionGuard guard(builder());
      builder().createBlock(&initialOp.getBodyRegion());
      builder().setInsertionPointToStart(initialOp.getBody());

      for (const auto& algorithm : initialAlgorithms) {
        if (!lower(*algorithm)) {
          return false;
        }
      }
    }

    if (!algorithms.empty()) {
      auto dynamicOp =
          builder().create<DynamicOp>(loc(record.getLocation()));

      mlir::OpBuilder::InsertionGuard guard(builder());
      builder().createBlock(&dynamicOp.getBodyRegion());
      builder().setInsertionPointToStart(dynamicOp.getBody());

      for (const auto& algorithm : algorithms) {
        if (!lower(*algorithm)) {
          return false;
        }
      }
    }

    // Lower the inner classes.
    for (const auto& innerClassNode : record.getInnerClasses()) {
      if (!lower(*innerClassNode->cast<ast::Class>())) {
        return false;
      }
    }

    return true;
  }
}
