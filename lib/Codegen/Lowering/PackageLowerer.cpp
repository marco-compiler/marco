#include "marco/Codegen/Lowering/PackageLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering
{
  PackageLowerer::PackageLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  void PackageLowerer::declare(const ast::Package& package)
  {
    mlir::Location location = loc(package.getLocation());

    // Create the package operation.
    auto packageOp = builder().create<PackageOp>(location, package.getName());

    mlir::OpBuilder::InsertionGuard guard(builder());
    builder().setInsertionPointToStart(packageOp.bodyBlock());

    // Declare the inner classes.
    for (const auto& innerClassNode : package.getInnerClasses()) {
      declare(*innerClassNode->cast<ast::Class>());
    }
  }

  void PackageLowerer::declareVariables(const ast::Package& package)
  {
    mlir::OpBuilder::InsertionGuard guard(builder());
    LookupScopeGuard lookupScopeGuard(&getContext());

    // Get the operation.
    auto packageOp = mlir::cast<PackageOp>(getClass(package));
    pushLookupScope(packageOp);
    builder().setInsertionPointToEnd(packageOp.bodyBlock());

    // Declare the variables.
    for (const auto& variable : package.getVariables()) {
      declare(*variable->cast<ast::Member>());
    }

    // Declare the variables of inner classes.
    for (const auto& innerClassNode : package.getInnerClasses()) {
      declareVariables(*innerClassNode->cast<ast::Class>());
    }
  }

  void PackageLowerer::lower(const ast::Package& package)
  {
    mlir::OpBuilder::InsertionGuard guard(builder());

    Lowerer::VariablesScope varScope(getVariablesSymbolTable());
    LookupScopeGuard lookupScopeGuard(&getContext());

    // Get the operation.
    auto packageOp = mlir::cast<PackageOp>(getClass(package));
    pushLookupScope(packageOp);
    builder().setInsertionPointToEnd(packageOp.bodyBlock());

    // Map the variables.
    insertVariable(
        "time",
        Reference::time(builder(), builder().getUnknownLoc()));

    for (VariableOp variableOp : packageOp.getVariables()) {
      insertVariable(
          variableOp.getSymName(),
          Reference::variable(
              builder(), variableOp->getLoc(),
              variableOp.getSymName(),
              variableOp.getVariableType().unwrap()));
    }

    // Lower the body.
    lowerClassBody(package);

    // Create the algorithms.
    llvm::SmallVector<const ast::Algorithm*> initialAlgorithms;
    llvm::SmallVector<const ast::Algorithm*> algorithms;

    for (const auto& algorithm : package.getAlgorithms()) {
      if (algorithm->cast<ast::Algorithm>()->isInitial()) {
        initialAlgorithms.push_back(algorithm->cast<ast::Algorithm>());
      } else {
        algorithms.push_back(algorithm->cast<ast::Algorithm>());
      }
    }

    if (!initialAlgorithms.empty()) {
      auto initialOp =
          builder().create<InitialOp>(loc(package.getLocation()));

      mlir::OpBuilder::InsertionGuard guard(builder());
      builder().createBlock(&initialOp.getBodyRegion());
      builder().setInsertionPointToStart(initialOp.getBody());

      for (const auto& algorithm : initialAlgorithms) {
        lower(*algorithm);
      }
    }

    if (!algorithms.empty()) {
      auto dynamicOp =
          builder().create<DynamicOp>(loc(package.getLocation()));

      mlir::OpBuilder::InsertionGuard guard(builder());
      builder().createBlock(&dynamicOp.getBodyRegion());
      builder().setInsertionPointToStart(dynamicOp.getBody());

      for (const auto& algorithm : algorithms) {
        lower(*algorithm);
      }
    }

    // Lower the inner classes.
    for (const auto& innerClassNode : package.getInnerClasses()) {
      lower(*innerClassNode->cast<ast::Class>());
    }
  }
}
