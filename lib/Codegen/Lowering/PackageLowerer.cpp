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

    // Lower the inner classes.
    for (const auto& innerClassNode : package.getInnerClasses()) {
      lower(*innerClassNode->cast<ast::Class>());
    }
  }
}
