#include "marco/Codegen/Lowering/BaseModelica/PackageLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering::bmodelica {
PackageLowerer::PackageLowerer(BridgeInterface *bridge) : Lowerer(bridge) {}

void PackageLowerer::declare(const ast::bmodelica::Package &package) {
  mlir::Location location = loc(package.getLocation());

  // Create the package operation.
  auto packageOp = builder().create<PackageOp>(location, package.getName());

  mlir::OpBuilder::InsertionGuard guard(builder());
  builder().setInsertionPointToStart(packageOp.bodyBlock());

  // Declare the inner classes.
  for (const auto &innerClassNode : package.getInnerClasses()) {
    declare(*innerClassNode->cast<ast::bmodelica::Class>());
  }
}

bool PackageLowerer::declareVariables(const ast::bmodelica::Package &package) {
  mlir::OpBuilder::InsertionGuard guard(builder());
  LookupScopeGuard lookupScopeGuard(&getContext());

  // Get the operation.
  auto packageOp = mlir::cast<PackageOp>(getClass(package));
  pushLookupScope(packageOp);
  builder().setInsertionPointToEnd(packageOp.bodyBlock());

  // Declare the variables.
  for (const auto &variable : package.getVariables()) {
    if (!declare(*variable->cast<ast::bmodelica::Member>())) {
      return false;
    }
  }

  // Declare the variables of inner classes.
  for (const auto &innerClassNode : package.getInnerClasses()) {
    if (!declareVariables(*innerClassNode->cast<ast::bmodelica::Class>())) {
      return false;
    }
  }

  return true;
}

bool PackageLowerer::lower(const ast::bmodelica::Package &package) {
  mlir::OpBuilder::InsertionGuard guard(builder());

  VariablesSymbolTable::VariablesScope varScope(getVariablesSymbolTable());
  LookupScopeGuard lookupScopeGuard(&getContext());

  // Get the operation.
  auto packageOp = mlir::cast<PackageOp>(getClass(package));
  pushLookupScope(packageOp);
  builder().setInsertionPointToEnd(packageOp.bodyBlock());

  // Map the variables.
  insertVariable("time", Reference::time(builder(), builder().getUnknownLoc()));

  for (VariableOp variableOp : packageOp.getVariables()) {
    insertVariable(variableOp.getSymName(),
                   Reference::variable(builder(), variableOp->getLoc(),
                                       variableOp.getSymName(),
                                       variableOp.getVariableType().unwrap()));
  }

  // Lower the body.
  if (!lowerClassBody(package)) {
    return false;
  }

  // Create the algorithms.
  llvm::SmallVector<const ast::bmodelica::Algorithm *> initialAlgorithms;
  llvm::SmallVector<const ast::bmodelica::Algorithm *> algorithms;

  for (const auto &algorithm : package.getAlgorithms()) {
    if (algorithm->cast<ast::bmodelica::Algorithm>()->isInitial()) {
      initialAlgorithms.push_back(algorithm->cast<ast::bmodelica::Algorithm>());
    } else {
      algorithms.push_back(algorithm->cast<ast::bmodelica::Algorithm>());
    }
  }

  if (!initialAlgorithms.empty()) {
    auto initialOp = builder().create<InitialOp>(loc(package.getLocation()));

    mlir::OpBuilder::InsertionGuard guard(builder());
    builder().createBlock(&initialOp.getBodyRegion());
    builder().setInsertionPointToStart(initialOp.getBody());

    for (const auto &algorithm : initialAlgorithms) {
      if (!lower(*algorithm)) {
        return false;
      }
    }
  }

  if (!algorithms.empty()) {
    auto dynamicOp = builder().create<DynamicOp>(loc(package.getLocation()));

    mlir::OpBuilder::InsertionGuard guard(builder());
    builder().createBlock(&dynamicOp.getBodyRegion());
    builder().setInsertionPointToStart(dynamicOp.getBody());

    for (const auto &algorithm : algorithms) {
      if (!lower(*algorithm)) {
        return false;
      }
    }
  }

  // Lower the inner classes.
  for (const auto &innerClassNode : package.getInnerClasses()) {
    if (!lower(*innerClassNode->cast<ast::bmodelica::Class>())) {
      return false;
    }
  }

  return true;
}
} // namespace marco::codegen::lowering::bmodelica
