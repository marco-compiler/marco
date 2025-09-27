#include "marco/Codegen/Lowering/AssignmentStatementLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering {
AssignmentStatementLowerer::AssignmentStatementLowerer(BridgeInterface *bridge)
    : Lowerer(bridge) {}

bool AssignmentStatementLowerer::lower(
    const ast::bmodelica::AssignmentStatement &statement) {
  mlir::Location statementLoc = loc(statement.getLocation());
  const auto *destinations = statement.getDestinations();

  mlir::Location valuesLoc = loc(statement.getExpression()->getLocation());
  auto values = lower(*statement.getExpression());
  if (!values) {
    return false;
  }

  assert(values->size() == destinations->size() &&
         "Unequal number of destinations and results");

  for (size_t i = 0, e = values->size(); i < e; ++i) {
    const auto *destinationRef =
        destinations->getExpression(i)
            ->cast<ast::bmodelica::ComponentReference>();

    if (!lowerAssignmentToComponentReference(statementLoc, *destinationRef,
                                             (*values)[i].get(valuesLoc))) {
      return false;
    }
  }

  return true;
}

bool AssignmentStatementLowerer::lowerAssignmentToComponentReference(
    mlir::Location assignmentLoc,
    const ast::bmodelica::ComponentReference &destination, mlir::Value value) {
  size_t pathLength = destination.getPathLength();

  llvm::SmallVector<mlir::Attribute> path;
  llvm::SmallVector<mlir::Value> subscripts;
  llvm::SmallVector<int64_t> subscriptsAmounts;

  for (size_t pathIndex = 0; pathIndex < pathLength; ++pathIndex) {
    const ast::bmodelica::ComponentReferenceEntry *refEntry =
        destination.getElement(pathIndex);

    path.push_back(mlir::FlatSymbolRefAttr::get(builder().getContext(),
                                                refEntry->getName()));

    size_t numOfSubscripts = refEntry->getNumOfSubscripts();
    subscriptsAmounts.push_back(static_cast<int64_t>(numOfSubscripts));

    for (size_t subscriptIndex = 0; subscriptIndex < numOfSubscripts;
         ++subscriptIndex) {
      mlir::Location subscriptLoc =
          loc(refEntry->getSubscript(subscriptIndex)->getLocation());

      std::optional<Results> loweredSubscript =
          lower(*refEntry->getSubscript(subscriptIndex));
      if (!loweredSubscript) {
        return false;
      }
      mlir::Value subscriptValue = (*loweredSubscript)[0].get(subscriptLoc);

      subscripts.push_back(subscriptValue);
    }
  }

  if (path.size() == 1) {
    std::optional<Reference> variableRef = lookupVariable(
        mlir::cast<mlir::FlatSymbolRefAttr>(path.front()).getValue());

    if (!variableRef) {
      emitIdentifierError(
          IdentifierError::IdentifierType::VARIABLE,
          mlir::cast<mlir::FlatSymbolRefAttr>(path.front()).getValue(),
          getVariablesSymbolTable().getVariables(true),
          destination.getLocation());

      return false;
    }

    variableRef->set(assignmentLoc, subscripts, value);
  } else {
    builder().create<VariableComponentSetOp>(
        assignmentLoc, builder().getArrayAttr(path), subscripts,
        builder().getI64ArrayAttr(subscriptsAmounts), value);
  }

  return true;
}
} // namespace marco::codegen::lowering
