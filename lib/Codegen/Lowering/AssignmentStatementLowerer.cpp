#include "marco/Codegen/Lowering/AssignmentStatementLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering
{
  AssignmentStatementLowerer::AssignmentStatementLowerer(
      BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  bool AssignmentStatementLowerer::lower(
      const ast::AssignmentStatement& statement)
  {
    mlir::Location statementLoc = loc(statement.getLocation());
    const auto* destinations = statement.getDestinations();

    mlir::Location valuesLoc = loc(statement.getExpression()->getLocation());
    auto values = lower(*statement.getExpression());
    if (!values) {
      return false;
    }

    assert(values->size() == destinations->size() &&
           "Unequal number of destinations and results");

    for (size_t i = 0, e = values->size(); i < e; ++i) {
      const auto* destinationRef = destinations->getExpression(i)
                                       ->cast<ast::ComponentReference>();

      size_t pathLength = destinationRef->getPathLength();

      llvm::SmallVector<mlir::FlatSymbolRefAttr> path;
      llvm::SmallVector<mlir::Value> subscripts;

      for (size_t pathIndex = 0; pathIndex < pathLength; ++pathIndex) {
        const ast::ComponentReferenceEntry* refEntry =
            destinationRef->getElement(pathIndex);

        path.push_back(mlir::FlatSymbolRefAttr::get(
            builder().getContext(), refEntry->getName()));

        size_t numOfSubscripts = refEntry->getNumOfSubscripts();

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
        std::optional<Reference> variableRef = lookupVariable(path.front().getValue());
        if (!variableRef) {
          std::set<std::string> visibleVariables = {};
          getVisibleVariables(visibleVariables);

          emitIdentifierError(IdentifierError::IdentifierType::VARIABLE, path.front().getValue(), 
                              visibleVariables, statement.getExpression()->getLocation());
          return false;
        }

        mlir::Value rhs = (*values)[i].get(valuesLoc);
        variableRef->set(statementLoc, subscripts, rhs);
      } else {
        builder().create<VariableComponentSetOp>(
            statementLoc,
            mlir::SymbolRefAttr::get(
                path.front().getAttr(),
                llvm::ArrayRef(path).drop_front()),
            subscripts,
            (*values)[i].get(valuesLoc));
      }
    }

    return true;
  }
}
