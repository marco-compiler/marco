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

  void AssignmentStatementLowerer::lower(
      const ast::AssignmentStatement& statement)
  {
    mlir::Location statementLoc = loc(statement.getLocation());
    const auto* destinations = statement.getDestinations();

    mlir::Location valuesLoc = loc(statement.getExpression()->getLocation());
    auto values = lower(*statement.getExpression());

    assert(values.size() == destinations->size() &&
           "Unequal number of destinations and results");

    for (size_t i = 0, e = values.size(); i < e; ++i) {
      const auto* destinationRef = destinations->getExpression(i)
                                       ->cast<ast::ComponentReference>();

      size_t pathLength = destinationRef->getPathLength();

      llvm::SmallVector<mlir::Attribute> path;
      llvm::SmallVector<mlir::Value> subscripts;
      llvm::SmallVector<int64_t> subscriptsAmounts;

      for (size_t pathIndex = 0; pathIndex < pathLength; ++pathIndex) {
        const ast::ComponentReferenceEntry* refEntry =
            destinationRef->getElement(pathIndex);

        path.push_back(mlir::FlatSymbolRefAttr::get(
            builder().getContext(), refEntry->getName()));

        size_t numOfSubscripts = refEntry->getNumOfSubscripts();
        subscriptsAmounts.push_back(static_cast<int64_t>(numOfSubscripts));

        for (size_t subscriptIndex = 0; subscriptIndex < numOfSubscripts;
             ++subscriptIndex) {
          mlir::Location subscriptLoc =
              loc(refEntry->getSubscript(subscriptIndex)->getLocation());

          mlir::Value subscriptValue =
              lower(*refEntry->getSubscript(subscriptIndex))[0]
                  .get(subscriptLoc);

          subscripts.push_back(subscriptValue);
        }
      }

      if (path.size() == 1) {
        Reference variableRef = lookupVariable(
            path.front().cast<mlir::FlatSymbolRefAttr>().getValue());

        mlir::Value rhs = values[i].get(valuesLoc);
        variableRef.set(statementLoc, subscripts, rhs);
      } else {
        builder().create<VariableComponentSetOp>(
            statementLoc,
            builder().getArrayAttr(path),
            subscripts,
            builder().getI64ArrayAttr(subscriptsAmounts),
            values[i].get(valuesLoc));
      }
    }
  }
}
