#include "marco/Codegen/Lowering/AssignmentStatementLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

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

    const auto* destinationsTuple = destinations->cast<ast::Tuple>();

    assert(values.size() == destinationsTuple->size() &&
           "Unequal number of destinations and results");

    for (size_t i = 0, e = values.size(); i < e; ++i) {
      auto destination = lower(*destinationsTuple->getExpression(i))[0];
      destination.set(statementLoc, values[i].get(valuesLoc));
    }
  }
}
