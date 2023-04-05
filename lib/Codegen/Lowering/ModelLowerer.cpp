#include "marco/Codegen/Lowering/ModelLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  ModelLowerer::ModelLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  void ModelLowerer::declare(const ast::Model& model)
  {
    mlir::Location location = loc(model.getLocation());

    // Create the model operation.
    auto modelOp = builder().create<ModelOp>(location, model.getName());

    mlir::OpBuilder::InsertionGuard guard(builder());
    builder().setInsertionPointToStart(modelOp.bodyBlock());

    // Declare the variables.
    declareClassVariables(model);

    // Declare the inner classes.
    for (const auto& innerClassNode : model.getInnerClasses()) {
      declare(*innerClassNode->cast<ast::Class>());
    }
  }

  void ModelLowerer::lower(const ast::Model& model)
  {
    mlir::OpBuilder::InsertionGuard guard(builder());
    Lowerer::VariablesScope varScope(getVariablesSymbolTable());

    // Get the operation.
    auto modelOp = mlir::cast<ModelOp>(getClass(model));
    builder().setInsertionPointToEnd(modelOp.bodyBlock());

    // Map the variables.
    insertVariable(
        "time",
        Reference::time(builder(), builder().getUnknownLoc()));

    for (VariableOp variableOp : modelOp.getVariables()) {
      insertVariable(
          variableOp.getSymName(),
          Reference::variable(
              builder(), variableOp->getLoc(),
              variableOp.getSymName(),
              variableOp.getVariableType().unwrap()));
    }

    // Create the binding equations.
    for (const auto& variableNode : model.getVariables()) {
      const ast::Member* variable = variableNode->cast<ast::Member>();

      if (variable->hasModification()) {
        if (const auto* modification = variable->getModification();
            modification->hasExpression()) {
          createBindingEquation(*variable, *modification->getExpression());
        }
      }
    }

    // Create the 'start' values.
    for (const auto& variableNode : model.getVariables()) {
      const ast::Member* variable = variableNode->cast<ast::Member>();

      if (variable->hasStartExpression()) {
        lowerStartAttribute(
            *variable,
            *variable->getStartExpression(),
            variable->getFixedProperty(),
            variable->getEachProperty());
      }
    }

    // Lower the body.
    lowerClassBody(model);

    // Lower the inner classes.
    for (const auto& innerClassNode : model.getInnerClasses()) {
      lower(*innerClassNode->cast<ast::Class>());
    }
  }
}
