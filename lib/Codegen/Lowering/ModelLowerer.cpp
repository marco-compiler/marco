#include "marco/Codegen/Lowering/ModelLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

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
    mlir::Block* bodyBlock = builder().createBlock(&modelOp.getBodyRegion());
    builder().setInsertionPointToStart(bodyBlock);

    // Declare the inner classes.
    for (const auto& innerClassNode : model.getInnerClasses()) {
      declare(*innerClassNode->cast<ast::Class>());
    }
  }

  void ModelLowerer::declareVariables(const ast::Model& model)
  {
    mlir::OpBuilder::InsertionGuard guard(builder());
    LookupScopeGuard lookupScopeGuard(&getContext());

    // Get the operation.
    auto modelOp = mlir::cast<ModelOp>(getClass(model));
    pushLookupScope(modelOp);
    builder().setInsertionPointToEnd(modelOp.getBody());

    // Declare the variables.
    for (const auto& variable : model.getVariables()) {
      declare(*variable->cast<ast::Member>());
    }

    // Declare the variables of inner classes.
    for (const auto& innerClassNode : model.getInnerClasses()) {
      declareVariables(*innerClassNode->cast<ast::Class>());
    }
  }

  void ModelLowerer::lower(const ast::Model& model)
  {
    mlir::OpBuilder::InsertionGuard guard(builder());

    Lowerer::VariablesScope varScope(getVariablesSymbolTable());
    LookupScopeGuard lookupScopeGuard(&getContext());

    // Get the operation.
    auto modelOp = mlir::cast<ModelOp>(getClass(model));
    pushLookupScope(modelOp);
    builder().setInsertionPointToEnd(modelOp.getBody());

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

    // Lower the attributes of the variables.
    for (const auto& variableNode : model.getVariables()) {
      const ast::Member* variable = variableNode->cast<ast::Member>();
      lowerVariableAttributes(modelOp, *variable);
    }

    // Lower the body.
    lowerClassBody(model);

    // Lower the inner classes.
    for (const auto& innerClassNode : model.getInnerClasses()) {
      lower(*innerClassNode->cast<ast::Class>());
    }
  }

  void ModelLowerer::lowerVariableAttributes(
      ModelOp modelOp, const ast::Member& variable)
  {
    if (!variable.hasModification()) {
      return;
    }

    const ast::Modification* modification = variable.getModification();

    if (!modification->hasClassModification()) {
      return;
    }

    const ast::ClassModification* classModification =
        modification->getClassModification();

    if (classModification) {
      auto variableOp = mlir::dyn_cast<VariableOp>(
          resolveSymbolName<VariableOp>(variable.getName(), modelOp));

      assert(variableOp != nullptr && "Variable not found");
      llvm::SmallVector<VariableOp> components;
      components.push_back(variableOp);

      lowerVariableAttributes(modelOp, components, *classModification);
    }
  }

  void ModelLowerer::lowerVariableAttributes(
      ModelOp modelOp,
      llvm::SmallVectorImpl<VariableOp>& components,
      const ast::ClassModification& classModification)
  {
    assert(!components.empty());

    if (classModification.hasStartExpression()) {
      llvm::SmallVector<mlir::FlatSymbolRefAttr> nestedRefs;

      for (size_t i = 1, e = components.size(); i < e; ++i) {
        nestedRefs.push_back(mlir::FlatSymbolRefAttr::get(
            components[i].getSymNameAttr()));
      }

      lowerStartAttribute(
          mlir::SymbolRefAttr::get(components[0].getSymNameAttr(), nestedRefs),
          *classModification.getStartExpression(),
          classModification.getFixedProperty(),
          classModification.getEachProperty());
    }

    VariableOp lastVariableOp = components.back();
    VariableType variableType = lastVariableOp.getVariableType();
    mlir::Type elementType = variableType.getElementType();

    if (auto recordType = elementType.dyn_cast<RecordType>()) {
      auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();

      auto recordOp = mlir::cast<RecordOp>(
          recordType.getRecordOp(getSymbolTable(), moduleOp));

      assert(recordOp != nullptr && "Record not found");

      for (VariableOp recordComponent : recordOp.getVariables()) {
        components.push_back(recordComponent);

        for (const auto& argumentNode : classModification.getArguments()) {
          const auto* argument = argumentNode->cast<ast::Argument>();

          const auto* elementModification =
              argument->dyn_cast<ast::ElementModification>();

          if (!elementModification) {
            continue;
          }

          if (elementModification->getName() != recordComponent.getSymName()) {
            continue;
          }

          if (!elementModification->hasModification()) {
            continue;
          }

          const ast::Modification* modification =
              elementModification->getModification();

          if (!modification->hasClassModification()) {
            continue;
          }

          const ast::ClassModification* innerClassModification =
              modification->getClassModification();

          if (!innerClassModification) {
            continue;
          }

          lowerVariableAttributes(modelOp, components,
                                  *innerClassModification);
        }

        components.pop_back();
      }
    }
  }
}
