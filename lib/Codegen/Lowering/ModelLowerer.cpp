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

  bool ModelLowerer::declareVariables(const ast::Model& model)
  {
    mlir::OpBuilder::InsertionGuard guard(builder());
    LookupScopeGuard lookupScopeGuard(&getContext());

    // Get the operation.
    auto modelOp = mlir::cast<ModelOp>(getClass(model));
    pushLookupScope(modelOp);
    builder().setInsertionPointToEnd(modelOp.getBody());

    // Declare the variables.
    for (const auto& variable : model.getVariables()) {
      const bool outcome = declare(*variable->cast<ast::Member>());
      if (!outcome) {
        return false;
      }
    }

    // Declare the variables of inner classes.
    for (const auto& innerClassNode : model.getInnerClasses()) {
      const bool outcome = declareVariables(*innerClassNode->cast<ast::Class>());
      if (!outcome) {
        return false;
      }
    }

    return true;
  }

  bool ModelLowerer::lower(const ast::Model& model)
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
          const bool outcome = 
              createBindingEquation(*variable, *modification->getExpression());
          if (!outcome) {
            return false;
          }
        }
      }
    }

    // Lower the attributes of the variables.
    for (const auto& variableNode : model.getVariables()) {
      const ast::Member* variable = variableNode->cast<ast::Member>();
      const bool outcome = lowerVariableAttributes(modelOp, *variable);
      if (!outcome) {
        return false;
      }
    }

    // Lower the body.
    bool outcome = lowerClassBody(model);
    if (!outcome) {
      return false;
    }

    // Create the algorithms.
    llvm::SmallVector<const ast::Algorithm*> initialAlgorithms;
    llvm::SmallVector<const ast::Algorithm*> algorithms;

    for (const auto& algorithm : model.getAlgorithms()) {
      if (algorithm->cast<ast::Algorithm>()->isInitial()) {
        initialAlgorithms.push_back(algorithm->cast<ast::Algorithm>());
      } else {
        algorithms.push_back(algorithm->cast<ast::Algorithm>());
      }
    }

    if (!initialAlgorithms.empty()) {
      auto initialOp =
          builder().create<InitialOp>(loc(model.getLocation()));

      mlir::OpBuilder::InsertionGuard guard(builder());
      builder().createBlock(&initialOp.getBodyRegion());
      builder().setInsertionPointToStart(initialOp.getBody());

      for (const auto& algorithm : initialAlgorithms) {
        const bool outcome = lower(*algorithm);
        if (!outcome) {
          return false;
        }
      }
    }

    if (!algorithms.empty()) {
      auto dynamicOp =
          builder().create<DynamicOp>(loc(model.getLocation()));

      mlir::OpBuilder::InsertionGuard guard(builder());
      builder().createBlock(&dynamicOp.getBodyRegion());
      builder().setInsertionPointToStart(dynamicOp.getBody());

      for (const auto& algorithm : algorithms) {
        const bool outcome = lower(*algorithm);
        if (!outcome) {
          return false;
        }
      }
    }

    // Lower the inner classes.
    for (const auto& innerClassNode : model.getInnerClasses()) {
      outcome = lower(*innerClassNode->cast<ast::Class>());
      if (!outcome) {
        return false;
      }
    }

    return true;
  }

  bool ModelLowerer::lowerVariableAttributes(
      ModelOp modelOp, const ast::Member& variable)
  {
    if (!variable.hasModification()) {
      return true;
    }

    const ast::Modification* modification = variable.getModification();

    if (!modification->hasClassModification()) {
      return true;
    }

    const ast::ClassModification* classModification =
        modification->getClassModification();

    if (classModification) {
      auto variableOp = mlir::dyn_cast<VariableOp>(
          resolveSymbolName<VariableOp>(variable.getName(), modelOp));

      assert(variableOp != nullptr && "Variable not found");
      llvm::SmallVector<VariableOp> components;
      components.push_back(variableOp);

      const bool outcome = 
          lowerVariableAttributes(modelOp, components, *classModification);
      if (!outcome) {
        const marco::SourceRange location = variable.getLocation();
        const std::string errorString = "Error in AST to MLIR conversion. Invalid fixed property for variable " + 
                                        std::string(variable.getName()) + ".";
        mlir::DiagnosticEngine& diag = getContext().getDiagEngine();
        diag.emit(loc(location), mlir::DiagnosticSeverity::Error) << errorString;
        return false;
      }
    }

    return true;
  }

  bool ModelLowerer::lowerVariableAttributes(
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

      const std::optional<bool> fixedProperty = classModification.getFixedProperty();
      if (!fixedProperty) {
        return false;
      }

      const bool outcome = lowerStartAttribute(
          mlir::SymbolRefAttr::get(components[0].getSymNameAttr(), nestedRefs),
          *classModification.getStartExpression(),
          fixedProperty.value(),
          classModification.getEachProperty());
      if (!outcome) {
        return false;
      }
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

          const bool outcome =
              lowerVariableAttributes(modelOp, components,
                                  *innerClassModification);
          if (!outcome) {
            return false;
          }
        }

        components.pop_back();
      }
    }

    return true;
  }
}
