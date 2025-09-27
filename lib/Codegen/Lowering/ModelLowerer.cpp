#include "marco/Codegen/Lowering/ModelLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering {
ModelLowerer::ModelLowerer(BridgeInterface *bridge) : Lowerer(bridge) {}

void ModelLowerer::declare(const ast::bmodelica::Model &model) {
  mlir::Location location = loc(model.getLocation());

  // Create the model operation.
  auto modelOp = builder().create<ModelOp>(location, model.getName());

  mlir::OpBuilder::InsertionGuard guard(builder());
  mlir::Block *bodyBlock = builder().createBlock(&modelOp.getBodyRegion());
  builder().setInsertionPointToStart(bodyBlock);

  // Declare the inner classes.
  for (const auto &innerClassNode : model.getInnerClasses()) {
    declare(*innerClassNode->cast<ast::bmodelica::Class>());
  }
}

bool ModelLowerer::declareVariables(const ast::bmodelica::Model &model) {
  mlir::OpBuilder::InsertionGuard guard(builder());
  LookupScopeGuard lookupScopeGuard(&getContext());

  // Get the operation.
  auto modelOp = mlir::cast<ModelOp>(getClass(model));
  pushLookupScope(modelOp);
  builder().setInsertionPointToEnd(modelOp.getBody());

  // Declare the variables.
  for (const auto &variable : model.getVariables()) {
    if (!declare(*variable->cast<ast::bmodelica::Member>())) {
      return false;
    }
  }

  // Declare the variables of inner classes.
  for (const auto &innerClassNode : model.getInnerClasses()) {
    if (!declareVariables(*innerClassNode->cast<ast::bmodelica::Class>())) {
      return false;
    }
  }

  return true;
}

bool ModelLowerer::lower(const ast::bmodelica::Model &model) {
  mlir::OpBuilder::InsertionGuard guard(builder());

  VariablesSymbolTable::VariablesScope varScope(getVariablesSymbolTable());
  LookupScopeGuard lookupScopeGuard(&getContext());

  // Get the operation.
  auto modelOp = mlir::cast<ModelOp>(getClass(model));
  pushLookupScope(modelOp);
  builder().setInsertionPointToEnd(modelOp.getBody());

  // Map the variables.
  insertVariable("time", Reference::time(builder(), builder().getUnknownLoc()));

  for (VariableOp variableOp : modelOp.getVariables()) {
    insertVariable(variableOp.getSymName(),
                   Reference::variable(builder(), variableOp->getLoc(),
                                       variableOp.getSymName(),
                                       variableOp.getVariableType().unwrap()));
  }

  // Create the binding equations.
  for (const auto &variableNode : model.getVariables()) {
    const ast::bmodelica::Member *variable =
        variableNode->cast<ast::bmodelica::Member>();

    if (variable->hasModification()) {
      if (const auto *modification = variable->getModification();
          modification->hasExpression()) {
        if (!createBindingEquation(*variable, *modification->getExpression())) {
          return false;
        }
      }
    }
  }

  // Lower the attributes of the variables.
  for (const auto &variableNode : model.getVariables()) {
    const ast::bmodelica::Member *variable =
        variableNode->cast<ast::bmodelica::Member>();
    if (!lowerVariableAttributes(modelOp, *variable)) {
      return false;
    }
  }

  // Lower the body.
  if (!lowerClassBody(model)) {
    return false;
  }

  // Create the algorithms.
  llvm::SmallVector<const ast::bmodelica::Algorithm *> initialAlgorithms;
  llvm::SmallVector<const ast::bmodelica::Algorithm *> algorithms;

  for (const auto &algorithm : model.getAlgorithms()) {
    if (algorithm->cast<ast::bmodelica::Algorithm>()->isInitial()) {
      initialAlgorithms.push_back(algorithm->cast<ast::bmodelica::Algorithm>());
    } else {
      algorithms.push_back(algorithm->cast<ast::bmodelica::Algorithm>());
    }
  }

  if (!initialAlgorithms.empty()) {
    auto initialOp = builder().create<InitialOp>(loc(model.getLocation()));

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
    auto dynamicOp = builder().create<DynamicOp>(loc(model.getLocation()));

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
  for (const auto &innerClassNode : model.getInnerClasses()) {
    if (!lower(*innerClassNode->cast<ast::bmodelica::Class>())) {
      return false;
    }
  }

  return true;
}

bool ModelLowerer::lowerVariableAttributes(
    ModelOp modelOp, const ast::bmodelica::Member &variable) {
  if (!variable.hasModification()) {
    return true;
  }

  const ast::bmodelica::Modification *modification = variable.getModification();

  if (!modification->hasClassModification()) {
    return true;
  }

  const ast::bmodelica::ClassModification *classModification =
      modification->getClassModification();

  if (classModification) {
    auto variableOp = mlir::dyn_cast<VariableOp>(
        resolveSymbolName<VariableOp>(variable.getName(), modelOp));

    assert(variableOp != nullptr && "Variable not found");
    llvm::SmallVector<VariableOp> components;
    components.push_back(variableOp);

    if (!lowerVariableAttributes(modelOp, components, *classModification)) {
      std::string errorString = "Invalid fixed property for variable " +
                                variable.getName().str() + ".";
      mlir::emitError(loc(variable.getLocation())) << errorString;
      return false;
    }
  }

  return true;
}

bool ModelLowerer::lowerVariableAttributes(
    ModelOp modelOp, llvm::SmallVectorImpl<VariableOp> &components,
    const ast::bmodelica::ClassModification &classModification) {
  assert(!components.empty());

  if (classModification.hasStartExpression()) {
    llvm::SmallVector<mlir::FlatSymbolRefAttr> nestedRefs;

    for (size_t i = 1, e = components.size(); i < e; ++i) {
      nestedRefs.push_back(
          mlir::FlatSymbolRefAttr::get(components[i].getSymNameAttr()));
    }

    std::optional<bool> fixedProperty = classModification.getFixedProperty();
    if (!fixedProperty) {
      return false;
    }

    if (!lowerStartAttribute(mlir::SymbolRefAttr::get(
                                 components[0].getSymNameAttr(), nestedRefs),
                             *classModification.getStartExpression(),
                             *fixedProperty,
                             classModification.getEachProperty())) {
      return false;
    }
  }

  VariableOp lastVariableOp = components.back();
  VariableType variableType = lastVariableOp.getVariableType();
  mlir::Type elementType = variableType.getElementType();

  if (auto recordType = mlir::dyn_cast<RecordType>(elementType)) {
    auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();

    auto recordOp = mlir::cast<RecordOp>(
        recordType.getRecordOp(getSymbolTable(), moduleOp));

    assert(recordOp != nullptr && "Record not found");

    for (VariableOp recordComponent : recordOp.getVariables()) {
      components.push_back(recordComponent);

      for (const auto &argumentNode : classModification.getArguments()) {
        const auto *argument = argumentNode->cast<ast::bmodelica::Argument>();

        const auto *elementModification =
            argument->dyn_cast<ast::bmodelica::ElementModification>();

        if (!elementModification) {
          continue;
        }

        if (elementModification->getName() != recordComponent.getSymName()) {
          continue;
        }

        if (!elementModification->hasModification()) {
          continue;
        }

        const ast::bmodelica::Modification *modification =
            elementModification->getModification();

        if (!modification->hasClassModification()) {
          continue;
        }

        const ast::bmodelica::ClassModification *innerClassModification =
            modification->getClassModification();

        if (!innerClassModification) {
          continue;
        }

        if (!lowerVariableAttributes(modelOp, components,
                                     *innerClassModification)) {
          return false;
        }
      }

      components.pop_back();
    }
  }

  return true;
}
} // namespace marco::codegen::lowering
