#include "marco/AST/Modelica/ComponentDeclaration.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void ComponentDeclaration::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(subscripts, callback);
  walk(modification, callback);
  walk(condition, callback);
  description.walkChildren(callback);
}

void ComponentDeclaration::addJSONProperties(llvm::json::Object &obj) const {
  obj["name"] = name;

  if (!subscripts.empty()) {
    obj["subscripts"] = modelica::toJSON(subscripts);
  }

  if (modification) {
    obj["modification"] = modelica::toJSON(modification.get());
  }

  if (condition) {
    obj["condition"] = modelica::toJSON(condition.get());
  }

  description.addToJSON(obj);
}

llvm::StringRef ComponentDeclaration::getName() const { return name; }

void ComponentDeclaration::setName(std::string newName) {
  name = std::move(newName);
}

ChildrenRef<Expression> ComponentDeclaration::getSubscripts() const {
  return subscripts;
}

void ComponentDeclaration::addSubscript(std::unique_ptr<Expression> node) {
  subscripts.emplace_back(std::move(node));
}

Modification *ComponentDeclaration::getModification() const {
  return modification.get();
}

void ComponentDeclaration::setModification(std::unique_ptr<Modification> node) {
  modification = std::move(node);
}

Expression *ComponentDeclaration::getCondition() const {
  return condition.get();
}

void ComponentDeclaration::setCondition(std::unique_ptr<Expression> node) {
  condition = std::move(node);
}

Description &ComponentDeclaration::getDescription() { return description; }

const Description &ComponentDeclaration::getDescription() const {
  return description;
}
} // namespace marco::ast::modelica
