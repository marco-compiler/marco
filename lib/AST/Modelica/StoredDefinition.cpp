#include "marco/AST/Modelica/StoredDefinition.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void StoredDefinition::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(classes, callback);
}

void StoredDefinition::addJSONProperties(llvm::json::Object &obj) const {
  if (within) {
    obj["within"] = modelica::toJSON(*within);
  }

  obj["classes"] = modelica::toJSON(classes);
}

const std::optional<Name> &StoredDefinition::getWithin() const {
  return within;
}

void StoredDefinition::setWithin(std::optional<Name> name) {
  within = std::move(name);
}

ChildrenRef<ClassDefinition> StoredDefinition::getClasses() const {
  return classes;
}

void StoredDefinition::addClass(std::unique_ptr<ClassDefinition> node) {
  classes.emplace_back(std::move(node));
}
} // namespace marco::ast::modelica
