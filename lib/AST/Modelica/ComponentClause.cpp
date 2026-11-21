#include "marco/AST/Modelica/ComponentClause.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void ComponentClause::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(subscripts, callback);
  walk(declarations, callback);
  walk(constrainingClause, callback);
}

void ComponentClause::addJSONProperties(llvm::json::Object &obj) const {
  prefixes.addToJSON(obj);
  typePrefix.addToJSON(obj);
  obj["type"] = modelica::toJSON(typeSpecifier);

  if (!subscripts.empty()) {
    obj["subscripts"] = modelica::toJSON(subscripts);
  }

  obj["declarations"] = modelica::toJSON(declarations);

  if (constrainingClause) {
    obj["constraining_clause"] = modelica::toJSON(constrainingClause.get());
  }
}

ElementPrefixes &ComponentClause::getPrefixes() { return prefixes; }

const ElementPrefixes &ComponentClause::getPrefixes() const { return prefixes; }

TypePrefix &ComponentClause::getTypePrefix() { return typePrefix; }

const TypePrefix &ComponentClause::getTypePrefix() const { return typePrefix; }

const Name &ComponentClause::getTypeSpecifier() const { return typeSpecifier; }

void ComponentClause::setTypeSpecifier(Name name) {
  typeSpecifier = std::move(name);
}

ChildrenRef<Expression> ComponentClause::getSubscripts() const {
  return subscripts;
}

void ComponentClause::addSubscript(std::unique_ptr<Expression> node) {
  subscripts.emplace_back(std::move(node));
}

ChildrenRef<ComponentDeclaration> ComponentClause::getDeclarations() const {
  return declarations;
}

void ComponentClause::addDeclaration(
    std::unique_ptr<ComponentDeclaration> node) {
  declarations.emplace_back(std::move(node));
}

ConstrainingClause *ComponentClause::getConstrainingClause() const {
  return constrainingClause.get();
}

void ComponentClause::setConstrainingClause(
    std::unique_ptr<ConstrainingClause> node) {
  constrainingClause = std::move(node);
}
} // namespace marco::ast::modelica
