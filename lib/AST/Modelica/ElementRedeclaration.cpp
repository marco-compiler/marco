#include "marco/AST/Modelica/ElementRedeclaration.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void ElementRedeclaration::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(element, callback);
  walk(constrainingClause, callback);
}

void ElementRedeclaration::addJSONProperties(llvm::json::Object &obj) const {
  obj["redeclare"] = redeclare;

  if (each) {
    obj["each"] = true;
  }

  if (final) {
    obj["final"] = true;
  }

  if (replaceable) {
    obj["replaceable"] = true;
  }

  obj["element"] = modelica::toJSON(element.get());

  if (constrainingClause) {
    obj["constraining_clause"] = modelica::toJSON(constrainingClause.get());
  }
}

bool ElementRedeclaration::isRedeclare() const { return redeclare; }

void ElementRedeclaration::setRedeclare(bool value) { redeclare = value; }

bool ElementRedeclaration::isEach() const { return each; }

void ElementRedeclaration::setEach(bool value) { each = value; }

bool ElementRedeclaration::isFinal() const { return final; }

void ElementRedeclaration::setFinal(bool value) { final = value; }

bool ElementRedeclaration::isReplaceable() const { return replaceable; }

void ElementRedeclaration::setReplaceable(bool value) { replaceable = value; }

Node *ElementRedeclaration::getElement() const { return element.get(); }

void ElementRedeclaration::setElement(std::unique_ptr<Node> node) {
  element = std::move(node);
}

ConstrainingClause *ElementRedeclaration::getConstrainingClause() const {
  return constrainingClause.get();
}

void ElementRedeclaration::setConstrainingClause(
    std::unique_ptr<ConstrainingClause> node) {
  constrainingClause = std::move(node);
}
} // namespace marco::ast::modelica
