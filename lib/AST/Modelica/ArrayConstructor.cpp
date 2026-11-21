#include "marco/AST/Modelica/ArrayConstructor.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void ArrayConstructor::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(elements, callback);
  walk(iterators, callback);
}

void ArrayConstructor::addJSONProperties(llvm::json::Object &obj) const {
  obj["elements"] = modelica::toJSON(elements);

  if (!iterators.empty()) {
    obj["iterators"] = modelica::toJSON(iterators);
  }
}

ChildrenRef<Expression> ArrayConstructor::getElements() const {
  return elements;
}

void ArrayConstructor::addElement(std::unique_ptr<Expression> node) {
  elements.emplace_back(std::move(node));
}

ChildrenRef<ForIndex> ArrayConstructor::getIterators() const {
  return iterators;
}

void ArrayConstructor::addIterator(std::unique_ptr<ForIndex> node) {
  iterators.emplace_back(std::move(node));
}
} // namespace marco::ast::modelica
