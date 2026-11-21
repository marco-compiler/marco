#include "marco/AST/Modelica/OutputList.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void OutputList::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(elements, callback);
}

void OutputList::addJSONProperties(llvm::json::Object &obj) const {
  obj["elements"] = modelica::toJSON(elements);
}

ChildrenRef<Expression> OutputList::getElements() const { return elements; }

void OutputList::addElement(std::unique_ptr<Expression> node) {
  elements.emplace_back(std::move(node));
}
} // namespace marco::ast::modelica
