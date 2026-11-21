#include "marco/AST/Modelica/ConnectEquation.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void ConnectEquation::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(lhs, callback);
  walk(rhs, callback);
  description.walkChildren(callback);
}

void ConnectEquation::addJSONProperties(llvm::json::Object &obj) const {
  obj["lhs"] = modelica::toJSON(lhs.get());
  obj["rhs"] = modelica::toJSON(rhs.get());
  Equation::addJSONProperties(obj);
}

ComponentReference *ConnectEquation::getLhs() const { return lhs.get(); }

void ConnectEquation::setLhs(std::unique_ptr<ComponentReference> node) {
  lhs = std::move(node);
}

ComponentReference *ConnectEquation::getRhs() const { return rhs.get(); }

void ConnectEquation::setRhs(std::unique_ptr<ComponentReference> node) {
  rhs = std::move(node);
}
} // namespace marco::ast::modelica
