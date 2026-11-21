#include "marco/AST/Modelica/CallEquation.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void CallEquation::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(call, callback);
  description.walkChildren(callback);
}

void CallEquation::addJSONProperties(llvm::json::Object &obj) const {
  obj["call"] = modelica::toJSON(call.get());
  Equation::addJSONProperties(obj);
}

Call *CallEquation::getCall() const { return call.get(); }

void CallEquation::setCall(std::unique_ptr<Call> node) {
  call = std::move(node);
}
} // namespace marco::ast::modelica
