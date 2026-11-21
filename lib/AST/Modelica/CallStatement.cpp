#include "marco/AST/Modelica/CallStatement.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void CallStatement::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(call, callback);
  description.walkChildren(callback);
}

void CallStatement::addJSONProperties(llvm::json::Object &obj) const {
  obj["call"] = modelica::toJSON(call.get());
  Statement::addJSONProperties(obj);
}

Call *CallStatement::getCall() const { return call.get(); }

void CallStatement::setCall(std::unique_ptr<Call> node) {
  call = std::move(node);
}
} // namespace marco::ast::modelica
