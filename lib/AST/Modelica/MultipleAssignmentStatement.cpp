#include "marco/AST/Modelica/MultipleAssignmentStatement.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void MultipleAssignmentStatement::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(destinations, callback);
  walk(call, callback);
  description.walkChildren(callback);
}

void MultipleAssignmentStatement::addJSONProperties(
    llvm::json::Object &obj) const {
  obj["destinations"] = modelica::toJSON(destinations);
  obj["call"] = modelica::toJSON(call.get());
  Statement::addJSONProperties(obj);
}

ChildrenRef<Expression> MultipleAssignmentStatement::getDestinations() const {
  return destinations;
}

void MultipleAssignmentStatement::addDestination(
    std::unique_ptr<Expression> node) {
  destinations.emplace_back(std::move(node));
}

Call *MultipleAssignmentStatement::getCall() const { return call.get(); }

void MultipleAssignmentStatement::setCall(std::unique_ptr<Call> node) {
  call = std::move(node);
}
} // namespace marco::ast::modelica
