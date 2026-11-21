#include "marco/AST/Modelica/AssignmentStatement.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void AssignmentStatement::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(destination, callback);
  walk(value, callback);
  description.walkChildren(callback);
}

void AssignmentStatement::addJSONProperties(llvm::json::Object &obj) const {
  obj["destination"] = modelica::toJSON(destination.get());
  obj["value"] = modelica::toJSON(value.get());
  Statement::addJSONProperties(obj);
}

Expression *AssignmentStatement::getDestination() const {
  return destination.get();
}

void AssignmentStatement::setDestination(std::unique_ptr<Expression> node) {
  destination = std::move(node);
}

Expression *AssignmentStatement::getValue() const { return value.get(); }

void AssignmentStatement::setValue(std::unique_ptr<Expression> node) {
  value = std::move(node);
}
} // namespace marco::ast::modelica
