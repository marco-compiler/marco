#include "marco/AST/Modelica/WhileStatement.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void WhileStatement::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(condition, callback);
  walk(statements, callback);
  description.walkChildren(callback);
}

void WhileStatement::addJSONProperties(llvm::json::Object &obj) const {
  obj["condition"] = modelica::toJSON(condition.get());
  obj["statements"] = modelica::toJSON(statements);
  Statement::addJSONProperties(obj);
}

Expression *WhileStatement::getCondition() const { return condition.get(); }

void WhileStatement::setCondition(std::unique_ptr<Expression> node) {
  condition = std::move(node);
}

ChildrenRef<Statement> WhileStatement::getStatements() const {
  return statements;
}

void WhileStatement::setStatements(Children<Statement> body) {
  statements = std::move(body);
}
} // namespace marco::ast::modelica
