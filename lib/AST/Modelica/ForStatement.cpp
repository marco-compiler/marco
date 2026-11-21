#include "marco/AST/Modelica/ForStatement.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void ForStatement::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(indices, callback);
  walk(statements, callback);
  description.walkChildren(callback);
}

void ForStatement::addJSONProperties(llvm::json::Object &obj) const {
  obj["indices"] = modelica::toJSON(indices);
  obj["statements"] = modelica::toJSON(statements);
  Statement::addJSONProperties(obj);
}

ChildrenRef<ForIndex> ForStatement::getIndices() const { return indices; }

void ForStatement::addIndex(std::unique_ptr<ForIndex> node) {
  indices.emplace_back(std::move(node));
}

ChildrenRef<Statement> ForStatement::getStatements() const {
  return statements;
}

void ForStatement::setStatements(Children<Statement> body) {
  statements = std::move(body);
}
} // namespace marco::ast::modelica
