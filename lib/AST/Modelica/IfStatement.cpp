#include "marco/AST/Modelica/IfStatement.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void IfStatement::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(conditions, callback);

  walk(branches, callback);

  walk(elseStatements, callback);
  description.walkChildren(callback);
}

void IfStatement::addJSONProperties(llvm::json::Object &obj) const {
  obj["conditions"] = modelica::toJSON(conditions);
  obj["branches"] = modelica::toJSON(branches);
  obj["else"] = modelica::toJSON(elseStatements);
  Statement::addJSONProperties(obj);
}

ChildrenRef<Expression> IfStatement::getConditions() const {
  return conditions;
}

size_t IfStatement::getNumBranches() const { return branches.size(); }

ChildrenRef<Statement> IfStatement::getBranch(size_t index) const {
  return branches[index];
}

void IfStatement::addBranch(std::unique_ptr<Expression> condition,
                            Children<Statement> statements) {
  conditions.emplace_back(std::move(condition));
  branches.push_back(std::move(statements));
}

ChildrenRef<Statement> IfStatement::getElseStatements() const {
  return elseStatements;
}

void IfStatement::setElseStatements(Children<Statement> statements) {
  elseStatements = std::move(statements);
}
} // namespace marco::ast::modelica
