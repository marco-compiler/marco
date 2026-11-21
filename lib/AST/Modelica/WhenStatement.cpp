#include "marco/AST/Modelica/WhenStatement.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void WhenStatement::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(conditions, callback);

  walk(branches, callback);

  description.walkChildren(callback);
}

void WhenStatement::addJSONProperties(llvm::json::Object &obj) const {
  obj["conditions"] = modelica::toJSON(conditions);
  obj["branches"] = modelica::toJSON(branches);
  Statement::addJSONProperties(obj);
}

ChildrenRef<Expression> WhenStatement::getConditions() const {
  return conditions;
}

size_t WhenStatement::getNumBranches() const { return branches.size(); }

ChildrenRef<Statement> WhenStatement::getBranch(size_t index) const {
  return branches[index];
}

void WhenStatement::addBranch(std::unique_ptr<Expression> condition,
                              Children<Statement> statements) {
  conditions.emplace_back(std::move(condition));
  branches.push_back(std::move(statements));
}
} // namespace marco::ast::modelica
