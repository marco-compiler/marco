#include "marco/AST/Modelica/IfEquation.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void IfEquation::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(conditions, callback);

  walk(branches, callback);

  walk(elseEquations, callback);
  description.walkChildren(callback);
}

void IfEquation::addJSONProperties(llvm::json::Object &obj) const {
  obj["conditions"] = modelica::toJSON(conditions);
  obj["branches"] = modelica::toJSON(branches);
  obj["else"] = modelica::toJSON(elseEquations);
  Equation::addJSONProperties(obj);
}

ChildrenRef<Expression> IfEquation::getConditions() const { return conditions; }

size_t IfEquation::getNumBranches() const { return branches.size(); }

ChildrenRef<Equation> IfEquation::getBranch(size_t index) const {
  return branches[index];
}

void IfEquation::addBranch(std::unique_ptr<Expression> condition,
                           Children<Equation> equations) {
  conditions.emplace_back(std::move(condition));
  branches.push_back(std::move(equations));
}

ChildrenRef<Equation> IfEquation::getElseEquations() const {
  return elseEquations;
}

void IfEquation::setElseEquations(Children<Equation> equations) {
  elseEquations = std::move(equations);
}
} // namespace marco::ast::modelica
