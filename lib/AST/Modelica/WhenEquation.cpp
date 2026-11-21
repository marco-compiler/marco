#include "marco/AST/Modelica/WhenEquation.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void WhenEquation::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(conditions, callback);

  walk(branches, callback);

  description.walkChildren(callback);
}

void WhenEquation::addJSONProperties(llvm::json::Object &obj) const {
  obj["conditions"] = modelica::toJSON(conditions);
  obj["branches"] = modelica::toJSON(branches);
  Equation::addJSONProperties(obj);
}

ChildrenRef<Expression> WhenEquation::getConditions() const {
  return conditions;
}

size_t WhenEquation::getNumBranches() const { return branches.size(); }

ChildrenRef<Equation> WhenEquation::getBranch(size_t index) const {
  return branches[index];
}

void WhenEquation::addBranch(std::unique_ptr<Expression> condition,
                             Children<Equation> equations) {
  conditions.emplace_back(std::move(condition));
  branches.push_back(std::move(equations));
}
} // namespace marco::ast::modelica
