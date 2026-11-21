#include "marco/AST/Modelica/EquationSection.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void EquationSection::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(equations, callback);
}

void EquationSection::addJSONProperties(llvm::json::Object &obj) const {
  obj["initial"] = initial;
  obj["equations"] = modelica::toJSON(equations);
}

bool EquationSection::isInitial() const { return initial; }

void EquationSection::setInitial(bool value) { initial = value; }

ChildrenRef<Equation> EquationSection::getEquations() const {
  return equations;
}

void EquationSection::addEquation(std::unique_ptr<Equation> node) {
  equations.emplace_back(std::move(node));
}
} // namespace marco::ast::modelica
