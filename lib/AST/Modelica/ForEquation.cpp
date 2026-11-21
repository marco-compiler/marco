#include "marco/AST/Modelica/ForEquation.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void ForEquation::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(indices, callback);
  walk(equations, callback);
  description.walkChildren(callback);
}

void ForEquation::addJSONProperties(llvm::json::Object &obj) const {
  obj["indices"] = modelica::toJSON(indices);
  obj["equations"] = modelica::toJSON(equations);
  Equation::addJSONProperties(obj);
}

ChildrenRef<ForIndex> ForEquation::getIndices() const { return indices; }

void ForEquation::addIndex(std::unique_ptr<ForIndex> node) {
  indices.emplace_back(std::move(node));
}

ChildrenRef<Equation> ForEquation::getEquations() const { return equations; }

void ForEquation::setEquations(Children<Equation> body) {
  equations = std::move(body);
}
} // namespace marco::ast::modelica
