#include "marco/AST/Modelica/AlgorithmSection.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void AlgorithmSection::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(statements, callback);
}

void AlgorithmSection::addJSONProperties(llvm::json::Object &obj) const {
  obj["initial"] = initial;
  obj["statements"] = modelica::toJSON(statements);
}

bool AlgorithmSection::isInitial() const { return initial; }

void AlgorithmSection::setInitial(bool value) { initial = value; }

ChildrenRef<Statement> AlgorithmSection::getStatements() const {
  return statements;
}

void AlgorithmSection::addStatement(std::unique_ptr<Statement> node) {
  statements.emplace_back(std::move(node));
}
} // namespace marco::ast::modelica
