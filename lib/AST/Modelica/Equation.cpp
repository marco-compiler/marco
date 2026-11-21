#include "marco/AST/Modelica/Equation.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void Equation::addJSONProperties(llvm::json::Object &obj) const {
  description.addToJSON(obj);
}

bool Equation::classof(const ASTNode *node) {
  return node->getKind<NodeKind>() >= NodeKind::Equation &&
         node->getKind<NodeKind>() <= NodeKind::Equation_Last;
}

Description &Equation::getDescription() { return description; }

const Description &Equation::getDescription() const { return description; }
} // namespace marco::ast::modelica
