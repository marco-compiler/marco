#include "marco/AST/Modelica/Statement.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void Statement::addJSONProperties(llvm::json::Object &obj) const {
  description.addToJSON(obj);
}

bool Statement::classof(const ASTNode *node) {
  return node->getKind<NodeKind>() >= NodeKind::Statement &&
         node->getKind<NodeKind>() <= NodeKind::Statement_Last;
}

Description &Statement::getDescription() { return description; }

const Description &Statement::getDescription() const { return description; }
} // namespace marco::ast::modelica
