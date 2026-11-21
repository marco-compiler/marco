#include "marco/AST/Modelica/Expression.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
bool Expression::classof(const ASTNode *node) {
  return node->getKind<NodeKind>() >= NodeKind::Expression &&
         node->getKind<NodeKind>() <= NodeKind::Expression_Last;
}
} // namespace marco::ast::modelica
