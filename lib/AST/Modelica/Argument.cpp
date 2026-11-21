#include "marco/AST/Modelica/Argument.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
bool Argument::classof(const ASTNode *node) {
  return node->getKind<NodeKind>() >= NodeKind::Argument &&
         node->getKind<NodeKind>() <= NodeKind::Argument_Last;
}
} // namespace marco::ast::modelica
