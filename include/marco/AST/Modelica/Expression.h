#ifndef MARCO_AST_MODELICA_EXPRESSION_H
#define MARCO_AST_MODELICA_EXPRESSION_H

#include "marco/AST/Modelica/Node.h"

namespace marco::ast::modelica {
/// Base class of the expressions.
class Expression : public Node {
public:
  using Node::Node;

  static bool classof(const ASTNode *node);
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_EXPRESSION_H
