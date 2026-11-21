#ifndef MARCO_AST_MODELICA_ARGUMENT_H
#define MARCO_AST_MODELICA_ARGUMENT_H

#include "marco/AST/Modelica/Node.h"

namespace marco::ast::modelica {
/// Base class of the arguments of a class modification.
class Argument : public Node {
public:
  using Node::Node;

  static bool classof(const ASTNode *node);
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_ARGUMENT_H
