#ifndef MARCO_AST_NODE_ARRAY_GENERATOR_H
#define MARCO_AST_NODE_ARRAY_GENERATOR_H

#include "marco/AST/Node/Expression.h"

namespace marco::ast {
class Expression;

class ArrayGenerator : public Expression {
public:
  using Expression::Expression;

  virtual ~ArrayGenerator();

  static bool classof(const ASTNode *node) {
    return node->getKind() >= ASTNode::Kind::Expression_ArrayGenerator &&
           node->getKind() <=
               ASTNode::Kind::Expression_ArrayGenerator_LastArrayGenerator;
  }

  virtual bool isLValue() const override { return false; }
};
} // namespace marco::ast

#endif // MARCO_AST_NODE_ARRAY_GENERATOR_H
