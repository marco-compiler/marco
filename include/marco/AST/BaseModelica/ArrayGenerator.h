#ifndef MARCO_AST_BASEMODELICA_NODE_ARRAYGENERATOR_H
#define MARCO_AST_BASEMODELICA_NODE_ARRAYGENERATOR_H

#include "marco/AST/BaseModelica/Expression.h"

namespace marco::ast::bmodelica {
class Expression;

class ArrayGenerator : public Expression {
public:
  using Expression::Expression;

  ~ArrayGenerator() override;

  static bool classof(const ASTNode *node) {
    return node->getKind<ASTNodeKind>() >=
               ASTNodeKind::Expression_ArrayGenerator &&
           node->getKind<ASTNodeKind>() <=
               ASTNodeKind::Expression_ArrayGenerator_LastArrayGenerator;
  }

  bool isLValue() const override { return false; }
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_NODE_ARRAYGENERATOR_H
