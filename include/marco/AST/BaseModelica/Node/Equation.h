#ifndef MARCO_AST_BASEMODELICA_NODE_EQUATION_H
#define MARCO_AST_BASEMODELICA_NODE_EQUATION_H

#include "marco/AST/BaseModelica/Node/ASTNode.h"

namespace marco::ast::bmodelica {
class Equation : public ASTNode {
public:
  using ASTNode::ASTNode;

  ~Equation() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() >= ASTNode::Kind::Equation &&
           node->getKind() <= ASTNode::Kind::Equation_LastEquation;
  }
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_NODE_EQUATION_H
