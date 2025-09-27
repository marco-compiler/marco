#ifndef MARCO_AST_BASEMODELICA_NODE_EQUALITYEQUATION_H
#define MARCO_AST_BASEMODELICA_NODE_EQUALITYEQUATION_H

#include "marco/AST/BaseModelica/Node/ASTNode.h"
#include <memory>

namespace marco::ast::bmodelica {
class Expression;

class EqualityEquation : public ASTNode {
public:
  explicit EqualityEquation(SourceRange location);

  EqualityEquation(const EqualityEquation &other);

  ~EqualityEquation() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Equation_Equality;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  Expression *getLhsExpression();

  const Expression *getLhsExpression() const;

  void setLhsExpression(std::unique_ptr<ASTNode> node);

  Expression *getRhsExpression();

  const Expression *getRhsExpression() const;

  void setRhsExpression(std::unique_ptr<ASTNode> node);

private:
  std::unique_ptr<ASTNode> lhs;
  std::unique_ptr<ASTNode> rhs;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_NODE_EQUALITYEQUATION_H
