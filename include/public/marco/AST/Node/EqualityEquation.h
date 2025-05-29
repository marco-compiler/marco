#ifndef PUBLIC_MARCO_AST_NODE_EQUALITYEQUATION_H
#define PUBLIC_MARCO_AST_NODE_EQUALITYEQUATION_H

#include "marco/AST/Node/ASTNode.h"
#include "marco/Parser/Location.h"
#include <llvm/Support/JSON.h>
#include <memory>

namespace marco::ast {
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
} // namespace marco::ast

#endif // PUBLIC_MARCO_AST_NODE_EQUALITYEQUATION_H
