#ifndef MARCO_AST_BASEMODELICA_FORINDEX_H
#define MARCO_AST_BASEMODELICA_FORINDEX_H

#include "marco/AST/BaseModelica/ASTNode.h"

namespace marco::ast::bmodelica {
class Expression;

class ForIndex : public ASTNode {
public:
  explicit ForIndex(SourceRange location);

  ForIndex(const ForIndex &other);

  ~ForIndex() override;

  static bool classof(const ASTNode *node) {
    return node->getKind<ASTNodeKind>() == ASTNodeKind::ForIndex;
  }

  std::unique_ptr<ast::ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  llvm::StringRef getName() const;

  void setName(llvm::StringRef newName);

  bool hasExpression() const;

  Expression *getExpression();

  const Expression *getExpression() const;

  void setExpression(std::unique_ptr<ASTNode> node);

private:
  std::string name;
  std::unique_ptr<ASTNode> expression;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_FORINDEX_H
