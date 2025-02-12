#ifndef MARCO_AST_NODE_FORINDEX_H
#define MARCO_AST_NODE_FORINDEX_H

#include "marco/AST/Node/ASTNode.h"

namespace marco::ast {
class Expression;

class ForIndex : public ASTNode {
public:
  explicit ForIndex(SourceRange location);

  ForIndex(const ForIndex &other);

  ~ForIndex() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::ForIndex;
  }

  std::unique_ptr<ASTNode> clone() const override;

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
} // namespace marco::ast

#endif // MARCO_AST_NODE_FORINDEX_H
