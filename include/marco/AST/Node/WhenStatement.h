#ifndef MARCO_AST_NODE_WHENSTATEMENT_H
#define MARCO_AST_NODE_WHENSTATEMENT_H

#include "marco/AST/Node/Statement.h"

namespace marco::ast {
class Expression;

class WhenStatement : public Statement {
public:
  explicit WhenStatement(SourceRange location);

  WhenStatement(const WhenStatement &other);

  ~WhenStatement() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Statement_When;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  Expression *getCondition();

  const Expression *getCondition() const;

  void setCondition(std::unique_ptr<ASTNode> node);

  size_t size() const;

  Statement *operator[](size_t index);

  const Statement *operator[](size_t index) const;

  llvm::ArrayRef<std::unique_ptr<ASTNode>> getStatements() const;

  void setStatements(llvm::ArrayRef<std::unique_ptr<ASTNode>> newStatements);

private:
  std::unique_ptr<ASTNode> condition;
  llvm::SmallVector<std::unique_ptr<ASTNode>> statements;
};
} // namespace marco::ast

#endif // MARCO_AST_NODE_WHENSTATEMENT_H
