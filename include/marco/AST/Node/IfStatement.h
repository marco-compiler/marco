#ifndef MARCO_AST_NODE_IFSTATEMENT_H
#define MARCO_AST_NODE_IFSTATEMENT_H

#include "marco/AST/Node/Statement.h"

namespace marco::ast {
class Expression;
class StatementsBlock;

class IfStatement : public ASTNode {
public:
  IfStatement(SourceRange location);

  IfStatement(const IfStatement &other);

  ~IfStatement() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Statement_If;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  Expression *getIfCondition();

  const Expression *getIfCondition() const;

  void setIfCondition(std::unique_ptr<ASTNode> node);

  StatementsBlock *getIfBlock();

  const StatementsBlock *getIfBlock() const;

  void setIfBlock(std::unique_ptr<ASTNode> node);

  size_t getNumOfElseIfBlocks() const;

  bool hasElseIfBlocks() const;

  Expression *getElseIfCondition(size_t index);

  const Expression *getElseIfCondition(size_t index) const;

  void setElseIfConditions(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

  StatementsBlock *getElseIfBlock(size_t index);

  const StatementsBlock *getElseIfBlock(size_t index) const;

  void setElseIfBlocks(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

  bool hasElseBlock() const;

  StatementsBlock *getElseBlock();

  const StatementsBlock *getElseBlock() const;

  void setElseBlock(std::unique_ptr<ASTNode> node);

private:
  std::unique_ptr<ASTNode> ifCondition;
  std::unique_ptr<ASTNode> ifBlock;

  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> elseIfConditions;
  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> elseIfBlocks;

  std::unique_ptr<ASTNode> elseBlock;
};
} // namespace marco::ast

#endif // MARCO_AST_NODE_IFSTATEMENT_H
