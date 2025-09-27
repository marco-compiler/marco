#ifndef MARCO_AST_BASEMODELICA_NODE_FORSTATEMENT_H
#define MARCO_AST_BASEMODELICA_NODE_FORSTATEMENT_H

#include "marco/AST/BaseModelica/Node/Statement.h"

namespace marco::ast::bmodelica {
class ForIndex;

class ForStatement : public Statement {
public:
  explicit ForStatement(SourceRange location);

  ForStatement(const ForStatement &other);

  ~ForStatement() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Statement_For;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  size_t getNumOfForIndices() const;

  ForIndex *getForIndex(size_t index);

  const ForIndex *getForIndex(size_t index) const;

  llvm::ArrayRef<std::unique_ptr<ASTNode>> getForIndices() const;

  void setForIndices(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

  size_t getNumOfStatements() const;

  Statement *getStatement(size_t index);

  const Statement *getStatement(size_t index) const;

  llvm::ArrayRef<std::unique_ptr<ASTNode>> getStatements() const;

  void setStatements(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

private:
  llvm::SmallVector<std::unique_ptr<ASTNode>> forIndices;
  llvm::SmallVector<std::unique_ptr<ASTNode>> statements;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_NODE_FORSTATEMENT_H
