#ifndef PUBLIC_MARCO_AST_NODE_FORSTATEMENT_H
#define PUBLIC_MARCO_AST_NODE_FORSTATEMENT_H

#include "marco/AST/Node/ASTNode.h"
#include "marco/AST/Node/Statement.h"
#include "marco/Parser/Location.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/JSON.h>
#include <cstddef>
#include <memory>

namespace marco::ast {
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
} // namespace marco::ast

#endif // PUBLIC_MARCO_AST_NODE_FORSTATEMENT_H
