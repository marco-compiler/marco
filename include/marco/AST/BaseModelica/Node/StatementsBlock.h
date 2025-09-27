#ifndef MARCO_AST_BASEMODELICA_NODE_STATEMENTSBLOCK_H
#define MARCO_AST_BASEMODELICA_NODE_STATEMENTSBLOCK_H

#include "marco/AST/BaseModelica/Node/ASTNode.h"
#include "llvm/ADT/STLExtras.h"

namespace marco::ast::bmodelica {
class Statement;

class StatementsBlock : public ASTNode {
public:
  StatementsBlock(SourceRange location);

  StatementsBlock(const StatementsBlock &other);

  ~StatementsBlock() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::StatementsBlock;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  size_t size() const;

  Statement *operator[](size_t index);

  const Statement *operator[](size_t index) const;

  void setBody(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

private:
  llvm::SmallVector<std::unique_ptr<ASTNode>> statements;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_NODE_STATEMENTSBLOCK_H
