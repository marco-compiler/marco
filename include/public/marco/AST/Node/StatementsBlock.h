#ifndef PUBLIC_MARCO_AST_NODE_STATEMENTSBLOCK_H
#define PUBLIC_MARCO_AST_NODE_STATEMENTSBLOCK_H


#include "llvm/ADT/ArrayRef.h"             // para llvm::ArrayRef
#include "llvm/ADT/SmallVector.h"          // para llvm::SmallVector
#include "llvm/Support/JSON.h"             // para llvm::json::Value

#include <cstddef>                         // para size_t
#include <memory>                          // para std::unique_ptr

#include "marco/AST/Node/ASTNode.h"
#include "marco/AST/Node/Statement.h"
#include "marco/Parser/Location.h"



namespace marco::ast {
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
} // namespace marco::ast

#endif // PUBLIC_MARCO_AST_NODE_STATEMENTSBLOCK_H
