#ifndef PUBLIC_MARCO_AST_NODE_ALGORITHM_H
#define PUBLIC_MARCO_AST_NODE_ALGORITHM_H

#include "marco/AST/Node/ASTNode.h"
#include "marco/Parser/Location.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/JSON.h>
#include <cstddef>
#include <memory>

namespace marco::ast {
class Statement;

class Algorithm : public ASTNode {
public:
  explicit Algorithm(SourceRange location);

  Algorithm(const Algorithm &other);

  ~Algorithm() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Algorithm;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  bool isInitial() const;

  void setInitial(bool value);

  size_t size() const;

  bool empty() const;

  Statement *operator[](size_t index);

  const Statement *operator[](size_t index) const;

  llvm::ArrayRef<std::unique_ptr<ASTNode>> getStatements();

  void setStatements(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

private:
  bool initial{false};
  llvm::SmallVector<std::unique_ptr<ASTNode>> statements;
};
} // namespace marco::ast

#endif // PUBLIC_MARCO_AST_NODE_ALGORITHM_H
