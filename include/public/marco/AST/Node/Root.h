#ifndef PUBLIC_MARCO_AST_NODE_ROOT_H
#define PUBLIC_MARCO_AST_NODE_ROOT_H

#include "marco/AST/Node/ASTNode.h"
#include "marco/Parser/Location.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/JSON.h>
#include <memory>

namespace marco::ast {
class Root : public ASTNode {
public:
  using ASTNode::ASTNode;

  Root(SourceRange location);

  Root(const Root &other);

  ~Root() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Root;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  /// Get the inner classes.
  llvm::ArrayRef<std::unique_ptr<ASTNode>> getInnerClasses() const;

  /// Set the inner classes.
  void setInnerClasses(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

private:
  llvm::SmallVector<std::unique_ptr<ASTNode>> innerClasses;
};
} // namespace marco::ast

#endif // PUBLIC_MARCO_AST_NODE_ROOT_H
