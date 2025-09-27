#ifndef MARCO_AST_BASEMODELICA_NODE_ROOT_H
#define MARCO_AST_BASEMODELICA_NODE_ROOT_H

#include "marco/AST/BaseModelica/Node/ASTNode.h"

namespace marco::ast::bmodelica {
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
  void
  setInnerClasses(llvm::ArrayRef<std::unique_ptr<ASTNode>> newInnerClasses);

private:
  llvm::SmallVector<std::unique_ptr<ASTNode>> innerClasses;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_NODE_ROOT_H
