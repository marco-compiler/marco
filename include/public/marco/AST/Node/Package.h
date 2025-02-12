#ifndef MARCO_AST_NODE_PACKAGE_H
#define MARCO_AST_NODE_PACKAGE_H

#include "marco/AST/Node/Class.h"

namespace marco::ast {
class Package : public Class {
public:
  explicit Package(SourceRange location);

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Class_Package;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;
};
} // namespace marco::ast

#endif // MARCO_AST_NODE_PACKAGE_H
