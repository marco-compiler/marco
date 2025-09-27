#ifndef MARCO_AST_BASEMODELICA_NODE_PACKAGE_H
#define MARCO_AST_BASEMODELICA_NODE_PACKAGE_H

#include "marco/AST/BaseModelica/Node/Class.h"

namespace marco::ast::bmodelica {
class Package : public Class {
public:
  explicit Package(SourceRange location);

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Class_Package;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_NODE_PACKAGE_H
