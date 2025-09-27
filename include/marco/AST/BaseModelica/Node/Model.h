#ifndef MARCO_AST_BASEMODELICA_NODE_MODEL_H
#define MARCO_AST_BASEMODELICA_NODE_MODEL_H

#include "marco/AST/BaseModelica/Node/Class.h"

namespace marco::ast::bmodelica {
class Model : public Class {
public:
  explicit Model(SourceRange location);

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Class_Model;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_NODE_MODEL_H
