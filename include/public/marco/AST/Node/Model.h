#ifndef PUBLIC_MARCO_AST_NODE_MODEL_H
#define PUBLIC_MARCO_AST_NODE_MODEL_H

#include "marco/AST/Node/ASTNode.h"
#include "marco/AST/Node/Class.h"
#include "marco/Parser/Location.h"
#include <llvm/Support/JSON.h>
#include <memory>

namespace marco::ast {
class Model : public Class {
public:
  explicit Model(SourceRange location);

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Class_Model;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;
};
} // namespace marco::ast

#endif // PUBLIC_MARCO_AST_NODE_MODEL_H
