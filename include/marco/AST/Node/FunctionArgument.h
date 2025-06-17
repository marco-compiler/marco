#ifndef MARCO_AST_NODE_FUNCTIONARGUMENT_H
#define MARCO_AST_NODE_FUNCTIONARGUMENT_H

#include "marco/AST/Node/ASTNode.h"

namespace marco::ast {
class FunctionArgument : public ASTNode {
public:
  using ASTNode::ASTNode;

  FunctionArgument(const FunctionArgument &other);

  ~FunctionArgument() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() >= ASTNode::Kind::FunctionArgument &&
           node->getKind() <=
               ASTNode::Kind::FunctionArgument_LastFunctionArgument;
  }

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;
};
} // namespace marco::ast

#endif // MARCO_AST_NODE_FUNCTIONARGUMENT_H
