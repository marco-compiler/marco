#ifndef MARCO_AST_BASEMODELICA_NODE_FUNCTIONARGUMENT_H
#define MARCO_AST_BASEMODELICA_NODE_FUNCTIONARGUMENT_H

#include "marco/AST/BaseModelica/Node/ASTNode.h"

namespace marco::ast::bmodelica {
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
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_NODE_FUNCTIONARGUMENT_H
