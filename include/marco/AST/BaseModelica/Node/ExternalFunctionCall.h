#ifndef MARCO_AST_NODE_EXTERNALFUNCTIONCALL_H
#define MARCO_AST_NODE_EXTERNALFUNCTIONCALL_H

#include "marco/AST/BaseModelica/Node/ASTNode.h"

namespace marco::ast::bmodelica {
class Expression;

class ExternalFunctionCall : public ASTNode {
public:
  using ASTNode::ASTNode;

  explicit ExternalFunctionCall(SourceRange location);

  ExternalFunctionCall(const ExternalFunctionCall &other);

  ~ExternalFunctionCall() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::ExternalFunctionCall;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  bool hasDestination() const;

  Expression *getDestination();

  const Expression *getDestination() const;

  void setDestination(std::unique_ptr<ASTNode> node);

  llvm::StringRef getCallee() const;

  void setCallee(std::string newCallee);

  size_t getNumOfArguments() const;

  Expression *getArgument(size_t index);

  const Expression *getArgument(size_t index) const;

  llvm::ArrayRef<std::unique_ptr<ASTNode>> getArguments() const;

  void setArguments(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

private:
  std::unique_ptr<ASTNode> destination;
  std::string callee;
  llvm::SmallVector<std::unique_ptr<ASTNode>> arguments;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_NODE_EXTERNALFUNCTIONCALL_H
