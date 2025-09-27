#ifndef MARCO_AST_BASEMODELICA_NODE_NAMEDFUNCTIONCALLARGUMENT_H
#define MARCO_AST_BASEMODELICA_NODE_NAMEDFUNCTIONCALLARGUMENT_H

#include "marco/AST/BaseModelica/Node/FunctionArgument.h"

namespace marco::ast::bmodelica {
class Expression;

class NamedFunctionArgument : public FunctionArgument {
public:
  NamedFunctionArgument(SourceRange location);

  NamedFunctionArgument(const NamedFunctionArgument &other);

  ~NamedFunctionArgument() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::FunctionArgument_Named;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  llvm::StringRef getName() const;

  void setName(llvm::StringRef newName);

  FunctionArgument *getValue();

  const FunctionArgument *getValue() const;

  void setValue(std::unique_ptr<ASTNode> node);

private:
  std::string name;
  std::unique_ptr<ASTNode> value;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_NODE_NAMEDFUNCTIONCALLARGUMENT_H
