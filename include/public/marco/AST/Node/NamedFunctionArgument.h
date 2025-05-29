#ifndef PUBLIC_MARCO_AST_NODE_NAMEDFUNCTIONARGUMENT_H
#define PUBLIC_MARCO_AST_NODE_NAMEDFUNCTIONARGUMENT_H

#include "marco/AST/Node/ASTNode.h"
#include "marco/AST/Node/FunctionArgument.h"
#include "marco/Parser/Location.h"
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/JSON.h>
#include <memory>
#include <string>

namespace marco::ast {
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
} // namespace marco::ast

#endif // PUBLIC_MARCO_AST_NODE_NAMEDFUNCTIONARGUMENT_H
