#ifndef MARCO_AST_NODE_CALLARGUMENT_H
#define MARCO_AST_NODE_CALLARGUMENT_H

#include "marco/AST/Node/ASTNode.h"

namespace marco::ast
{
  class Expression;

  class CallArgument : public ASTNode
  {
    public:
      CallArgument(SourceRange location);

      CallArgument(const CallArgument& other);

      ~CallArgument() override;

      static bool classof(const ASTNode* node)
      {
        return node->getKind() == ASTNode::Kind::CallArgument;
      }

      std::unique_ptr<ASTNode> clone() const override;

      llvm::json::Value toJSON() const override;

      bool isNamed() const;

      llvm::StringRef getName() const;

      void setName(llvm::StringRef newName);

      Expression* getValue();

      const Expression* getValue() const;

      void setValue(std::unique_ptr<ASTNode> node);

    private:
      std::optional<std::string> name;
      std::unique_ptr<ASTNode> value;
  };
}

#endif // MARCO_AST_NODE_CALLARGUMENT_H
