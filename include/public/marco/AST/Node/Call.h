#ifndef MARCO_AST_NODE_CALL_H
#define MARCO_AST_NODE_CALL_H

#include "marco/AST/Node/Expression.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

namespace marco::ast
{
  class CallArgument;

	class Call : public Expression
	{
		public:
      explicit Call(SourceRange location);

      Call(const Call& other);

      ~Call() override;

      static bool classof(const ASTNode* node)
      {
        return node->getKind() == ASTNode::Kind::Expression_Call;
      }

      std::unique_ptr<ASTNode> clone() const override;

      llvm::json::Value toJSON() const override;

      bool isLValue() const override;

      Expression* getCallee();

      const Expression* getCallee() const;

      void setCallee(std::unique_ptr<ASTNode> node);

      size_t getNumOfArguments() const;

      CallArgument* getArgument(size_t index);

      const CallArgument* getArgument(size_t index) const;

      llvm::ArrayRef<std::unique_ptr<ASTNode>> getArguments() const;

      void setArguments(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

    private:
      std::unique_ptr<ASTNode> callee;
      llvm::SmallVector<std::unique_ptr<ASTNode>> arguments;
	};
}

#endif // MARCO_AST_NODE_CALL_H
