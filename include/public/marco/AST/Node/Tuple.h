#ifndef MARCO_AST_NODE_TUPLE_H
#define MARCO_AST_NODE_TUPLE_H

#include "marco/AST/Node/Expression.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

namespace marco::ast
{
	/// A tuple is a container for destinations of a call. It is NOT an
	/// array-like structure that is supposed to be summable, passed around or
	/// whatever.
	class Tuple : public Expression
	{
		public:
      Tuple(SourceRange location);

      Tuple(const Tuple& other);

      ~Tuple() override;

      static bool classof(const ASTNode* node)
      {
        return node->getKind() == ASTNode::Kind::Expression_Tuple;
      }

      std::unique_ptr<ASTNode> clone() const override;

      llvm::json::Value toJSON() const override;

      bool isLValue() const override;

      size_t size() const;

      Expression* getExpression(size_t index);

      const Expression* getExpression(size_t index) const;

      void setExpressions(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

    private:
      llvm::SmallVector<std::unique_ptr<ASTNode>> expressions;
	};
}

#endif // MARCO_AST_NODE_TUPLE_H
