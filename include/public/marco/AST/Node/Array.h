#ifndef MARCO_AST_NODE_ARRAY_H
#define MARCO_AST_NODE_ARRAY_H

#include "marco/AST/Node/Expression.h"
#include "marco/AST/Node/Type.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

namespace marco::ast
{
	class Expression;

	class Array : public Expression
	{
		public:
      explicit Array(SourceRange location);

      Array(const Array& other);

      static bool classof(const ASTNode* node)
      {
        return node->getKind() == ASTNode::Kind::Expression_Array;
      }

      std::unique_ptr<ASTNode> clone() const override;

      llvm::json::Value toJSON() const override;

      bool isLValue() const override;

      size_t size() const;

      Expression* operator[](size_t index);

      const Expression* operator[](size_t index) const;

      void setValues(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

    private:
      llvm::SmallVector<std::unique_ptr<ASTNode>> values;
	};
}

#endif // MARCO_AST_NODE_ARRAY_H
