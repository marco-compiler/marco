#ifndef MARCO_AST_NODE_ARRAY_CONSTANT_H
#define MARCO_AST_NODE_ARRAY_CONSTANT_H

#include "marco/AST/Node/ArrayGenerator.h"
#include "marco/AST/Node/Type.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

namespace marco::ast
{
	class ArrayGenerator;

	class ArrayConstant : public ArrayGenerator
	{
		public:
      explicit ArrayConstant(SourceRange location);

      ArrayConstant(const ArrayConstant& other);

      static bool classof(const ASTNode* node)
      {
        return node->getKind() == ASTNode::Kind::Expression_ArrayGenerator_ArrayConstant;
      }

      std::unique_ptr<ASTNode> clone() const override;

      llvm::json::Value toJSON() const override;

      size_t size() const;

      Expression* operator[](size_t index);

      const Expression* operator[](size_t index) const;

      void setValues(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

    private:
      llvm::SmallVector<std::unique_ptr<ASTNode>> values;
	};
}

#endif // MARCO_AST_NODE_ARRAY_CONSTANT_H
