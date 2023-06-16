#ifndef MARCO_AST_NODE_ARRAY_FOR_GENERATOR_H
#define MARCO_AST_NODE_ARRAY_FOR_GENERATOR_H

#include "marco/AST/Node/ArrayGenerator.h"
#include "marco/AST/Node/Type.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

namespace marco::ast
{
	class ArrayGenerator;
  class Induction;

	class ArrayForGenerator : public ArrayGenerator
	{
		public:
      explicit ArrayForGenerator(SourceRange location);

      ArrayForGenerator(const ArrayForGenerator& other);

      static bool classof(const ASTNode* node)
      {
        return node->getKind() == ASTNode::Kind::Expression_ArrayGenerator_ArrayForGenerator;
      }

      std::unique_ptr<ASTNode> clone() const override;

      llvm::json::Value toJSON() const override;

      Expression* getValue();

      const Expression* getValue() const;

      void setValue(std::unique_ptr<ASTNode> node);

      unsigned getNumIndices() const;

      Induction* getIndex(unsigned index);

      const Induction* getIndex(unsigned index) const;

      void setIndices(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

    private:
      std::unique_ptr<ASTNode> value;
      llvm::SmallVector<std::unique_ptr<ASTNode>> indices;
	};
}

#endif // MARCO_AST_NODE_ARRAY_FOR_GENERATOR_H
