#ifndef MARCO_AST_NODE_TYPEPREFIX_H
#define MARCO_AST_NODE_TYPEPREFIX_H

#include "marco/AST/Node/ASTNode.h"
#include "llvm/ADT/SmallVector.h"
#include <string>
#include <type_traits>

namespace marco::ast
{
	enum class VariabilityQualifier
	{
		discrete,
		parameter,
		constant,
		none
	};

	enum class IOQualifier
	{
		input,
		output,
		none
	};

	class TypePrefix : public ASTNode
	{
		public:
      explicit TypePrefix(SourceRange location);

      TypePrefix(const TypePrefix& other);

      ~TypePrefix() override;

      static bool classof(const ASTNode* node)
      {
        return node->getKind() == ASTNode::Kind::TypePrefix;
      }

      std::unique_ptr<ASTNode> clone() const override;

      llvm::json::Value toJSON() const override;

      void setVariabilityQualifier(VariabilityQualifier qualifier);

      void setIOQualifier(IOQualifier qualifier);

      bool isDiscrete() const;
      bool isParameter() const;
      bool isConstant() const;

      bool isInput() const;
      bool isOutput() const;

		private:
      VariabilityQualifier variabilityQualifier;
      IOQualifier ioQualifier;
	};
}

#endif // MARCO_AST_NODE_TYPEPREFIX_H
