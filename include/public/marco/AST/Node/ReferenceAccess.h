#ifndef MARCO_AST_NODE_REFERENCEACCESS_H
#define MARCO_AST_NODE_REFERENCEACCESS_H

#include "marco/AST/Node/Expression.h"
#include "marco/AST/Node/Type.h"
#include <string>

namespace marco::ast
{
	/// A reference access is pretty much any use of a variable at the moment.
	class ReferenceAccess : public Expression
	{
		public:
      explicit ReferenceAccess(SourceRange location);

      ReferenceAccess(const ReferenceAccess& other);

      ~ReferenceAccess() override;

      static bool classof(const ASTNode* node)
      {
        return node->getKind() == ASTNode::Kind::Expression_ReferenceAccess;
      }

      std::unique_ptr<ASTNode> clone() const override;

      llvm::json::Value toJSON() const override;

      bool isLValue() const override;

      llvm::StringRef getName() const;

      void setName(llvm::StringRef name);

      bool isDummy() const;

      void setDummy(bool value);

    private:
      std::string name;
      bool dummy;
	};
}

#endif // MARCO_AST_NODE_REFERENCEACCESS_H
