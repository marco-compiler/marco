#ifndef MARCO_AST_NODE_COMPONENTREFERENCE_H
#define MARCO_AST_NODE_COMPONENTREFERENCE_H

#include "marco/AST/Node/Expression.h"
#include "marco/AST/Node/Type.h"
#include <string>

namespace marco::ast
{
  class ComponentReferenceEntry;

	class ComponentReference : public Expression
	{
		public:
      explicit ComponentReference(SourceRange location);

      ComponentReference(const ComponentReference& other);

      ~ComponentReference() override;

      static bool classof(const ASTNode* node)
      {
        return node->getKind() == ASTNode::Kind::Expression_ComponentReference;
      }

      std::unique_ptr<ASTNode> clone() const override;

      llvm::json::Value toJSON() const override;

      bool isLValue() const override;

      bool isDummy() const;

      void setDummy(bool value);

      bool isGlobalLookup() const;

      void setGlobalLookup(bool global);

      size_t getPathLength() const;

      ComponentReferenceEntry* getElement(size_t index);

      const ComponentReferenceEntry* getElement(size_t index) const;

      void setPath(llvm::ArrayRef<std::unique_ptr<ASTNode>> newPath);

      std::string getName() const;

    private:
      bool dummy;
      bool globalLookup;
      llvm::SmallVector<std::unique_ptr<ASTNode>> path;
	};
}

#endif // MARCO_AST_NODE_COMPONENTREFERENCE_H
