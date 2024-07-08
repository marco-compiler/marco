#ifndef MARCO_AST_NODE_MEMBER_H
#define MARCO_AST_NODE_MEMBER_H

#include "marco/AST/Node/ASTNode.h"
#include "marco/AST/Node/TypePrefix.h"
#include <optional>
#include <string>

namespace marco::ast
{
  class Expression;
  class Modification;
  class VariableType;

	class Member : public ASTNode
	{
		public:
      explicit Member(SourceRange location);

      Member(const Member& other);

      ~Member() override;

      static bool classof(const ASTNode* node)
      {
        return node->getKind() == ASTNode::Kind::Member;
      }

      std::unique_ptr<ASTNode> clone() const override;

      llvm::json::Value toJSON() const override;

      llvm::StringRef getName() const;

      void setName(llvm::StringRef newName);

      VariableType* getType();

      const VariableType* getType() const;

      void setType(std::unique_ptr<ASTNode> node);

      TypePrefix* getTypePrefix();

      const TypePrefix* getTypePrefix() const;

      void setTypePrefix(std::unique_ptr<ASTNode> node);

      bool isPublic() const;

      void setPublic(bool value);

      bool isDiscrete() const;
      bool isParameter() const;
      bool isConstant() const;

      bool isInput() const;
      bool isOutput() const;

      bool hasModification() const;

      Modification* getModification();

      const Modification* getModification() const;

      void setModification(std::unique_ptr<ASTNode> node);

      /// @name Modification properties
      /// {

      bool hasExpression() const;

      Expression* getExpression();

      const Expression* getExpression() const;

      bool hasStartExpression() const;

      Expression* getStartExpression();

      const Expression* getStartExpression() const;

      std::optional<bool> getFixedProperty() const;

      bool getEachProperty() const;

      /// }

    private:
      std::string name;
      std::unique_ptr<ASTNode> type;
      std::unique_ptr<ASTNode> typePrefix;
      bool isPublicMember;
      std::unique_ptr<ASTNode> modification;
	};
}

#endif // MARCO_AST_NODE_MEMBER_H
