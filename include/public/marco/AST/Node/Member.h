#ifndef MARCO_AST_NODE_MEMBER_H
#define MARCO_AST_NODE_MEMBER_H

#include "marco/AST/Node/ASTNode.h"
#include "marco/AST/Node/Type.h"
#include "marco/AST/Node/TypePrefix.h"
#include <optional>
#include <string>

namespace marco::ast
{
  class Expression;
  class Modification;

	class Member
			: public ASTNode,
				public impl::Cloneable<Member>,
				public impl::Dumpable<Member>
	{
		public:
      template<typename... Args>
      static std::unique_ptr<Member> build(Args&&... args)
      {
        return std::unique_ptr<Member>(new Member(std::forward<Args>(args)...));
      }

      Member(const Member& other);
      Member(Member&& other);

      ~Member() override;

      Member& operator=(const Member& other);
      Member& operator=(Member&& other);

      friend void swap(Member& first, Member& second);

      void print(llvm::raw_ostream& os, size_t indents = 0) const override;

      bool operator==(const Member& other) const;
      bool operator!=(const Member& other) const;

      llvm::StringRef getName() const;

      Type& getType();
      const Type& getType() const;

      void setType(Type new_type);

      TypePrefix getTypePrefix() const;

      bool isPublic() const;
      bool isParameter() const;
      bool isInput() const;
      bool isOutput() const;

      bool hasModification() const;
      Modification* getModification();
      const Modification* getModification() const;

      /// @name Modification properties
      /// {

      bool hasExpression() const;

      Expression* getExpression();

      const Expression* getExpression() const;

      bool hasStartExpression() const;

      Expression* getStartExpression();

      const Expression* getStartExpression() const;

      bool getFixedProperty() const;

      bool getEachProperty() const;

      /// }

		private:
      Member(
          SourceRange location,
          llvm::StringRef name,
          Type type,
          TypePrefix typePrefix,
          bool isPublic = true,
          std::unique_ptr<Modification> modification = nullptr);

    private:
      std::string name;
      Type type;
      TypePrefix typePrefix;
      bool isPublicMember;
      std::unique_ptr<Modification> modification;
	};
}

#endif // MARCO_AST_NODE_MEMBER_H
