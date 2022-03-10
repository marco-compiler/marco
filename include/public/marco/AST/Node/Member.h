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

		[[nodiscard]] bool operator==(const Member& other) const;
		[[nodiscard]] bool operator!=(const Member& other) const;

		[[nodiscard]] llvm::StringRef getName() const;

		[[nodiscard]] Type& getType();
		[[nodiscard]] const Type& getType() const;

		[[nodiscard]] bool hasInitializer() const;
		[[nodiscard]] Expression* getInitializer();
		[[nodiscard]] const Expression* getInitializer() const;

		[[nodiscard]] bool hasStartOverload() const;
		[[nodiscard]] Expression* getStartOverload();
		[[nodiscard]] const Expression* getStartOverload() const;

		[[nodiscard]] bool isPublic() const;
		[[nodiscard]] bool isParameter() const;
		[[nodiscard]] bool isInput() const;
		[[nodiscard]] bool isOutput() const;

		private:
		Member(
				SourceRange location,
				llvm::StringRef name,
				Type tp,
				TypePrefix prefix,
				llvm::Optional<std::unique_ptr<Expression>> initializer = llvm::None,
				bool isPublic = true,
				llvm::Optional<std::unique_ptr<Expression>> startOverload = llvm::None);

		std::string name;
		Type type;
		TypePrefix typePrefix;
	  llvm::Optional<std::unique_ptr<Expression>> initializer;
		bool isPublicMember;
		llvm::Optional<std::unique_ptr<Expression>> startOverload;
	};
}

#endif // MARCO_AST_NODE_MEMBER_H
