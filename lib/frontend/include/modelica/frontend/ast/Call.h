#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>

#include "Expression.h"

namespace modelica::frontend
{
	class ReferenceAccess;

	class Call
			: public impl::ExpressionCRTP<Call>,
				public impl::Cloneable<Call>
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using args_iterator = Container<std::unique_ptr<Expression>>::iterator;
		using args_const_iterator = Container<std::unique_ptr<Expression>>::const_iterator;

		Call(SourcePosition location,
				 std::unique_ptr<ReferenceAccess> function,
				 llvm::ArrayRef<std::unique_ptr<Expression>> args,
				 Type type);

		Call(const Call& other);
		Call(Call&& other);
		~Call() override;

		Call& operator=(const Call& other);
		Call& operator=(Call&& other);

		friend void swap(Call& first, Call& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() == ASTNodeKind::EXPRESSION_CALL;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] bool isLValue() const override;

		[[nodiscard]] bool operator==(const Call& other) const;
		[[nodiscard]] bool operator!=(const Call& other) const;

		[[nodiscard]] Expression* operator[](size_t index);
		[[nodiscard]] const Expression* operator[](size_t index) const;

		[[nodiscard]] ReferenceAccess* getFunction();
		[[nodiscard]] const ReferenceAccess* getFunction() const;

		[[nodiscard]] size_t argumentsCount() const;

		[[nodiscard]] args_iterator begin();
		[[nodiscard]] args_const_iterator begin() const;

		[[nodiscard]] args_iterator end();
		[[nodiscard]] args_const_iterator end() const;

		private:
		std::unique_ptr<ReferenceAccess> function;
		Container<std::unique_ptr<Expression>> args;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Call& obj);

	std::string toString(const Call& obj);
}
