#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>

#include "ASTNode.h"
#include "Type.h"

namespace modelica::frontend
{
	class Expression;
	class ReferenceAccess;

	class Call
			: public ASTNode,
				public impl::Dumpable<Call>
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using args_iterator = Container<std::unique_ptr<Expression>>::iterator;
		using args_const_iterator = Container<std::unique_ptr<Expression>>::const_iterator;

		Call(const Call& other);
		Call(Call&& other);
		~Call() override;

		Call& operator=(const Call& other);
		Call& operator=(Call&& other);

		friend void swap(Call& first, Call& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] bool isLValue() const;

		[[nodiscard]] bool operator==(const Call& other) const;
		[[nodiscard]] bool operator!=(const Call& other) const;

		[[nodiscard]] Expression* operator[](size_t index);
		[[nodiscard]] const Expression* operator[](size_t index) const;

		[[nodiscard]] Type& getType();
		[[nodiscard]] const Type& getType() const;
		void setType(Type tp);

		[[nodiscard]] Expression* getFunction();
		[[nodiscard]] const Expression* getFunction() const;

		[[nodiscard]] Expression* getArg(size_t index);
		[[nodiscard]] const Expression* getArg(size_t index) const;

		[[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Expression>> getArgs();
		[[nodiscard]] llvm::ArrayRef<std::unique_ptr<Expression>> getArgs() const;

		[[nodiscard]] size_t argumentsCount() const;

		[[nodiscard]] args_iterator begin();
		[[nodiscard]] args_const_iterator begin() const;

		[[nodiscard]] args_iterator end();
		[[nodiscard]] args_const_iterator end() const;

		private:
		friend class Expression;

		Call(SourceRange location,
				 Type type,
				 std::unique_ptr<Expression> function,
				 llvm::ArrayRef<std::unique_ptr<Expression>> args);

		Type type;
		std::unique_ptr<Expression> function;
		Container<std::unique_ptr<Expression>> args;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Call& obj);

	std::string toString(const Call& obj);
}
