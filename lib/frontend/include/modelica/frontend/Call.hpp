#pragma once

#include <boost/iterator/indirect_iterator.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>

namespace modelica
{
	class Expression;

	class Call
	{
		private:
		using UniqueExpression = std::unique_ptr<Expression>;
		using Container = llvm::SmallVector<UniqueExpression, 3>;

		public:
		using args_iterator = boost::indirect_iterator<Container::iterator>;
		using args_const_iterator = boost::indirect_iterator<Container::const_iterator>;

		Call(Expression fun, llvm::ArrayRef<Expression> args = {});
		Call(const Call& other);
		Call(Call&& other) = default;

		Call& operator=(const Call& other);
		Call& operator=(Call&& other) = default;

		~Call() = default;

		[[nodiscard]] bool operator==(const Call& other) const;
		[[nodiscard]] bool operator!=(const Call& other) const;

		[[nodiscard]] Expression& operator[](size_t index);
		[[nodiscard]] const Expression& operator[](size_t index) const;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] Expression& getFunction();
		[[nodiscard]] const Expression& getFunction() const;

		[[nodiscard]] size_t argumentsCount() const;

		[[nodiscard]] args_iterator begin();
		[[nodiscard]] args_const_iterator begin() const;

		[[nodiscard]] args_iterator end();
		[[nodiscard]] args_const_iterator end() const;

		private:
		UniqueExpression function;
		Container args;
	};
}	 // namespace modelica
