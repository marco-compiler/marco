#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>

namespace modelica
{
	class Expression;

	class Call
	{
		public:
		using UniqueExpr = std::unique_ptr<Expression>;

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

		[[nodiscard]] llvm::SmallVectorImpl<UniqueExpr>::iterator begin();
		[[nodiscard]] llvm::SmallVectorImpl<UniqueExpr>::const_iterator begin()
				const;

		[[nodiscard]] llvm::SmallVectorImpl<UniqueExpr>::iterator end();
		[[nodiscard]] llvm::SmallVectorImpl<UniqueExpr>::const_iterator end() const;

		private:
		UniqueExpr function;
		llvm::SmallVector<UniqueExpr, 3> args;
	};
}	 // namespace modelica
