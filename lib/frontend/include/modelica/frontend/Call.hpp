#pragma once

#include <memory>

#include "llvm/ADT/SmallVector.h"
namespace modelica
{
	class Expression;
	class Call
	{
		public:
		using UniqueExpr = std::unique_ptr<Expression>;
		explicit Call(UniqueExpr fun, llvm::SmallVector<UniqueExpr, 3> args = {})
				: function(std::move(fun)), args(std::move(args))
		{
		}

		Call(const Call& other);
		Call(Call&& other) = default;
		Call& operator=(const Call& other);
		Call& operator=(Call&& other) = default;

		~Call() = default;

		[[nodiscard]] const Expression& getFunction() const { return *function; }
		[[nodiscard]] Expression& getFunction() { return *function; }
		[[nodiscard]] size_t argumentsCount() const { return args.size(); }
		[[nodiscard]] const Expression& operator[](size_t index) const
		{
			assert(index <= argumentsCount());
			return *args[index];
		}

		[[nodiscard]] Expression& operator[](size_t index)
		{
			assert(index <= argumentsCount());
			return *args[index];
		}

		[[nodiscard]] bool operator==(const Call& other) const;

		[[nodiscard]] bool operator!=(const Call& other) const
		{
			return !(*this == other);
		}

		private:
		UniqueExpr function;
		llvm::SmallVector<UniqueExpr, 3> args;
	};
}	 // namespace modelica
