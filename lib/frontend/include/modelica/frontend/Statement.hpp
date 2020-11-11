#pragma once

#include <iterator>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <vector>

#include "Expression.hpp"

namespace modelica
{
	class Statement
	{
		public:
		Statement(Expression destination, Expression expression);
		Statement(
				std::initializer_list<Expression> destinations, Expression expression);

		template<typename Iter>
		Statement(
				Iter destinationsBegin, Iter destinationsEnd, Expression expression)
				: destination(Tuple(destinationsBegin, destinationsEnd)),
					expression(std::move(expression))
		{
		}

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents) const;

		[[nodiscard]] std::vector<Expression*> getDestinations();

		[[nodiscard]] Expression& getExpression();
		[[nodiscard]] const Expression& getExpression() const;

		private:
		template<typename T>
		[[nodiscard]] bool destinationIsA() const
		{
			return std::holds_alternative<T>(destination);
		}

		template<typename T>
		[[nodiscard]] T& getDestination()
		{
			assert(destinationIsA<T>());
			return std::get<T>(destination);
		}

		// Where the result of the expression has to be stored.
		// A vector is needed because functions may have multiple outputs.
		std::variant<Expression, Tuple> destination;

		// Right-hand side expression of the assignment
		Expression expression;
	};
}	 // namespace modelica
