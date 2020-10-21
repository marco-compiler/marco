#pragma once

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
		Statement(llvm::ArrayRef<Expression> destinations, Expression expression);

		void dump(llvm::raw_ostream& os, size_t indents) const;

		[[nodiscard]] llvm::SmallVectorImpl<Expression>& getDestinations();
		[[nodiscard]] Expression& getExpression();

		[[nodiscard]] const llvm::SmallVectorImpl<Expression>& getDestinations()
				const;
		[[nodiscard]] const Expression& getExpression() const;

		private:
		// Where the result of the expression has to be stored.
		// A vector is needed because functions may have multiple outputs.
		llvm::SmallVector<Expression, 3> destinations;

		// Right-hand side expression of the assignment
		Expression expression;
	};
}	 // namespace modelica
