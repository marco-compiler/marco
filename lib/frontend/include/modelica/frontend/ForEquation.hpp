#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <modelica/frontend/Equation.hpp>
#include <modelica/frontend/Expression.hpp>
#include <modelica/frontend/Induction.hpp>

namespace modelica
{
	/**
	 * For equations are different with respect to regular equations
	 * because they introduce a set of inductions, and thus a new set of names
	 * available withing the for cycle.
	 *
	 * Inductions are mapped to a set of indexes so that an from a name we can
	 * deduce a index and from a index we can deduce a name
	 */
	class ForEquation
	{
		public:
		ForEquation(llvm::ArrayRef<Induction> inductions, Equation equation);

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] llvm::SmallVectorImpl<Induction>& getInductions();
		[[nodiscard]] const llvm::SmallVectorImpl<Induction>& getInductions() const;
		[[nodiscard]] size_t inductionsCount() const;

		[[nodiscard]] Equation& getEquation();
		[[nodiscard]] const Equation& getEquation() const;

		private:
		llvm::SmallVector<Induction, 3> inductions;
		Equation equation;
	};
}	 // namespace modelica
