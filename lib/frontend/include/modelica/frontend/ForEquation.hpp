#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>

#include "Equation.hpp"
#include "Expression.hpp"

namespace modelica
{
	/**
	 * An induction is the in memory of a piece of code such as
	 * for i in 0:20. and induction holds a name and the begin and end
	 * expressions.
	 *
	 * Notice that for the compiler we made the assumption that all range will be
	 * of step one.
	 */
	class Induction
	{
		public:
		Induction(std::string indVar, Expression begin, Expression end);

		void dump() const;
		void dump(llvm::raw_ostream& os = llvm::outs(), size_t indents = 0) const;

		[[nodiscard]] const std::string& getName() const;

		[[nodiscard]] Expression& getBegin();
		[[nodiscard]] const Expression& getBegin() const;

		[[nodiscard]] Expression& getEnd();
		[[nodiscard]] const Expression& getEnd() const;

		[[nodiscard]] size_t getInductionIndex() const;
		void setInductionIndex(size_t index);

		private:
		Expression begin;
		Expression end;
		size_t inductionIndex;
		std::string inductionVar;
	};

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
		ForEquation(llvm::ArrayRef<Induction> ind, Equation eq);

		void dump() const;
		void dump(llvm::raw_ostream& os = llvm::outs(), size_t indents = 0) const;

		[[nodiscard]] llvm::SmallVectorImpl<Induction>& getInductions();
		[[nodiscard]] const llvm::SmallVectorImpl<Induction>& getInductions() const;
		[[nodiscard]] size_t inductionsCount() const;

		[[nodiscard]] Equation& getEquation();
		[[nodiscard]] const Equation& getEquation() const;

		private:
		llvm::SmallVector<Induction, 3> induction;
		Equation equation;
	};
}	 // namespace modelica
