#pragma once

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/frontend/Equation.hpp"
#include "modelica/frontend/Expression.hpp"

namespace modelica
{
	/**
	 * an induction is the in memory of a piece of code such as
	 * for i in 0:20. and induction holds a name and the begin and end
	 * expressions.
	 *
	 * Notice that for the compiler we made the assumption that all range will be
	 * of step one.
	 */
	class Induction
	{
		public:
		explicit Induction(std::string indVar, Expression begin, Expression end)
				: begin(std::move(begin)),
					end(std::move(end)),
					inductionIndex(0),
					inductionVar(std::move(indVar))

		{
		}

		[[nodiscard]] const std::string& getName() const { return inductionVar; }
		[[nodiscard]] const Expression& getBegin() const { return begin; }
		[[nodiscard]] const Expression& getEnd() const { return end; }

		[[nodiscard]] Expression& getBegin() { return begin; }
		[[nodiscard]] Expression& getEnd() { return end; }

		void dump(llvm::raw_ostream& OS = llvm::outs(), size_t indents = 0) const;

		[[nodiscard]] size_t getInductionIndex() const { return inductionIndex; }
		void setInductionIndex(size_t index) { inductionIndex = index; }

		private:
		Expression begin;
		Expression end;
		size_t inductionIndex;
		std::string inductionVar;
	};

	/**
	 * For equations are different with respect to regular equations
	 * because they introduce a set of inductions, and thus a new set of names
	 * avialable withing the for cycle.
	 *
	 * Inductions are mapped to a set of indicies so that an from a name we can
	 * deduce a index and from a index we can deduce a name
	 */
	class ForEquation
	{
		public:
		ForEquation(llvm::SmallVector<Induction, 3> ind, Equation eq);

		[[nodiscard]] const auto& getInductions() const { return induction; }
		[[nodiscard]] size_t inductionsCount() const { return induction.size(); }
		[[nodiscard]] auto& getInductions() { return induction; }

		[[nodiscard]] Equation& getEquation() { return equation; }
		[[nodiscard]] const Equation& getEquation() const { return equation; }

		void dump(llvm::raw_ostream& OS = llvm::outs(), size_t indents = 0) const;

		private:
		llvm::SmallVector<Induction, 3> induction;
		Equation equation;
	};
}	 // namespace modelica
