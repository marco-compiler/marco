#pragma once

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/frontend/Equation.hpp"
#include "modelica/frontend/Expression.hpp"

namespace modelica
{
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

		void dump(llvm::raw_ostream& OS = llvm::outs(), size_t indents = 0) const
		{
			OS.indent(indents);
			OS << "induction var " << inductionVar << "\n";

			OS.indent(indents);
			OS << "from ";
			begin.dump(OS, indents + 1);
			OS << "\n";
			OS.indent(indents);
			OS << "to";
			end.dump(OS, indents + 1);
		}

		[[nodiscard]] size_t getInductionIndex() const { return inductionIndex; }
		void setInductionIndex(size_t index) { inductionIndex = index; }

		private:
		Expression begin;
		Expression end;
		size_t inductionIndex;
		std::string inductionVar;
	};

	class ForEquation
	{
		public:
		ForEquation(llvm::SmallVector<Induction, 3> ind, Equation eq)
				: induction(std::move(ind)), equation(std::move(eq))
		{
			for (size_t a = 0; a < induction.size(); a++)
				induction[a].setInductionIndex(a);
		}

		[[nodiscard]] const auto& getInductions() const { return induction; }
		[[nodiscard]] size_t inductionsCount() const { return induction.size(); }
		[[nodiscard]] auto& getInductions() { return induction; }

		[[nodiscard]] Equation& getEquation() { return equation; }
		[[nodiscard]] const Equation& getEquation() const { return equation; }

		void dump(llvm::raw_ostream& OS = llvm::outs(), size_t indents = 0) const
		{
			OS << "for equation\n";
			for (const auto& ind : induction)
			{
				ind.dump(OS, indents + 1);
				OS << "\n";
			}
			equation.dump(OS, indents + 1);
		}

		private:
		llvm::SmallVector<Induction, 3> induction;
		Equation equation;
	};
}	 // namespace modelica
