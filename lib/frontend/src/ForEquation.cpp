#include "modelica/frontend/ForEquation.hpp"

#include "modelica/utils/IRange.hpp"

using namespace llvm;
using namespace std;
using namespace modelica;

void Induction::dump(llvm::raw_ostream& OS, size_t indents) const
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

void ForEquation::dump(llvm::raw_ostream& OS, size_t indents) const
{
	OS << "for equation\n";
	for (const auto& ind : induction)
	{
		ind.dump(OS, indents + 1);
		OS << "\n";
	}
	equation.dump(OS, indents + 1);
}

ForEquation::ForEquation(llvm::SmallVector<Induction, 3> ind, Equation eq)
		: induction(std::move(ind)), equation(std::move(eq))
{
	for (auto a : irange(induction.size()))
		induction[a].setInductionIndex(a);
}
