#include <modelica/frontend/ForEquation.hpp>
#include <modelica/utils/IRange.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

ForEquation::ForEquation(ArrayRef<Induction> ind, Equation eq)
		: induction(iterator_range<ArrayRef<Induction>::iterator>(move(ind))),
			equation(move(eq))
{
	for (auto a : irange(induction.size()))
		induction[a].setInductionIndex(a);
}

void ForEquation::dump() const { dump(outs(), 0); }

void ForEquation::dump(llvm::raw_ostream& os, size_t indents) const
{
	os << "for equation\n";

	for (const auto& ind : induction)
	{
		ind.dump(os, indents + 1);
		os << "\n";
	}

	equation.dump(os, indents + 1);
}

SmallVectorImpl<Induction>& ForEquation::getInductions() { return induction; }

const SmallVectorImpl<Induction>& ForEquation::getInductions() const
{
	return induction;
}

size_t ForEquation::inductionsCount() const { return induction.size(); }

Equation& ForEquation::getEquation() { return equation; }

const Equation& ForEquation::getEquation() const { return equation; }
