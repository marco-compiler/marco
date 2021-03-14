#include <modelica/frontend/AST.h>
#include <modelica/utils/IRange.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

ForEquation::ForEquation(ArrayRef<Induction> inductions, Equation equation)
		: inductions(inductions.begin(), inductions.end()), equation(move(equation))
{
	for (auto i : irange(inductions.size()))
		this->inductions[i].setInductionIndex(i);
}

void ForEquation::dump() const { dump(outs(), 0); }

void ForEquation::dump(llvm::raw_ostream& os, size_t indents) const
{
	os << "for equation\n";

	for (const auto& ind : inductions)
	{
		ind.dump(os, indents + 1);
		os << "\n";
	}

	equation.dump(os, indents + 1);
}

SmallVectorImpl<Induction>& ForEquation::getInductions() { return inductions; }

const SmallVectorImpl<Induction>& ForEquation::getInductions() const
{
	return inductions;
}

size_t ForEquation::inductionsCount() const { return inductions.size(); }

Equation& ForEquation::getEquation() { return equation; }

const Equation& ForEquation::getEquation() const { return equation; }
