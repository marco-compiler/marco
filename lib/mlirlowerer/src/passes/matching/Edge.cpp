#include <marco/mlirlowerer/passes/matching/Edge.h>
#include <marco/mlirlowerer/passes/model/Equation.h>

using namespace marco::codegen::model;

Edge::Edge(
		const Equation equation,
		const Variable variable,
		VectorAccess vectorAccess,
		ExpressionPath access,
		size_t index,
		size_t eqDesc,
		size_t varDesc)
		: equation(std::move(equation)),
			variable(std::move(variable)),
			vectorAccess(vectorAccess),
			index(index),
			pathToExp(std::move(access)),
			eqDesc(eqDesc),
			varDesc(varDesc)
{
	if (this->equation.isForLoop())
		invertedAccess = vectorAccess.invert();
}

Equation Edge::getEquation() const
{
	return equation;
}

Variable Edge::getVariable() const
{
	return variable;
}

const VectorAccess& Edge::getVectorAccess() const
{
	return vectorAccess;
}

const VectorAccess& Edge::getInvertedAccess() const
{
	return invertedAccess;
}

marco::IndexSet& Edge::getSet()
{
	return set;
}

const marco::IndexSet& Edge::getSet() const
{
	return set;
}

marco::IndexSet Edge::map(const IndexSet& currentSet) const
{
	return vectorAccess.map(currentSet);
}

marco::IndexSet Edge::invertMap(const IndexSet& currentSet) const
{
	return invertedAccess.map(currentSet);
}

bool Edge::empty() const
{
	return set.empty();
}

size_t Edge::getIndex() const
{
	return index;
}

ExpressionPath& Edge::getPath()
{
	return pathToExp;
}

const ExpressionPath& Edge::getPath() const
{
	return pathToExp;
}

void Edge::dump(llvm::raw_ostream& OS) const
{
	OS << "EDGE: Eq " << eqDesc << " to Var " << varDesc;
	OS << "\n";
	OS << "\tForward Map: ";
	vectorAccess.dump(OS);
	OS << " -> Backward Map: ";
	invertedAccess.dump(OS);
	OS << "\n\tCurrent Flow: ";
	set.dump(OS);
	OS << "\n";
}
