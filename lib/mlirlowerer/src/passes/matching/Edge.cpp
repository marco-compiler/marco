#include <modelica/mlirlowerer/passes/matching/Edge.h>
#include <modelica/mlirlowerer/passes/model/Equation.h>

using namespace modelica::codegen::model;

Edge::Edge(const Equation& eq,
					 const Variable& var,
					 VectorAccess vAccess,
					 ExpressionPath access,
					 size_t index)
		: vectorAccess(std::move(vAccess)),
			equation(&eq),
			variable(&var),
			pathToExp(std::move(access)),
			index(index)
{
	if (eq.isForEquation())
		invertedAccess = vectorAccess.invert();
}

const Equation& Edge::getEquation() const
{
	return *equation;
}

const Variable& Edge::getVariable() const
{
	return *variable;
}

const VectorAccess& Edge::getVectorAccess() const
{
	return vectorAccess;
}

const VectorAccess& Edge::getInvertedAccess() const
{
	return invertedAccess;
}

modelica::IndexSet& Edge::getSet()
{
	return set;
}

const modelica::IndexSet& Edge::getSet() const
{
	return set;
}

modelica::IndexSet Edge::map(const IndexSet& set) const
{
	return vectorAccess.map(set);
}

modelica::IndexSet Edge::invertMap(const IndexSet& set) const
{
	return invertedAccess.map(set);
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
