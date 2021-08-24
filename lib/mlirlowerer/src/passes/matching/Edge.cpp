#include <marco/mlirlowerer/passes/matching/Edge.h>
#include <marco/mlirlowerer/passes/model/Equation.h>

using namespace marco::codegen::model;

Edge::Edge(Equation equation,
					 Variable variable,
					 VectorAccess vectorAccess,
					 ExpressionPath access,
					 size_t index)
		: equation(std::move(equation)),
			variable(std::move(variable)),
			vectorAccess(vectorAccess),
			invertedAccess(vectorAccess.invert()),
			index(index),
			pathToExp(std::move(access))
{
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

marco::IndexSet Edge::map(const IndexSet& set) const
{
	return vectorAccess.map(set);
}

marco::IndexSet Edge::invertMap(const IndexSet& set) const
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
