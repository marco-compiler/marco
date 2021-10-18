#include <marco/mlirlowerer/passes/matching/MatchedEquationLookup.h>

using namespace marco;
using namespace codegen::model;

IndexesOfEquation::IndexesOfEquation(const Model& model, Equation equation)
		: equation(equation),
			access(equation.getDeterminedVariable()),
			invertedAccess(access.getAccess().invert()),
			indexSet(access.getAccess().map(equation.getInductions())),
			variable(model.getVariable(access.getVar()))
{
}

const Equation& IndexesOfEquation::getEquation() const
{
	return equation;
}

const Variable& IndexesOfEquation::getVariable() const
{
	return variable;
}

const VectorAccess& IndexesOfEquation::getEqToVar() const
{
	return access.getAccess();
}

const VectorAccess& IndexesOfEquation::getVarToEq() const
{
	return invertedAccess;
}

const MultiDimInterval& IndexesOfEquation::getInterval() const
{
	return indexSet;
}

MatchedEquationLookup::MatchedEquationLookup(const Model& model)
{
	for (const auto& equation : model.getEquations())
		addEquation(equation, model);
}

MatchedEquationLookup::MatchedEquationLookup(const Model& model, llvm::ArrayRef<Equation> equations)
{
	for (const auto& equation : equations)
		addEquation(equation, model);
}

void MatchedEquationLookup::addEquation(Equation equation, const Model& model)
{
	IndexesOfEquation index(model, equation);
	Variable var = index.getVariable();
	variables.emplace(var, std::move(index));
}

MatchedEquationLookup::iterator_range MatchedEquationLookup::eqsDeterminingVar(const Variable& var)
{
	auto range = variables.equal_range(var);
	return llvm::make_range(range.first, range.second);
}

MatchedEquationLookup::const_iterator_range MatchedEquationLookup::eqsDeterminingVar(const Variable& var) const
{
	auto range = variables.equal_range(var);
	return llvm::make_range(range.first, range.second);
}

MatchedEquationLookup::iterator MatchedEquationLookup::begin()
{
	return variables.begin();
}

MatchedEquationLookup::const_iterator MatchedEquationLookup::begin() const
{
	return variables.begin();
}

MatchedEquationLookup::iterator MatchedEquationLookup::end()
{
	return variables.end();
}

MatchedEquationLookup::const_iterator MatchedEquationLookup::end() const
{
	return variables.end();
}
