#include "marco/matching/MatchedEquationLookup.hpp"

using namespace marco;
using namespace std;
using namespace llvm;

IndexesOfEquation::IndexesOfEquation(
		const Model& model, const ModEquation& equation)
		: content(equation),
			equations({ equation }),
			accesses({ equation.getDeterminedVariable() }),
			variables({ &model.getVar(accesses.front().getVarName()) }),
			directAccesses({ accesses.front().getAccess() }),
			invertedAccesses({ accesses.front().getAccess().invert() }),
			indexSets({ accesses.front().getAccess().map(equation.getInductions()) })
{
}

IndexesOfEquation::IndexesOfEquation(
		const Model& model, const ModBltBlock& bltBlock)
		: content(bltBlock), equations(bltBlock.getEquations())
{
	for (const ModEquation& eq : bltBlock.getEquations())
	{
		accesses.push_back(eq.getDeterminedVariable());
		variables.push_back(&model.getVar(accesses.back().getVarName()));
		directAccesses.push_back(accesses.back().getAccess());
		invertedAccesses.push_back(accesses.back().getAccess().invert());
		indexSets.push_back(directAccesses.back().map(eq.getInductions()));
	}
}

MatchedEquationLookup::MatchedEquationLookup(const Model& model)
{
	for (const ModEquation& equation : model)
		addEquation(equation, model);
	for (const ModBltBlock& bltBlock : model.getBltBlocks())
		addBltBlock(bltBlock, model);
}

MatchedEquationLookup::MatchedEquationLookup(
		const Model& model, ArrayRef<ModEquation> equs)
{
	assert(model.getBltBlocks().empty());
	for (const ModEquation& equation : equs)
		addEquation(equation, model);
}

void MatchedEquationLookup::addEquation(
		const ModEquation& equation, const Model& model)
{
	IndexesOfEquation* index = new IndexesOfEquation(model, equation);
	const ModVariable* var = index->getVariable();
	variables.emplace(var, index);
}

void MatchedEquationLookup::addBltBlock(
		const ModBltBlock& bltBlock, const Model& model)
{
	IndexesOfEquation* index = new IndexesOfEquation(model, bltBlock);
	for (const ModVariable* var : index->getVariables())
		variables.emplace(var, index);
}
