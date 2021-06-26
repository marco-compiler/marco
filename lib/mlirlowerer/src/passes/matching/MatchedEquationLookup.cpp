#include <marco/mlirlowerer/passes/matching/MatchedEquationLookup.h>

using namespace marco;
using namespace codegen::model;

IndexesOfEquation::IndexesOfEquation(const Model& model, Equation equation)
		: content(equation),
			equations({ equation }),
			accesses({ equation.getDeterminedVariable() }),
			variables({ model.getVariable(accesses.front().getVar()) }),
			directAccesses({ accesses.front().getAccess() }),
			invertedAccesses({accesses.front().getAccess().invert() }),
			indexSets({ accesses.front().getAccess().map(equation.getInductions()) })
{
}

IndexesOfEquation::IndexesOfEquation(
		const Model& model, BltBlock bltBlock)
		: content(bltBlock), equations(bltBlock.getEquations())
{
	for (Equation& eq : bltBlock.getEquations())
	{
		accesses.push_back(eq.getDeterminedVariable());
		variables.push_back(model.getVariable(accesses.back().getVar()));
		directAccesses.push_back(accesses.back().getAccess());
		invertedAccesses.push_back(accesses.back().getAccess().invert());
		indexSets.push_back(directAccesses.back().map(eq.getInductions()));
	}
}

bool IndexesOfEquation::isEquation() const
{
	return std::holds_alternative<Equation>(content);
}

bool IndexesOfEquation::isBltBlock() const
{
	return std::holds_alternative<BltBlock>(content);
}

const std::variant<Equation, BltBlock>& IndexesOfEquation::getContent() const
{
	return content;
}

const Equation& IndexesOfEquation::getEquation() const
{
	assert(isEquation());
	return std::get<Equation>(content);
}

const BltBlock& IndexesOfEquation::getBltBlock() const
{
	assert(isBltBlock());
	return std::get<BltBlock>(content);
}

const llvm::SmallVector<Equation, 3>& IndexesOfEquation::getEquations() const
{
	return equations;
}

const llvm::SmallVector<Variable, 3>& IndexesOfEquation::getVariables() const
{
	return variables;
}

const llvm::SmallVector<VectorAccess, 3>& IndexesOfEquation::getEqToVars() const
{
	return directAccesses;
}

const llvm::SmallVector<VectorAccess, 3>& IndexesOfEquation::getVarToEqs() const
{
	return invertedAccesses;
}

const llvm::SmallVector<MultiDimInterval, 3>& IndexesOfEquation::getIntervals() const
{
	return indexSets;
}

const Variable& IndexesOfEquation::getVariable() const
{
	assert(isEquation());
	return variables.front();
}

const VectorAccess& IndexesOfEquation::getEqToVar() const
{
	assert(isEquation());
	return directAccesses.front();
}

const VectorAccess& IndexesOfEquation::getVarToEq() const
{
	assert(isEquation());
	return invertedAccesses.front();
}

const MultiDimInterval& IndexesOfEquation::getInterval() const
{
	assert(isEquation());
	return indexSets.front();
}

size_t IndexesOfEquation::size() const
{
	return equations.size();
}

MatchedEquationLookup::MatchedEquationLookup(const Model& model)
{
	for (const Equation& equation : model.getEquations())
		addEquation(equation, model);
	for (const BltBlock& bltBlock : model.getBltBlocks())
		addBltBlock(bltBlock, model);
}

MatchedEquationLookup::MatchedEquationLookup(const Model& model, llvm::ArrayRef<Equation> equations)
{
	assert(model.getBltBlocks().empty());
	for (const Equation& equation : equations)
		addEquation(equation, model);
}

void MatchedEquationLookup::addEquation(Equation equation, const Model& model)
{
	IndexesOfEquation* index = new IndexesOfEquation(model, equation);
	const Variable& var = index->getVariable();
	variables.emplace(var, index);
}

void MatchedEquationLookup::addBltBlock(BltBlock bltBlock, const Model& model)
{
	IndexesOfEquation* index = new IndexesOfEquation(model, bltBlock);
	for (const Variable& var : index->getVariables())
		variables.emplace(var, index);
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
