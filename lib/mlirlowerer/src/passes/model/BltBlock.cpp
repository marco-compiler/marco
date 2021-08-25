#include <marco/mlirlowerer/passes/model/BltBlock.h>

using namespace marco::codegen::model;

BltBlock::BltBlock(llvm::ArrayRef<Equation> equations)
		: equations(equations.begin(), equations.end()), isForwardDirection(true)
{
}

BltBlock::Container<Equation>& BltBlock::getEquations()
{
	return equations;
}

const BltBlock::Container<Equation>& BltBlock::getEquations() const
{
	return equations;
}

size_t BltBlock::equationsCount() const
{
	size_t count = 0;

	for (const Equation& equation : equations)
		count += equation.getInductions().size();

	return count;
}

void BltBlock::addEquation(Equation equation)
{
	equations.push_back(equation);
}

bool BltBlock::isForward() const
{
	return isForwardDirection;
}

void BltBlock::setForward(bool isForward)
{
	isForwardDirection = isForward;
	for (Equation& equation : equations)
		equation.setForward(isForward);
}

size_t BltBlock::size() const
{
	size_t size = 0;

	for (const Equation& equation : equations)
		size += equation.getInductions().size();

	return size;
}
