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

Equation& BltBlock::operator[](size_t index)
{
	assert(index < equations.size());
	return equations[index];
}

const Equation& BltBlock::operator[](size_t index) const
{
	assert(index < equations.size());
	return equations[index];
}

void BltBlock::insert(size_t index, Equation equation)
{
	assert(index <= equations.size());
	equations.insert(equations.begin() + index, equation);
}

void BltBlock::erase(size_t index)
{
	assert(index < equations.size());
	equations.erase(equations.begin() + index);
}

size_t BltBlock::equationsCount() const
{
	size_t count = 0;

	for (const Equation& equation : equations)
		count += equation.getInductions().size();

	return count;
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
	return equations.size();
}
