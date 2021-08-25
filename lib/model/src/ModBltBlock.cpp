#include "marco/model/ModBltBlock.hpp"

#include "marco/model/SymbolicDifferentiation.hpp"

using namespace std;
using namespace marco;
using namespace llvm;

/**
 * This constructor initialize the BLT block with the equations it contains and
 * the variable matched to those equation. It then compute the residual function
 * and the jacobian matrix.
 */
ModBltBlock::ModBltBlock(SmallVector<ModEquation, 3> equs, string bltName)
		: equations(std::move(equs))
{
	for (const ModEquation& eq : equations)
		addTemplate(eq);
	computeResidualFunction();
	computeJacobianMatrix();
	body = make_shared<ModBltTemplate>(equations, bltName);
}

void ModBltBlock::addTemplate(const ModEquation& eq)
{
	if (!eq.getTemplate()->getName().empty())
		if (templates.find(eq.getTemplate()) == templates.end())
			templates.emplace(eq.getTemplate());
}

[[nodiscard]] size_t ModBltBlock::size() const
{
	size_t size = 0;
	for (const ModEquation& eq : equations)
		size += eq.getInductions().size();
	return size;
}

void ModBltBlock::dump(llvm::raw_ostream& OS) const
{
	OS << "template blt-block-" << body->getName() << "\n";
}

/**
 * This method compute the residual function of the BLT block by calculating the
 * difference between the right hand and the left hand of every equation.
 */
void ModBltBlock::computeResidualFunction()
{
	for (ModEquation& eq : equations)
	{
		ModExp newElement =
				ModExp::subtract(ModExp(eq.getRight()), ModExp(eq.getLeft()));
		newElement.tryFoldConstant();
		residualFunction.push_back(move(newElement));
	}
}

/**
 * This method compute the jacobian matrix of the BLT block by calculating the
 * derivative of every equation with respect to every variable in the BLT block.
 */
void ModBltBlock::computeJacobianMatrix()
{
	for (ModEquation& eq : equations)
	{
		jacobianMatrix.push_back(SmallVector<ModExp, 3>());
		for (ModEquation& eqIndex : equations)
		{
			ModExp matchedVarExp = eqIndex.getMatchedExp();
			ModEquation eqDerivative = differentiate(eq, matchedVarExp);

			ModExp newElement = ModExp::subtract(
					move(eqDerivative.getRight()), move(eqDerivative.getLeft()));
			newElement.tryFoldConstant();
			jacobianMatrix.back().push_back(move(newElement));
		}
	}
}
