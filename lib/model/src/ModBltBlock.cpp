#include "modelica/model/ModBltBlock.hpp"

#include "modelica/model/SymbolicDifferentiation.hpp"

using namespace std;
using namespace modelica;
using namespace llvm;

[[nodiscard]] size_t ModBltBlock::startingIndex(const string& varName) const
{
	auto varIterator = vars.find(varName);
	assert(varIterator != vars.end());

	size_t count = 0;
	for (const auto& var : make_range(vars.begin(), varIterator))
		count += var.second.size();

	return count;
}

/**
 * This constructor initialize the BLT block with the equations it contains and
 * the variable matched to those equation. It then compute the residual function
 * and the jacobian matrix.
 */
ModBltBlock::ModBltBlock(
		SmallVector<ModEquation, 3> equs, SmallVector<ModVariable, 3> vars)
		: equations(std::move(equs))
{
	for (const auto& v : vars)
		addVar(v);
	for (const auto& eq : equations)
		addTemplate(eq);
	computeResidualFunction();
	computeJacobianMatrix();
}

void ModBltBlock::addTemplate(const ModEquation& eq)
{
	if (!eq.getTemplate()->getName().empty())
		if (templates.find(eq.getTemplate()) == templates.end())
			templates.emplace(eq.getTemplate());
}

bool ModBltBlock::addVar(ModVariable exp)
{
	auto name = exp.getName();
	if (vars.find(name) != vars.end())
		return false;

	vars.try_emplace(move(name), std::move(exp));
	return true;
}

void ModBltBlock::dump(llvm::raw_ostream& OS) const
{
	OS << "\tinit\n";
	for (const auto& var : getVars())
	{
		OS << "\t";
		var.second.dump(OS);
	}

	if (!getTemplates().empty())
		OS << "\ttemplate\n";
	for (const auto& temp : getTemplates())
	{
		OS << "\t";
		temp->dump(true, OS);
		OS << "\n";
	}

	OS << "\tupdate\n";
	for (const auto& update : *this)
	{
		OS << "\t";
		update.dump(OS);
	}
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
			ModVariable var = getVar(eqIndex.getMatchedVarReference());
			ModExp index = ModConst();
			if (var.size() != 1)
				index = ModExp::at(
						ModExp(var.getName(), var.getInit().getModType()),
						ModExp::index(ModConst(0)));

			ModEquation eqDerivative = differentiate(eq, var, index);
			ModExp newElement = ModExp::subtract(
					move(eqDerivative.getRight()), move(eqDerivative.getLeft()));
			newElement.tryFoldConstant();
			jacobianMatrix.back().push_back(move(newElement));
		}
	}
}
