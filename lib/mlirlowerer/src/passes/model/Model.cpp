#include <marco/mlirlowerer/passes/model/Equation.h>
#include <marco/mlirlowerer/passes/model/Expression.h>
#include <marco/mlirlowerer/passes/model/Induction.h>
#include <marco/mlirlowerer/passes/model/Model.h>
#include <marco/mlirlowerer/passes/model/Operation.h>
#include <marco/mlirlowerer/passes/model/Reference.h>
#include <marco/mlirlowerer/passes/model/Variable.h>
#include <marco/mlirlowerer/passes/model/VectorAccess.h>
#include <marco/mlirlowerer/dialects/modelica/ModelicaDialect.h>

using namespace marco::codegen;
using namespace marco::codegen::model;
using namespace modelica;

class VariablesFinder
{
	public:
	VariablesFinder(llvm::SmallVector<std::shared_ptr<Variable>, 3>& variables)
			: variables(&variables)
	{
	}

	void operator()(const Constant& constant) const
	{

	}

	void operator()(const Reference& reference) const
	{
		if (!contains(reference.getVar()))
			variables->push_back(std::make_shared<Variable>(reference.getVar()));
	}

	void operator()(const Operation& operation) const
	{
		for (const auto& arg : operation)
			arg->visit(*this);
	}

	void operator()(const Induction& constant) const
	{

	}

	private:
	[[nodiscard]] bool contains(mlir::Value value) const
	{
		return std::any_of(variables->begin(), variables->end(),
											 [&](const auto& var) {
												 return var->getReference() == value;
											 });
	}

	llvm::SmallVector<std::shared_ptr<Variable>, 3>* variables;
};

Model::Model(SimulationOp op,
						 llvm::ArrayRef<std::shared_ptr<Variable>> variables,
						 llvm::ArrayRef<Equation> equations,
						 llvm::ArrayRef<BltBlock> bltBlocks)
		: op(op),
			variables(variables.begin(), variables.end()),
			equations(equations.begin(), equations.end()),
			bltBlocks(bltBlocks.begin(), bltBlocks.end())
{
}

Model Model::build(SimulationOp op)
{
	llvm::SmallVector<std::shared_ptr<Variable>, 3> variables;
	llvm::SmallVector<Equation, 3> equations;

	op.walk([&](EquationOp equation) {
		equations.push_back(Equation::build(equation));
	});

	op.walk([&](ForEquationOp forEquation) {
		equations.push_back(Equation::build(forEquation));
	});

	for (const Equation& equation : equations)
	{
		equation.lhs().visit(VariablesFinder(variables));
		equation.rhs().visit(VariablesFinder(variables));
	}

	return Model(op, variables, equations);
}

void Model::reloadIR()
{
	equations.clear();

	op.walk([&](EquationOp equation) {
		equations.push_back(Equation::build(equation));
	});

	op.walk([&](ForEquationOp forEquation) {
		equations.push_back(Equation::build(forEquation));
	});
}

Model::iterator Model::begin()
{
	return equations.begin();
}

Model::const_iterator Model::begin() const
{
	return equations.begin();
}

Model::iterator Model::end()
{
	return equations.end();
}

Model::const_iterator Model::end() const
{
	return equations.end();
}

SimulationOp Model::getOp() const
{
	return op;
}

bool Model::hasVariable(mlir::Value var) const
{
	return std::any_of(variables.begin(), variables.end(),
										 [&](const auto& v) { return var == v->getReference(); });
}

Variable Model::getVariable(mlir::Value var) const
{
	for (const auto& v : variables)
		if (var == (*v).getReference())
			return *v;

	assert(false && "Not found");
}

Model::Container<std::shared_ptr<Variable>>& Model::getVariables()
{
	return variables;
}

const Model::Container<std::shared_ptr<Variable>>& Model::getVariables() const
{
	return variables;
}

void Model::addVariable(mlir::Value var)
{
	if (!hasVariable(var))
		variables.push_back(std::make_shared<Variable>(var));
}

Model::Container<Equation>& Model::getEquations()
{
	return equations;
}

const Model::Container<Equation>& Model::getEquations() const
{
	return equations;
}

void Model::addEquation(Equation equation)
{
	equations.push_back(equation);
}

Model::Container<BltBlock>& Model::getBltBlocks()
{
	return bltBlocks;
}

const Model::Container<BltBlock>& Model::getBltBlocks() const
{
	return bltBlocks;
}

void Model::addBltBlock(BltBlock bltBlock)
{
	bltBlocks.push_back(bltBlock);
}

size_t Model::equationsCount() const
{
	size_t count = 0;

	for (const Equation& equation : equations)
		count += equation.getInductions().size();
	for (const BltBlock& bltBlock : bltBlocks)
		count += bltBlock.equationsCount();

	return count;
}

size_t Model::nonStateNonConstCount() const
{
	size_t count = 0;

	for (const auto& var : variables)
		if (!var->isState() && !var->isConstant())
			count += var->toIndexSet().size();

	return count;
}
