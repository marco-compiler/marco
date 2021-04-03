#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/Variable.h>
#include <modelica/mlirlowerer/passes/model/VectorAccess.h>

using namespace modelica::codegen;
using namespace modelica::codegen::model;

Model::Model(SimulationOp op,
						 llvm::ArrayRef<std::shared_ptr<Variable>> variables,
						 llvm::ArrayRef<std::shared_ptr<Equation>> equations)
		: op(op)
{
	for (const auto& variable : variables)
		this->variables.emplace_back(std::make_shared<Variable>(*variable));

	for (const auto& equation : equations)
		addEquation(*equation);
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
	for (const auto& v : variables)
		if (var == (*v).getReference())
			return true;

	return false;
}

Variable& Model::getVariable(mlir::Value var)
{
	for (const auto& v : variables)
		if (var == (*v).getReference())
			return *v;

	assert(false && "Not found");
}

const Variable& Model::getVariable(mlir::Value var) const
{
	for (const auto& v : variables)
		if (var == (*v).getReference())
			return *v;

	assert(false && "Not found");
}

Model::Container<Variable>& Model::getVariables()
{
	return variables;
}

const Model::Container<Variable>& Model::getVariables() const
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
	equations.push_back(std::make_shared<Equation>(equation));

	if (!equation.getTemplate()->getName().empty())
		if (templates.find(equation.getTemplate()) == templates.end())
			templates.emplace(equation.getTemplate());
}

const Model::TemplateMap& Model::getTemplates() const
{
	return templates;
}

size_t Model::equationsCount() const
{
	size_t count = 0;

	for (const auto& equation : equations)
		count += equation->getInductions().size();

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
