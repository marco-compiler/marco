#include "modelica/model/Model.hpp"

using namespace std;
using namespace modelica;
using namespace llvm;

[[nodiscard]] size_t Model::startingIndex(const string& varName) const
{
	auto varIterator = vars.find(varName);
	assert(varIterator != vars.end());

	size_t count = 0;
	for (const auto& var : make_range(vars.begin(), varIterator))
		count += var.second.size();

	return count;
}

Model::Model(SmallVector<ModEquation, 3> equs, StringMap<ModVariable> vars)
		: equations(std::move(equs)), vars(std::move(vars))
{
	for (const auto& eq : equations)
		addTemplate(eq);
}

void Model::addTemplate(const ModEquation& eq)
{
	if (templates.find(eq.getTemplate()) != templates.end())
		templates.try_emplace(
				eq.getTemplate(), "eq" + std::to_string(templates.size()));
}

bool Model::addVar(ModVariable exp)
{
	auto name = exp.getName();
	if (vars.find(name) != vars.end())
		return false;

	vars.try_emplace(move(name), std::move(exp));
	return true;
}

Model::Model(vector<ModEquation> equationsV, StringMap<ModVariable> vars)
		: vars(move(vars))
{
	for (auto& m : equationsV)
		equations.push_back(std::move(m));

	for (const auto& eq : equations)
		addTemplate(eq);
}
