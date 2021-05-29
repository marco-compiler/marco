#include "marco/model/Model.hpp"

using namespace std;
using namespace marco;
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
	if (!eq.getTemplate()->getName().empty())
		if (eqTemplates.find(eq.getTemplate()) == eqTemplates.end())
			eqTemplates.insert(eq.getTemplate());
}

void Model::addTemplate(const ModBltBlock& bltBlock)
{
	if (!bltBlock.getTemplate()->getName().empty())
		if (bltTemplates.find(bltBlock.getTemplate()) == bltTemplates.end())
			bltTemplates.insert(bltBlock.getTemplate());
}

bool Model::addVar(ModVariable exp)
{
	auto name = exp.getName();
	if (vars.find(name) != vars.end())
		return false;

	vars.try_emplace(move(name), std::move(exp));
	return true;
}

void Model::dump(llvm::raw_ostream& OS) const
{
	OS << "init\n";
	for (const auto& var : getVars())
		var.second.dump(OS);

	OS << "template\n";
	for (const auto& temp : eqTemplates)
	{
		temp->dump(true, OS);
		OS << "\n";
	}
	for (const auto& temp : bltTemplates)
	{
		temp->dump(true, OS);
	}

	OS << "update\n";
	for (const ModEquation& eq : equations)
		eq.dump(OS);
	for (const ModBltBlock& bltBlock : bltBlocks)
		bltBlock.dump(OS);
}
