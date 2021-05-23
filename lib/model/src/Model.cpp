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
		if (templates.find(eq.getTemplate()) == templates.end())
			templates.emplace(eq.getTemplate());
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

	if (!getTemplates().empty())
		OS << "template\n";
	for (const auto& temp : getTemplates())
	{
		temp->dump(true, OS);
		OS << "\n";
	}

	if (!getBltBlocks().empty())
		OS << "blt-blocks\n";
	for (const auto& bltBlock : getBltBlocks())
		bltBlock.dump(OS);

	OS << "update\n";
	for (const auto& update : *this)
		update.dump(OS);
}
