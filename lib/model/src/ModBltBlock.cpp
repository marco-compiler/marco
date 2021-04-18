#include "modelica/model/ModBltBlock.hpp"

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

ModBltBlock::ModBltBlock(
		SmallVector<ModEquation, 3> equs, SmallVector<ModVariable, 3> vars)
		: equations(std::move(equs))
{
	for (const auto& v : vars)
		addVar(v);
	for (const auto& eq : equations)
		addTemplate(eq);
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
