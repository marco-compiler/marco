#include "modelica/model/ModLoop.hpp"

using namespace std;
using namespace modelica;
using namespace llvm;

ModLoop::ModLoop(SmallVector<ModEquation, 3> equs, StringMap<ModVariable> vars)
		: equations(std::move(equs)), vars(std::move(vars))
{
	for (const auto& eq : equations)
		addTemplate(eq);
}

void ModLoop::addTemplate(const ModEquation& eq)
{
	if (!eq.getTemplate()->getName().empty())
		if (templates.find(eq.getTemplate()) == templates.end())
			templates.emplace(eq.getTemplate());
}

void ModLoop::dump(llvm::raw_ostream& OS) const
{
	OS << "loop init\n";
	for (const auto& var : getVars())
	{
		OS << "\t";
		var.second.dump(OS);
	}

	if (!getTemplates().empty())
		OS << "\tloop template\n";
	for (const auto& temp : getTemplates())
	{
		OS << "\t";
		temp->dump(true, OS);
		OS << "\n";
	}

	OS << "\tloop update\n";
	for (const auto& update : *this)
	{
		OS << "\t";
		update.dump(OS);
	}
}
