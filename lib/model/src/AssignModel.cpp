#include "modelica/model/AssignModel.hpp"

using namespace std;
using namespace modelica;
using namespace llvm;

void AssignModel::addTemplate(const variant<Assigment, ModBltBlock>& update)
{
	if (!holds_alternative<Assigment>(update))
		return;

	Assigment assigment = get<Assigment>(update);
	if (assigment.getTemplate()->getName().empty() ||
			templates.find(assigment.getTemplate()) != templates.end())
		return;

	templates.emplace(assigment.getTemplate());
}

void AssignModel::dump(raw_ostream& OS) const
{
	OS << "init\n";
	for (const auto& var : variables)
		var.second.dump(OS);

	if (!templates.empty())
		OS << "templates\n";
	for (const auto& pair : templates)
	{
		pair->dump(true, OS);
		OS << "\n";
	}

	OS << "update\n";
	for (const auto& update : updates)
	{
		if (holds_alternative<Assigment>(update))
			get<Assigment>(update).dump(OS);
		else
			get<ModBltBlock>(update).dump(OS);
	}
}
