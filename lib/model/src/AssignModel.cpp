#include "marco/model/AssignModel.hpp"

using namespace std;
using namespace marco;
using namespace llvm;

void AssignModel::addTemplate(const variant<Assigment, ModBltBlock>& update)
{
	if (holds_alternative<Assigment>(update))
		templates.emplace(get<Assigment>(update).getTemplate());
	else
		templates.emplace(get<ModBltBlock>(update).getTemplate());
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
		if (holds_alternative<shared_ptr<ModEqTemplate>>(pair))
		{
			get<shared_ptr<ModEqTemplate>>(pair)->dump(true, OS);
			OS << "\n";
		}
		else
		{
			get<shared_ptr<ModBltTemplate>>(pair)->dump(true, OS);
		}
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
