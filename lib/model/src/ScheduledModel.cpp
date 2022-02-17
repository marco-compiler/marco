#include "marco/model/ScheduledModel.hpp"

using namespace std;
using namespace marco;
using namespace llvm;

ScheduledModel::ScheduledModel(llvm::StringMap<ModVariable> variables)
		: variables(move(variables))
{
	for (const auto& update : updates)
		addTemplate(get<ModEquation>(update));
}

void ScheduledModel::addTemplate(
		const variant<ModEquation, ModBltBlock>& update)
{
	if (holds_alternative<ModEquation>(update))
		templates.emplace(get<ModEquation>(update).getTemplate());
	else
		templates.emplace(get<ModBltBlock>(update).getTemplate());
}

void ScheduledModel::dump(llvm::raw_ostream& OS) const
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

	OS << "updates\n";
	for (const auto& update : updates)
	{
		if (holds_alternative<ModEquation>(update))
			get<ModEquation>(update).dump(OS);
		else
			get<ModBltBlock>(update).dump(OS);
	}
}
