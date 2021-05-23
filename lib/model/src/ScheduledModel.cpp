#include "modelica/model/ScheduledModel.hpp"

using namespace std;
using namespace modelica;
using namespace llvm;

ScheduledModel::ScheduledModel(llvm::StringMap<ModVariable> variables)
		: variables(std::move(variables))
{
	for (const auto& update : updates)
		if (std::holds_alternative<ModEquation>(update))
			addTemplate(std::get<ModEquation>(update));
}

void ScheduledModel::addTemplate(const ModEquation& eq)
{
	if (!eq.getTemplate()->getName().empty())
		if (templates.find(eq.getTemplate()) == templates.end())
			templates.emplace(eq.getTemplate());
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
		pair->dump(true, OS);
		OS << "\n";
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
