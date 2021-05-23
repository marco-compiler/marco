#include "modelica/passes/CleverDAE.hpp"

#include "llvm/Support/Error.h"
#include "modelica/model/AssignModel.hpp"
#include "modelica/model/ModBltBlock.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModErrors.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/model/Model.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

Expected<AssignModel> modelica::addBltBlocks(ScheduledModel& model)
{
	AssignModel out;
	size_t implicitCount = 0, derivativeCount = 0;

	for (auto& content : model.getUpdates())
	{
		// Add the algebraic loops, which are already in BLT blocks.
		if (holds_alternative<ModBltBlock>(content))
		{
			out.addUpdate(get<ModBltBlock>(content));
			continue;
		}

		ModEquation update = get<ModEquation>(content);
		ModEquation u =
				update.clone(update.getTemplate()->getName() + "explicitated");
		string matchedVar = u.getMatchedVarReference();

		// Transform all implicit equations in BLT blocks.
		if (auto error = u.explicitate(); error)
			return move(error);

		if (u.isImplicit())
		{
			out.addUpdate(ModBltBlock(
					{ update },
					{ ModVariable(model.getVar(matchedVar)) },
					"implicit" + to_string(implicitCount++)));
		}

		// Transform all differential equations in BLT blocks.
		else if (
				matchedVar.substr(0, 4) == "der_" &&
				model.getVar(matchedVar.substr(4)).isState())
		{
			out.addUpdate(ModBltBlock(
					{ update },
					{ ModVariable(model.getVar(matchedVar)) },
					"derivative" + to_string(implicitCount++)));
		}

		// Add the remaining equations to the assignments list.
		else
		{
			auto& templ = u.getTemplate();
			out.addUpdate(
					Assigment(templ, move(update.getInductions()), update.isForward()));
		}
	}

	// Add all the variables of the model to the assign model.
	for (auto& var : model.getVars())
	{
		if (!out.addVar(move(var.second)))
			return make_error<GlobalVariableCreationFailure>(var.first().str());
	}

	return out;
}
