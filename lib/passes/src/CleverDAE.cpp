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

Expected<AssignModel> modelica::addBLTBlocks(Model& model)
{
	AssignModel out;

	for (ModEquation& update : model.getEquations())
	{
		ModEquation u =
				update.clone(update.getTemplate()->getName() + "explicitated");
		string matchedVar = u.getMatchedVarReference();

		// Transform all implicit equations in BLT blocks.
		if (auto error = u.explicitate(); error)
			return move(error);

		if (u.isImplicit())
		{
			out.addBltBlock(
					ModBltBlock({ update }, { ModVariable(model.getVar(matchedVar)) }));
		}

		// Transform all differential equations in BLT blocks.
		else if (
				matchedVar.substr(0, 4) == "der_" &&
				model.getVar(matchedVar.substr(4)).isState())
		{
			out.addBltBlock(
					ModBltBlock({ update }, { ModVariable(model.getVar(matchedVar)) }));
		}

		// Add the remaining equations to the assignments list.
		else
		{
			auto& templ = u.getTemplate();
			out.emplaceUpdate(
					templ, move(update.getInductions()), update.isForward());
		}
	}

	// Add the algebraic loops, which are already in BLT blocks.
	for (ModBltBlock& bltBlock : model.getBltBlocks())
	{
		out.addBltBlock(move(bltBlock));
	}

	// Add all the variables of the model to the assign model.
	for (auto& var : model.getVars())
	{
		if (!out.addVar(move(var.second)))
			return make_error<GlobalVariableCreationFailure>(var.first().str());
	}

	return out;
}
