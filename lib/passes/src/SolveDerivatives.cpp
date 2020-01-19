#include "modelica/passes/SolveDerivatives.hpp"

#include "modelica/model/ModVariable.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

Expected<AssignModel> modelica::solveDer(EntryModel&& model, float deltaTime)
{
	AssignModel out;

	for (auto& var : model.getVars())
	{
		auto& name = var.second.getName();
		auto& exp = var.second.getInit();
		if (!out.addVar(ModVariable(move(name), move(exp))))
			return make_error<GlobalVariableCreationFailure>(var.first().str());
	}

	if (!out.addVar(ModVariable("deltaTime", ModExp(ModConst(deltaTime)))))
		return make_error<GlobalVariableCreationFailure>(
				"delta time was already present when solving derivatives");

	for (auto& update : model.getEquations())
	{
		auto left = move(update.getLeft());
		auto right = move(update.getRight());
		if (left.isCall() && left.getCall().getName() == "der")
		{
			auto leftInner = move(left.getCall().at(0));
			right = right * ModExp("deltaTime", BultinModTypes::FLOAT);
			right = ModExp::add(move(right), leftInner);
			out.emplaceUpdate(
					move(leftInner), move(right), move(update.getInductions()));
		}
		else
			out.emplaceUpdate(move(left), move(right), move(update.getInductions()));
	}

	return out;
}
