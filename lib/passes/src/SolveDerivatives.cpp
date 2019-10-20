#include "modelica/passes/SolveDerivatives.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

Expected<AssignModel> modelica::solveDer(EntryModel&& model)
{
	AssignModel out;

	for (auto& var : model.getVars())
	{
		auto& name = var.second.getName();
		auto& exp = var.second.getInit();
		if (!out.addVar(move(name), move(exp)))
			return make_error<GlobalVariableCreationFailure>(var.first().str());
	}

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
