#include "marco/passes/ForwardEuler.hpp"

#include <memory>

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "marco/model/ModEqTemplate.hpp"
#include "marco/model/ModErrors.hpp"
#include "marco/model/ModExp.hpp"
#include "marco/model/ModVariable.hpp"
#include "marco/model/Model.hpp"
#include "marco/model/VectorAccess.hpp"
#include "marco/utils/IRange.hpp"

using namespace marco;
using namespace std;
using namespace llvm;

static ModExp varToExp(const ModVariable& var)
{
	auto access = ModExp(var.getName(), var.getInit().getModType());
	if (var.getInit().getModType().isScalar())
		return access;

	auto inds = var.toMultiDimInterval();
	for (auto i : irange<int>(inds.dimensions()))
	{
		auto ind = ModExp::induction(ModConst(i));
		access = ModExp::at(access, move(ind));
	}
	return access;
}

Expected<AssignModel> marco::addApproximation(
		ScheduledModel& model, double deltaTime)
{
	AssignModel out;

	for (auto& var : model.getVars())
	{
		if (!out.addVar(move(var.second)))
			return make_error<GlobalVariableCreationFailure>(var.first().str());
	}

	if (!out.addVar(ModVariable("deltaTime", ModExp(ModConst(deltaTime)))))
		return make_error<GlobalVariableCreationFailure>(
				"delta time was already present when solving derivatives");

	for (auto& content : model.getUpdates())
	{
		if (!holds_alternative<ModEquation>(content))
			return make_error<UnsolvableAlgebraicLoop>();

		ModEquation update = get<ModEquation>(content);
		auto u = update.clone(update.getTemplate()->getName() + "explicitated");
		if (auto e = u.explicitate(); e)
			return move(e);
		auto& templ = u.getTemplate();
		out.addUpdate(
				Assigment(templ, move(update.getInductions()), update.isForward()));
	}

	for (const auto& varP : out.getVars())
	{
		auto& var = varP.second;
		if (!var.isState())
			continue;
		auto left = varToExp(varP.second);

		std::string derName = "der_" + varP.first().str();
		auto derVar = out.getVars().find(derName);
		if (derVar == out.getVars().end())
			return make_error<GlobalVariableCreationFailure>(
					derName + " was not found");

		auto right = varToExp(derVar->second);
		auto leftCp = left;
		right = move(leftCp) +
						(ModExp("deltaTime", BultinModTypes::FLOAT) * move(right));

		auto templ = make_shared<ModEqTemplate>(
				move(left), move(right), derName + "_update");
		out.addUpdate(Assigment(templ, var.toMultiDimInterval()));
	}

	return out;
}
