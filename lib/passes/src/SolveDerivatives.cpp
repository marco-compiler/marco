#include "modelica/passes/SolveDerivatives.hpp"

#include <memory>

#include "llvm/Support/Error.h"
#include "modelica/model/EntryModel.hpp"
#include "modelica/model/ModEqTemplate.hpp"
#include "modelica/model/ModErrors.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/model/VectorAccess.hpp"
#include "modelica/utils/IRange.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

static Error replaceDer(ModExp& call, EntryModel& model)
{
	auto firstArg = move(call.getCall().at(0));
	auto& access = firstArg.getReferredVectorAccessExp();
	if (!access.isReferenceAccess())
		return make_error<UnkownVariable>(
				" cannot use der operator on a not reference access");

	const auto& varName = access.getReference();
	auto& var = model.getVar(varName);

	auto newName = "der_" + varName;
	access.setReference(newName);
	if (!var.isState())
	{
		var.setIsState(true);
		model.emplaceVar(newName, var.getInit());
	}

	call = move(firstArg);
	return Error::success();
}

static Error solveDer(ModExp& exp, EntryModel& model)
{
	if (exp.isCall() && exp.getCall().getName() == "der")
		return replaceDer(exp, model);

	for (auto& arg : exp)
		if (auto error = solveDer(arg, model); error)
			return error;

	return Error::success();
}

static Error solveDer(ModEquation& eq, EntryModel& model)
{
	if (auto error = solveDer(eq.getLeft(), model); error)
		return error;
	if (auto error = solveDer(eq.getRight(), model); error)
		return error;
	return Error::success();
}

Error modelica::solveDer(EntryModel& model)
{
	for (auto& eq : model.getEquations())
		if (auto error = ::solveDer(eq, model); error)
			return error;
	return Error::success();
}

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

Expected<AssignModel> modelica::addAproximation(
		EntryModel& model, float deltaTime)
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

	for (auto& update : model.getEquations())
	{
		auto& templ = update.getTemplate();
		out.emplaceUpdate(templ, move(update.getInductions()), update.isForward());
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
		out.emplaceUpdate(templ, var.toMultiDimInterval());
	}

	return out;
}
