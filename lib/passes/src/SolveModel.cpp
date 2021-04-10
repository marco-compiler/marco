#include "modelica/passes/SolveModel.hpp"

#include <memory>

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/model/ModEqTemplate.hpp"
#include "modelica/model/ModErrors.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/model/Model.hpp"
#include "modelica/model/VectorAccess.hpp"
#include "modelica/utils/IRange.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

static Error replaceDer(ModExp& call, Model& model)
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

static Error solveDer(ModExp& exp, Model& model)
{
	if (exp.isCall() && exp.getCall().getName() == "der")
		return replaceDer(exp, model);

	for (auto& arg : exp)
		if (auto error = solveDer(arg, model); error)
			return error;

	return Error::success();
}

static Error solveDer(ModEquation& eq, Model& model)
{
	if (auto error = solveDer(eq.getLeft(), model); error)
		return error;
	if (auto error = solveDer(eq.getRight(), model); error)
		return error;
	return Error::success();
}

Error modelica::solveDer(Model& model)
{
	for (auto& eq : model.getEquations())
		if (auto error = ::solveDer(eq, model); error)
			return error;
	return Error::success();
}
