#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

#include "llvm/Support/raw_ostream.h"
#include "marco/model/ModEquation.hpp"
#include "marco/model/ModExp.hpp"
#include "marco/model/ModMatchers.hpp"
#include "marco/model/ModVariable.hpp"
#include "marco/model/Model.hpp"
#include "marco/model/VectorAccess.hpp"
#include "marco/utils/IndexSet.hpp"
namespace marco
{
	static void replaceUses(const ModEquation& newEq, ModEquation& original)
	{
		auto var = newEq.getDeterminedVariable();
		ReferenceMatcher matcher(original);
		for (auto& acc : matcher)
		{
			auto pathToVar = AccessToVar::fromExp(acc.getExp());
			if (pathToVar.getVarName() != var.getVarName())
				continue;

			auto composed = newEq.composeAccess(pathToVar.getAccess());
			original.reachExp(acc) = composed.getRight();
		}
	}

	inline llvm::Error linearySolve(
			llvm::SmallVectorImpl<ModEquation>& equs, const Model& model)
	{
		for (auto eq = equs.rbegin(); eq != equs.rend(); eq++)
		{
			for (auto eq2 = eq + 1; eq2 != equs.rend(); eq2++)
				replaceUses(*eq, *eq2);
		}

		for (auto& eq : equs)
			eq = eq.groupLeftHand();

		return llvm::Error::success();
	}
}	 // namespace marco
