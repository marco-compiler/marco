#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

#include "llvm/Support/Error.h"
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
	static llvm::Error replaceUses(
			const ModEquation& newEq, ModEquation& original)
	{
		auto var = newEq.getDeterminedVariable();
		ReferenceMatcher matcher(original);
		for (auto& acc : matcher)
		{
			auto pathToVar = AccessToVar::fromExp(acc.getExp());
			if (pathToVar.getVarName() != var.getVarName())
				continue;

			auto composed = newEq.composeAccess(pathToVar.getAccess());
			if (!composed)
				return composed.takeError();
			original.reachExp(acc) = (*composed).getRight();
		}
		return llvm::Error::success();
	}

	inline llvm::Error linearySolve(llvm::SmallVectorImpl<ModEquation>& equs)
	{
		for (auto eq = equs.rbegin(); eq != equs.rend(); eq++)
		{
			for (auto eq2 = eq + 1; eq2 != equs.rend(); eq2++)
				if (auto error = replaceUses(*eq, *eq2); error)
					return error;
		}

		for (auto& eq : equs)
			eq = eq.groupLeftHand();

		return llvm::Error::success();
	}
}	 // namespace marco
