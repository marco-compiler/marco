#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <modelica/utils/IndexSet.hpp>

#include "Equation.h"
#include "Expression.h"
#include "Model.h"
#include "ReferenceMatcher.h"
#include "Variable.h"
#include "VectorAccess.h"

namespace modelica::codegen::model
{
	static void replaceUses(const Equation& newEq, Equation& original)
	{
		auto var = newEq.getDeterminedVariable();
		ReferenceMatcher matcher(original);

		for (auto& acc : matcher)
		{
			auto pathToVar = AccessToVar::fromExp(acc.getExp());

			if (pathToVar.getVar() != var.getVar())
				continue;

			auto composed = newEq.composeAccess(pathToVar.getAccess());
			original.reachExp(acc) = composed->rhs();
		}
	}

	inline llvm::Error linearySolve(llvm::SmallVectorImpl<Equation::Ptr>& equations)
	{
		for (auto eq = equations.rbegin(); eq != equations.rend(); eq++)
		{
			for (auto eq2 = eq + 1; eq2 != equations.rend(); eq2++)
				replaceUses(**eq, **eq2);
		}

		//for (auto& eq : equations)
		//	eq = eq->groupLeftHand();

		return llvm::Error::success();
	}
}	 // namespace modelica
