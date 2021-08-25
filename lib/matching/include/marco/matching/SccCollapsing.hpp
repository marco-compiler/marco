#pragma once

#include <llvm/Support/Error.h>

#include "marco/model/Model.hpp"

namespace marco
{
	/**
	 * This method searches and tries to resolve all Algebraic Loops in the model.
	 * Some of them are caused by the vector equations and some are caused by the
	 * model itself. If the algorithm fails to solve some of this Algebraic Loops,
	 * the corresponding equations are inserted into a ModBltBlock that can be
	 * handled with a proper solver in the loweing phase.
	 *
	 * @param model The matched model.
	 * @param maxIterations Maximum depth search for the algorithm.
	 * @return The collapsed model.
	 */
	llvm::Expected<Model> solveScc(Model&& model, size_t maxIterations = 100);
}	 // namespace marco
