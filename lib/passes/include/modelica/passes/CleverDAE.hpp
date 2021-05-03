#pragma once
#include "modelica/model/AssignModel.hpp"
#include "modelica/model/Model.hpp"

namespace modelica
{
	/**
	 * This method transforms all differential equations and implicit equations
	 * into BLT blocks within the model. Then the assigned model, ready for
	 * lowering, is returned.
	 *
	 * @param model The matched, collapsed and scheduled model.
	 * @return The assigned model.
	 */
	llvm::Expected<AssignModel> addBLTBlocks(Model& model);
}	 // namespace modelica
