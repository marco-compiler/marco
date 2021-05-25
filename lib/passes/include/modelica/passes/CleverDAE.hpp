#pragma once
#include "modelica/model/AssignModel.hpp"
#include "modelica/model/ScheduledModel.hpp"

namespace modelica
{
	/**
	 * This method transforms all differential equations and implicit equations
	 * into BLT blocks within the model. The explicitable equations are
	 * transformed into assignments. Then the assigned model is returned.
	 *
	 * @param model The matched, collapsed and scheduled model.
	 * @return The assigned model.
	 */
	[[nodiscard]] llvm::Expected<AssignModel> addBltBlocks(ScheduledModel& model);
}	 // namespace modelica
