#pragma once
#include "modelica/model/AssignModel.hpp"
#include "modelica/model/ScheduledModel.hpp"

namespace modelica
{
	/**
	 * This method explicitate all equations in the model, so that they can be
	 * converted into assignments. It also adds to the model the time step used by
	 * the Euler method and the equations for the update of the state variables.
	 * This method fails if some ModBltBlocks or implicit equations are present.
	 *
	 * @param model The matched, collapsed and scheduled model.
	 * @param deltaTime The constant time step used by the Euler method.
	 * @return The assigned model.
	 */
	llvm::Expected<AssignModel> addApproximation(
			ScheduledModel& model, double deltaTime);
}	 // namespace modelica
