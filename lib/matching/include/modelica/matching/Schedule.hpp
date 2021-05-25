#pragma once

#include "modelica/model/Model.hpp"
#include "modelica/model/ScheduledModel.hpp"

namespace modelica
{
	/**
	 * This method reorders all updates, which can be ModEquations or
	 * ModBltBlocks, so that it will be possible to lower them in order without
	 * unresolved dependencies among the updates.
	 *
	 * @param model The matched and collapsed model.
	 * @return The scheduled model.
	 */
	[[nodiscard]] ScheduledModel schedule(const Model& model);
}	 // namespace modelica
