#pragma once

#include "marco/model/Model.hpp"
#include "marco/model/ScheduledModel.hpp"

namespace marco
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
}	 // namespace marco
