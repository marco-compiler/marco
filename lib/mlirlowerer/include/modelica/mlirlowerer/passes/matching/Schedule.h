#pragma once

#include <modelica/mlirlowerer/passes/model/Model.h>

namespace modelica::codegen::model
{
	/**
	 * This method reorders all updates, which can be Equations or BltBlocks, so
	 * that it will be possible to lower them in order without unresolved
	 * dependencies among the updates.
	 *
	 * @param model The matched and collapsed model.
	 */
	[[nodiscard]] mlir::LogicalResult schedule(Model& model);
}	 // namespace modelica::codegen::model
