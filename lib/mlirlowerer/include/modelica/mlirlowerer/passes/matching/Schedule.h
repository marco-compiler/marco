#pragma once

#include <modelica/mlirlowerer/passes/model/Model.h>

namespace modelica::codegen::model
{
	[[nodiscard]] mlir::LogicalResult schedule(Model& model);
}
