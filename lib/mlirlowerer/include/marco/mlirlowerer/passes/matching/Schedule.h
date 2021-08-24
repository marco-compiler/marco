#pragma once

#include <marco/mlirlowerer/passes/model/Model.h>

namespace marco::codegen::model
{
	[[nodiscard]] mlir::LogicalResult schedule(Model& model);
}
