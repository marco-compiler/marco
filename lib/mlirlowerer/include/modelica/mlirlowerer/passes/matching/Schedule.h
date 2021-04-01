#pragma once

#include <modelica/mlirlowerer/passes/model/Model.h>

namespace modelica::codegen::model
{
	[[nodiscard]] Model schedule(const Model& model);
}
