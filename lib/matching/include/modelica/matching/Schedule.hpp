#pragma once

#include "modelica/model/Model.hpp"
#include "modelica/model/ScheduledModel.hpp"

namespace modelica
{
	[[nodiscard]] ScheduledModel schedule(const Model& model);
}
