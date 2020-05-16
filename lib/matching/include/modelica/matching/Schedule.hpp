#pragma once

#include "modelica/model/AssignModel.hpp"
#include "modelica/model/Model.hpp"
namespace modelica
{
	[[nodiscard]] Model schedule(const Model& model);
}
