#pragma once

#include "modelica/model/AssignModel.hpp"
#include "modelica/model/EntryModel.hpp"
namespace modelica
{
	[[nodiscard]] AssignModel schedule(const EntryModel& model);
}
