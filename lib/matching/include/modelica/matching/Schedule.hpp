#pragma once

#include "modelica/model/AssignModel.hpp"
#include "modelica/model/EntryModel.hpp"
namespace modelica
{
	[[nodiscard]] EntryModel schedule(const EntryModel& model);
}
