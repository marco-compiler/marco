#pragma once
#include "modelica/model/AssignModel.hpp"
#include "modelica/model/ScheduledModel.hpp"

namespace modelica
{
	llvm::Expected<AssignModel> addApproximation(
			ScheduledModel& model, double deltaTime);
}	 // namespace modelica
