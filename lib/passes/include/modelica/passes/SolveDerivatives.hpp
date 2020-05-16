#pragma once
#include "modelica/model/AssignModel.hpp"
#include "modelica/model/EntryModel.hpp"
#include "modelica/model/ModErrors.hpp"

namespace modelica
{
	llvm::Expected<AssignModel> addAproximation(
			EntryModel& model, float deltaTime);

	llvm::Error solveDer(EntryModel& model);
}	 // namespace modelica
