#pragma once
#include "modelica/model/AssignModel.hpp"
#include "modelica/model/EntryModel.hpp"
#include "modelica/model/ModErrors.hpp"

namespace modelica
{
	llvm::Expected<AssignModel> solveDer(EntryModel&& model, float deltaTime);
}	 // namespace modelica
