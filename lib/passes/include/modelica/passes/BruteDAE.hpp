#pragma once
#include "modelica/model/AssignModel.hpp"
#include "modelica/model/Model.hpp"

namespace modelica
{
	llvm::Expected<AssignModel> addJacobianAndResidual(Model& model);
}	 // namespace modelica
