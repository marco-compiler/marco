#pragma once
#include "llvm/Support/Error.h"
#include "modelica/model/Model.hpp"

namespace modelica
{
	inline llvm::Expected<Model> constantFold(Model&& model)
	{
		for (auto& eq : model)
			eq.foldConstants();
		return std::move(model);
	}
}	 // namespace modelica
