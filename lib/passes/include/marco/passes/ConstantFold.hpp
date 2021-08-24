#pragma once
#include "llvm/Support/Error.h"
#include "marco/model/Model.hpp"

namespace marco
{
	inline llvm::Expected<Model> constantFold(Model&& model)
	{
		for (auto& eq : model)
			eq.foldConstants();
		return std::move(model);
	}
}	 // namespace marco
