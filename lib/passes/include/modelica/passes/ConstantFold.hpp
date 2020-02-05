#pragma once
#include "llvm/Support/Error.h"
#include "modelica/model/EntryModel.hpp"

namespace modelica
{
	inline llvm::Expected<EntryModel> constantFold(EntryModel&& model)
	{
		for (auto& eq : model)
			eq.foldConstants();
		return std::move(model);
	}
}	 // namespace modelica
