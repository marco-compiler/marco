#pragma once

#include "modelica/model/Model.hpp"

namespace modelica
{
	class EntryModel: public Model
	{
		public:
		EntryModel(
				std::vector<ModEquation> equations, llvm::StringMap<ModVariable> vars)
				: Model(std::move(equations), std::move(vars))
		{
		}
		EntryModel() = default;
	};
}	 // namespace modelica
