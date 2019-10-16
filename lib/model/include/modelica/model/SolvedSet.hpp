#pragma once

#include "llvm/ADT/StringMap.h"
#include "modelica/model/ModExp.hpp"

namespace modelica
{
	class SolvedSet
	{
		public:
		[[nodiscard]] const llvm::StringMap<ModExp>& getInit() const
		{
			return initialization;
		}
		[[nodiscard]] const llvm::StringMap<ModExp>& getUpdates() const
		{
			return updates;
		}

		private:
		llvm::StringMap<ModExp> initialization;
		llvm::StringMap<ModExp> updates;
	};
}	 // namespace modelica
