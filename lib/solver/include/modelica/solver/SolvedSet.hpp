#pragma once

#include "llvm/ADT/StringMap.h"
#include "modelica/simulation/SimExp.hpp"

namespace modelica
{
	class SolvedSet
	{
		public:
		[[nodiscard]] const llvm::StringMap<SimExp>& getInit() const
		{
			return initialization;
		}
		[[nodiscard]] const llvm::StringMap<SimExp>& getUpdates() const
		{
			return updates;
		}

		private:
		llvm::StringMap<SimExp> initialization;
		llvm::StringMap<SimExp> updates;
	};
}	 // namespace modelica
