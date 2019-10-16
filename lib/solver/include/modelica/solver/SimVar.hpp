#pragma once
#include "modelica/simulation/SimExp.hpp"

namespace modelica
{
	class SimVar
	{
		private:
		std::string name;
		SimExp init;
	};
}	 // namespace modelica
