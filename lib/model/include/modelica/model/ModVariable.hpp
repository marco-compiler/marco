#pragma once
#include "modelica/model/ModExp.hpp"

namespace modelica
{
	class ModVar
	{
		private:
		std::string name;
		ModExp init;
	};
}	 // namespace modelica
