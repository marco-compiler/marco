#pragma once
#include <vector>

#include "llvm/ADT/StringMap.h"
#include "modelica/model/ConnectEquation.hpp"
#include "modelica/model/DependencyGraph.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/model/ModelJacobian.hpp"

namespace modelica
{
	class Model
	{
		private:
		std::vector<std::unique_ptr<ModEqu>> equations;
		llvm::StringMap<ModVar> vars;
		ModelJacobian jacobian;
		DependecyGraph dependencyGraph;
		std::vector<ConnectEquation> connects;
	};
}	 // namespace modelica
