#pragma once
#include <vector>

#include "llvm/ADT/StringMap.h"
#include "modelica/solver/ConnectEquation.hpp"
#include "modelica/solver/DependencyGraph.hpp"
#include "modelica/solver/ModelJacobian.hpp"
#include "modelica/solver/SimEqu.hpp"
#include "modelica/solver/SimVar.hpp"

namespace modelica
{
	class Model
	{
		private:
		std::vector<std::unique_ptr<SimEqu>> equations;
		llvm::StringMap<SimVar> vars;
		ModelJacobian jacobian;
		DependecyGraph dependencyGraph;
		std::vector<ConnectEquation> connects;
	};
}	 // namespace modelica
