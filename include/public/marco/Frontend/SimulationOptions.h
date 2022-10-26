#ifndef MARCO_FRONTEND_SIMULATIONOPTIONS_H
#define MARCO_FRONTEND_SIMULATIONOPTIONS_H

#include "marco/Codegen/Transforms/ModelSolving/Solver.h"
#include <string>

namespace marco::frontend
{
  struct SimulationOptions
  {
    std::string modelName = "";
    marco::codegen::Solver solver = "forwardEuler";
  };
}

#endif // MARCO_FRONTEND_SIMULATIONOPTIONS_H
