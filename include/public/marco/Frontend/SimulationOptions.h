#ifndef MARCO_FRONTEND_SIMULATIONOPTIONS_H
#define MARCO_FRONTEND_SIMULATIONOPTIONS_H

#include "marco/Codegen/Transforms/ModelSolving/Solver.h"
#include <string>

namespace marco::frontend
{
  struct IDAOptions
  {
    bool equidistantTimeGrid = false;
  };

  struct SimulationOptions
  {
    std::string modelName = "";

    double startTime = 0;
    double endTime = 1;
    double timeStep = 0.1;

    marco::codegen::Solver solver = "forwardEuler";
    IDAOptions ida;
  };
}

#endif // MARCO_FRONTEND_SIMULATIONOPTIONS_H
