#ifndef MARCO_FRONTEND_SIMULATIONOPTIONS_H
#define MARCO_FRONTEND_SIMULATIONOPTIONS_H

#include "marco/Codegen/Transforms/ModelSolving/ModelSolving.h"
#include <string>

namespace marco::frontend
{
  struct IDAOptions
  {
    double relativeTolerance;
    double absoluteTolerance;
    bool equidistantTimeGrid;
  };

  struct SimulationOptions
  {
    std::string modelName;

    double startTime;
    double endTime;
    double timeStep;

    marco::codegen::Solver solver;
    IDAOptions ida;

    SimulationOptions();
  };
}

#endif // MARCO_FRONTEND_SIMULATIONOPTIONS_H
