#ifndef MARCO_FRONTEND_SIMULATIONOPTIONS_H
#define MARCO_FRONTEND_SIMULATIONOPTIONS_H

#include <string>

namespace marco::frontend
{
  struct SimulationOptions
  {
    std::string modelName = "";
    std::string solver = "forwardEuler";
    bool IDAReducedSystem = true;
    bool IDAReducedDerivatives = true;
    bool IDAJacobianOneSweep = true;
  };
}

#endif // MARCO_FRONTEND_SIMULATIONOPTIONS_H
