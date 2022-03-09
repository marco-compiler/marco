#ifndef MARCO_FRONTEND_SIMULATIONOPTIONS_H
#define MARCO_FRONTEND_SIMULATIONOPTIONS_H

namespace marco::frontend
{
  struct SimulationOptions
  {
    std::string modelName;

    double startTime;
    double endTime;
    double timeStep;

    SimulationOptions();
  };
}

#endif // MARCO_FRONTEND_SIMULATIONOPTIONS_H
