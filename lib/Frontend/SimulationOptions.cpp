#include "marco/Frontend/SimulationOptions.h"
#include "marco/Codegen/Bridge.h"

namespace marco::frontend
{
  SimulationOptions::SimulationOptions()
  {
    auto defaultModelSolvingOptions = codegen::ModelSolvingOptions::getDefaultOptions();

    this->startTime = defaultModelSolvingOptions.startTime;
    this->endTime = defaultModelSolvingOptions.endTime;
    this->timeStep = defaultModelSolvingOptions.timeStep;

    this->solver = defaultModelSolvingOptions.solver;

    this->ida.relativeTolerance = defaultModelSolvingOptions.ida.relativeTolerance;
    this->ida.absoluteTolerance = defaultModelSolvingOptions.ida.absoluteTolerance;
    this->ida.equidistantTimeGrid = defaultModelSolvingOptions.ida.equidistantTimeGrid;
  }
}