#include "marco/Codegen/Bridge.h"
#include "marco/Frontend/SimulationOptions.h"

namespace marco::frontend
{
  SimulationOptions::SimulationOptions()
  {
    auto defaultCodegen = codegen::CodegenOptions::getDefaultOptions();

    this->startTime = defaultCodegen.startTime;
    this->endTime = defaultCodegen.endTime;
    this->timeStep = defaultCodegen.timeStep;
  }
}