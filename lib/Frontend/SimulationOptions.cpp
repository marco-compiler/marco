#include "marco/Frontend/SimulationOptions.h"
#include "marco/Codegen/Bridge.h"

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