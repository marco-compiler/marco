#include "marco/Dialect/Simulation/Attributes.h"
#include "marco/Dialect/Simulation/SimulationDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::simulation;

//===---------------------------------------------------------------------===//
// Tablegen attribute definitions
//===---------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/Simulation/SimulationAttributes.cpp.inc"

//===---------------------------------------------------------------------===//
// SimulationDialect
//===---------------------------------------------------------------------===//

namespace mlir::simulation
{
  void SimulationDialect::registerAttributes()
  {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "marco/Dialect/Simulation/SimulationAttributes.cpp.inc"
        >();
  }
}

//===---------------------------------------------------------------------===//
// Attributes
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// VariableAttr

namespace mlir::simulation
{
  int64_t VariableAttr::getRank() const
  {
    return getDimensions().size();
  }
}
