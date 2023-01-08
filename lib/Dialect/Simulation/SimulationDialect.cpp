#include "marco/Dialect/Simulation/SimulationDialect.h"
#include "marco/Dialect/Simulation/Ops.h"
#include "marco/Dialect/Simulation/Attributes.h"
#include "marco/Dialect/Simulation/Types.h"
#include "mlir/IR/DialectImplementation.h"

using namespace ::mlir;
using namespace ::mlir::simulation;

#include "marco/Dialect/Simulation/SimulationDialect.cpp.inc"

namespace
{
  struct SimulationOpAsmDialectInterface : public OpAsmDialectInterface
  {
    SimulationOpAsmDialectInterface(Dialect *dialect)
        : OpAsmDialectInterface(dialect)
    {
    }

    AliasResult getAlias(Attribute attr, raw_ostream &os) const override
    {
      if (attr.isa<MultidimensionalRangeAttr>()) {
        os << "range";
        return AliasResult::OverridableAlias;
      }

      if (attr.isa<VariableAttr>()) {
        os << "var";
        return AliasResult::OverridableAlias;
      }

      if (attr.isa<DerivativeAttr>()) {
        os << "der";
        return AliasResult::OverridableAlias;
      }

      return AliasResult::NoAlias;
    }

    AliasResult getAlias(Type type, raw_ostream &os) const final
    {
      return AliasResult::NoAlias;
    }
  };
}

//===---------------------------------------------------------------------===//
// Simulation dialect
//===---------------------------------------------------------------------===//

namespace mlir::simulation
{
  void SimulationDialect::initialize()
  {
    registerAttributes();

    addOperations<
#define GET_OP_LIST
#include "marco/Dialect/Simulation/Simulation.cpp.inc"
        >();

    addInterface<SimulationOpAsmDialectInterface>();
  }
}
