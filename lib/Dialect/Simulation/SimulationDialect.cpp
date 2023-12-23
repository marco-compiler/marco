#include "marco/Dialect/Simulation/SimulationDialect.h"

using namespace ::mlir::simulation;

#include "marco/Dialect/Simulation/SimulationDialect.cpp.inc"

namespace
{
  struct SimulationOpAsmDialectInterface : public mlir::OpAsmDialectInterface
  {
    explicit SimulationOpAsmDialectInterface(mlir::Dialect* dialect)
        : OpAsmDialectInterface(dialect)
    {
    }

    AliasResult getAlias(
        mlir::Attribute attr, llvm::raw_ostream& os) const override
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

    AliasResult getAlias(mlir::Type type, llvm::raw_ostream& os) const final
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

    addTypes<
#define GET_TYPEDEF_LIST
#include "marco/Dialect/Simulation/SimulationTypes.cpp.inc"
        >();

    addInterface<SimulationOpAsmDialectInterface>();
  }
}
