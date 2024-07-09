#include "marco/Dialect/Modeling/IR/Modeling.h"
#include "marco/Dialect/Modeling/IR/Ops.h"
#include "marco/Dialect/Modeling/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace ::mlir::modeling;

#include "marco/Dialect/Modeling/IR/Modeling.cpp.inc"

//===---------------------------------------------------------------------===//
// Modeling dialect
//===---------------------------------------------------------------------===//

namespace
{
  struct ModelingOpAsmDialectInterface : public mlir::OpAsmDialectInterface
  {
    explicit ModelingOpAsmDialectInterface(mlir::Dialect* dialect)
        : OpAsmDialectInterface(dialect)
    {
    }

    AliasResult getAlias(
        mlir::Attribute attr, llvm::raw_ostream& os) const override
    {
      return AliasResult::NoAlias;
    }

    AliasResult getAlias(mlir::Type type, llvm::raw_ostream& os) const final
    {
      return AliasResult::NoAlias;
    }
  };
}

namespace mlir::modeling
{
  void ModelingDialect::initialize()
  {
    registerAttributes();

    addOperations<
#define GET_OP_LIST
#include "marco/Dialect/Modeling/IR/ModelingOps.cpp.inc"
        >();

    addInterfaces<
        ModelingOpAsmDialectInterface>();
  }
}
