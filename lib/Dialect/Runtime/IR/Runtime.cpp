#include "marco/Dialect/Runtime/IR/Runtime.h"

using namespace ::mlir::runtime;

#include "marco/Dialect/Runtime/IR/Runtime.cpp.inc"

namespace
{
  struct RuntimeOpAsmDialectInterface : public mlir::OpAsmDialectInterface
  {
    explicit RuntimeOpAsmDialectInterface(mlir::Dialect* dialect)
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
// Runtime dialect
//===---------------------------------------------------------------------===//

namespace mlir::runtime
{
  void RuntimeDialect::initialize()
  {
    registerAttributes();

    addOperations<
#define GET_OP_LIST
#include "marco/Dialect/Runtime/IR/RuntimeOps.cpp.inc"
        >();

    addInterface<RuntimeOpAsmDialectInterface>();
  }
}
