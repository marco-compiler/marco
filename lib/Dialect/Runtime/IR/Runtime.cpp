#include "marco/Dialect/Runtime/IR/Runtime.h"

using namespace ::mlir::runtime;

#include "marco/Dialect/Runtime/IR/Runtime.cpp.inc"

namespace {
struct RuntimeOpAsmDialectInterface : public mlir::OpAsmDialectInterface {
  explicit RuntimeOpAsmDialectInterface(mlir::Dialect *dialect)
      : OpAsmDialectInterface(dialect) {}

  AliasResult getAlias(mlir::Attribute attr,
                       llvm::raw_ostream &os) const override {
    if (mlir::isa<MultidimensionalRangeAttr>(attr)) {
      os << "range";
      return AliasResult::OverridableAlias;
    }

    if (mlir::isa<VariableAttr>(attr)) {
      os << "var";
      return AliasResult::OverridableAlias;
    }

    if (mlir::isa<DerivativeAttr>(attr)) {
      os << "der";
      return AliasResult::OverridableAlias;
    }

    return AliasResult::NoAlias;
  }

  AliasResult getAlias(mlir::Type type, llvm::raw_ostream &os) const final {
    return AliasResult::NoAlias;
  }
};
} // namespace

//===---------------------------------------------------------------------===//
// Runtime dialect
//===---------------------------------------------------------------------===//

namespace mlir::runtime {
void RuntimeDialect::initialize() {
  registerAttributes();
  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "marco/Dialect/Runtime/IR/RuntimeOps.cpp.inc"

      >();

  addInterface<RuntimeOpAsmDialectInterface>();
}
} // namespace mlir::runtime
