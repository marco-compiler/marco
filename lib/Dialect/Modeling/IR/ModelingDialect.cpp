#include "marco/Dialect/Modeling/IR/ModelingDialect.h"
#include "marco/Dialect/Modeling/IR/Ops.h"
#include "marco/Dialect/Modeling/IR/Attributes.h"
#include "marco/Dialect/Modeling/IR/Types.h"
#include "mlir/IR/DialectImplementation.h"

using namespace ::mlir::modeling;

#include "marco/Dialect/Modeling/IR/ModelingDialect.cpp.inc"

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
      if (attr.isa<IndexSetAttr>()) {
        os << "index_set";
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

namespace mlir::modeling
{
  void ModelingDialect::initialize()
  {
    registerAttributes();

    addOperations<
#define GET_OP_LIST
#include "marco/Dialect/Modeling/IR/Modeling.cpp.inc"
        >();

    addTypes<
#define GET_TYPEDEF_LIST
#include "marco/Dialect/Modeling/IR/ModelingTypes.cpp.inc"
        >();

    addInterfaces<
        ModelingOpAsmDialectInterface>();
  }
}
