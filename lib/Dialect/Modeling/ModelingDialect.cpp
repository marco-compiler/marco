#include "marco/Dialect/Modeling/ModelingDialect.h"
#include "marco/Dialect/Modeling/Ops.h"
#include "marco/Dialect/Modeling/Attributes.h"
#include "marco/Dialect/Modeling/Types.h"
#include "mlir/IR/DialectImplementation.h"

using namespace ::mlir;
using namespace ::mlir::modeling;

#include "marco/Dialect/Modeling/ModelingDialect.cpp.inc"

//===---------------------------------------------------------------------===//
// Modeling dialect
//===---------------------------------------------------------------------===//

namespace
{
  struct ModelingOpAsmDialectInterface : public OpAsmDialectInterface
  {
    ModelingOpAsmDialectInterface(Dialect *dialect)
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
#include "marco/Dialect/Modeling/Modeling.cpp.inc"
        >();

    addInterfaces<
        ModelingOpAsmDialectInterface>();
  }
}
