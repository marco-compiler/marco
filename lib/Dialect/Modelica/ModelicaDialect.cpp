#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/Modelica/Ops.h"
#include "marco/Dialect/Modelica/Attributes.h"
#include "marco/Dialect/Modelica/Types.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir;
using namespace ::mlir::modelica;

#include "marco/Dialect/Modelica/ModelicaDialect.cpp.inc"

namespace mlir::modelica
{
  //===----------------------------------------------------------------------===//
  // Modelica dialect
  //===----------------------------------------------------------------------===//

  void ModelicaDialect::initialize()
  {
    registerTypes();
    registerAttributes();

    addOperations<
      #define GET_OP_LIST
      #include "marco/Dialect/Modelica/Modelica.cpp.inc"
        >();
  }

  Operation* ModelicaDialect::materializeConstant(
      mlir::OpBuilder& builder, mlir::Attribute value, mlir::Type type, mlir::Location loc)
  {
    return builder.create<ConstantOp>(loc, type, value);
  }
}
