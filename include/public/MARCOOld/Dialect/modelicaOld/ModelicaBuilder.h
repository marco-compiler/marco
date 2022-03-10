#ifndef MARCO_DIALECT_MODELICA_MODELICABUILDER_H
#define MARCO_DIALECT_MODELICA_MODELICABUILDER_H

#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/IR/Builders.h"

namespace mlir::modelica
{
  class ModelicaBuilder : public mlir::OpBuilder
  {
    public:
      ModelicaBuilder(mlir::MLIRContext* context);

      BooleanType getBooleanType();

      IntegerType getIntegerType();

      RealType getRealType();

      ArrayType getArrayType(
          ArrayAllocationScope allocationScope,
          mlir::Type elementType,
          llvm::ArrayRef<long> shape = llvm::None);

      BooleanAttr getBooleanAttribute(bool value);

      IntegerAttr getIntegerAttribute(long value);

      RealAttr getRealAttribute(double value);

      mlir::Attribute getZeroAttribute(mlir::Type type);
  };
}

#endif // MARCO_DIALECT_MODELICA_MODELICABUILDER_H
