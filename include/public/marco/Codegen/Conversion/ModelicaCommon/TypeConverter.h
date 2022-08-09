#ifndef MARCO_CODEGEN_CONVERSION_MODELICACOMMON_TYPECONVERTER_H
#define MARCO_CODEGEN_CONVERSION_MODELICACOMMON_TYPECONVERTER_H

#include "marco/Dialect/IDA/IDADialect.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::modelica
{
	class TypeConverter : public mlir::TypeConverter
  {
		public:
      TypeConverter(unsigned int bitWidth);

    private:
      mlir::Type convertBooleanType(mlir::modelica::BooleanType type);
      mlir::Type convertIntegerType(mlir::modelica::IntegerType type);
      mlir::Type convertRealType(mlir::modelica::RealType type);
      mlir::Type convertArrayType(mlir::modelica::ArrayType type);
      mlir::Type convertUnrankedArrayType(mlir::modelica::UnrankedArrayType type);

    private:
		  unsigned int bitWidth;
	};
}

#endif // MARCO_CODEGEN_CONVERSION_MODELICACOMMON_TYPECONVERTER_H
