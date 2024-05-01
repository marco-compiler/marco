#ifndef MARCO_CODEGEN_CONVERSION_MODELICACOMMON_TYPECONVERTER_H
#define MARCO_CODEGEN_CONVERSION_MODELICACOMMON_TYPECONVERTER_H

#include "marco/Dialect/BaseModelica/BaseModelicaDialect.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::bmodelica
{
	class TypeConverter : public mlir::TypeConverter
  {
		public:
      TypeConverter(unsigned int bitWidth);

    private:
      mlir::Type convertBooleanType(mlir::bmodelica::BooleanType type);
      mlir::Type convertIntegerType(mlir::bmodelica::IntegerType type);
      mlir::Type convertRealType(mlir::bmodelica::RealType type);
      mlir::Type convertArrayType(mlir::bmodelica::ArrayType type);

      mlir::Type convertUnrankedArrayType(
          mlir::bmodelica::UnrankedArrayType type);

    private:
		  unsigned int bitWidth;
	};
}

#endif // MARCO_CODEGEN_CONVERSION_MODELICACOMMON_TYPECONVERTER_H
