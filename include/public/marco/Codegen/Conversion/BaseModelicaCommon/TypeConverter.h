#ifndef MARCO_CODEGEN_CONVERSION_BASEMODELICACOMMON_TYPECONVERTER_H
#define MARCO_CODEGEN_CONVERSION_BASEMODELICACOMMON_TYPECONVERTER_H

#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/IDA/IR/IDA.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::bmodelica {
class TypeConverter : public mlir::TypeConverter {
public:
  TypeConverter();

private:
  mlir::Type convertBooleanType(mlir::bmodelica::BooleanType type);
  mlir::Type convertIntegerType(mlir::bmodelica::IntegerType type);
  mlir::Type convertRealType(mlir::bmodelica::RealType type);
  mlir::Type convertArrayType(mlir::bmodelica::ArrayType type);

  mlir::Type convertUnrankedArrayType(mlir::bmodelica::UnrankedArrayType type);

  mlir::Type convertTensorType(mlir::TensorType type);
};
} // namespace mlir::bmodelica

#endif // MARCO_CODEGEN_CONVERSION_BASEMODELICACOMMON_TYPECONVERTER_H