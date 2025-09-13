#ifndef MARCO_CODEGEN_CONVERSION_BASEMODELICACOMMON_CTYPECONVERTER_H
#define MARCO_CODEGEN_CONVERSION_BASEMODELICACOMMON_CTYPECONVERTER_H

#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::bmodelica {
class CTypeConverter : public mlir::TypeConverter {
  int booleanBitWidth{32};
  int integerBitWidth{32};
  int indexBitWidth{64};

public:
  CTypeConverter(mlir::MLIRContext *context,
                 const mlir::DataLayout &dataLayout);

private:
  mlir::Type convertBooleanType(mlir::bmodelica::BooleanType type);
  mlir::Type convertIntegerType(mlir::bmodelica::IntegerType type);
  mlir::Type convertRealType(mlir::bmodelica::RealType type);
  mlir::Type convertPointerType(mlir::bmodelica::PointerType type);

  mlir::Type convertIntegerType(mlir::IntegerType type);
  mlir::Type convertFloatType(mlir::FloatType type);

  std::optional<LogicalResult>
  convertArrayType(ArrayType type,
                    llvm::SmallVectorImpl<Type> &converted);
};
} // namespace mlir::bmodelica

#endif // MARCO_CODEGEN_CONVERSION_BASEMODELICACOMMON_CTYPECONVERTER_H