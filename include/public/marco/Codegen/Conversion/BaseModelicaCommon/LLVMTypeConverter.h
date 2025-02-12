#ifndef MARCO_CODEGEN_CONVERSION_BASEMODELICACOMMON_LLVMTYPECONVERTER_H
#define MARCO_CODEGEN_CONVERSION_BASEMODELICACOMMON_LLVMTYPECONVERTER_H

#include "marco/Codegen/Conversion/BaseModelicaCommon/TypeConverter.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::bmodelica {
class LLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  LLVMTypeConverter(mlir::MLIRContext *context,
                    const mlir::LowerToLLVMOptions &options);

private:
  mlir::Type forwardConversion(mlir::Type type);

  mlir::Type convertRangeType(mlir::bmodelica::RangeType type);

private:
  mlir::bmodelica::TypeConverter baseTypeConverter;
  mlir::LLVMTypeConverter llvmTypeConverter;
};
} // namespace mlir::bmodelica

#endif // MARCO_CODEGEN_CONVERSION_BASEMODELICACOMMON_LLVMTYPECONVERTER_H