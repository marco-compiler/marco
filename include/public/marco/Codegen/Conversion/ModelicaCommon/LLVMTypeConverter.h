#ifndef MARCO_CODEGEN_CONVERSION_MODELICACOMMON_LLVMTYPECONVERTER_H
#define MARCO_CODEGEN_CONVERSION_MODELICACOMMON_LLVMTYPECONVERTER_H

#include "marco/Dialect/BaseModelica/ModelicaDialect.h"
#include "marco/Codegen/Conversion/ModelicaCommon/TypeConverter.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::bmodelica
{
  class LLVMTypeConverter : public mlir::LLVMTypeConverter
  {
    public:
      LLVMTypeConverter(
        mlir::MLIRContext* context,
        const mlir::LowerToLLVMOptions& options,
        unsigned int bitWidth);

    private:
      mlir::Type forwardConversion(mlir::Type type);

      mlir::Type convertRangeType(mlir::bmodelica::RangeType type);

    private:
      unsigned int bitWidth;
      mlir::bmodelica::TypeConverter baseTypeConverter;
      mlir::LLVMTypeConverter llvmTypeConverter;
  };
}

#endif // MARCO_CODEGEN_CONVERSION_MODELICACOMMON_LLVMTYPECONVERTER_H
