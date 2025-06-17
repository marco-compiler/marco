#ifndef MARCO_CODEGEN_CONVERSION_IDACOMMON_LLVMTYPECONVERTER_H
#define MARCO_CODEGEN_CONVERSION_IDACOMMON_LLVMTYPECONVERTER_H

#include "marco/Dialect/IDA/IR/IDA.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::ida {
// We inherit from the LLVMTypeConverter in order to retrieve the converted MLIR
// index type.
class LLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  LLVMTypeConverter(mlir::MLIRContext *context,
                    const mlir::LowerToLLVMOptions &options);

  mlir::Type convertInstanceType(InstanceType type);
  mlir::Type convertVariableType(VariableType type);
  mlir::Type convertEquationType(EquationType type);
};
} // namespace mlir::ida

#endif // MARCO_CODEGEN_CONVERSION_IDACOMMON_LLVMTYPECONVERTER_H
