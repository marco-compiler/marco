#ifndef MARCO_CODEGEN_CONVERSION_KINSOLCOMMON_LLVMTYPECONVERTER_H
#define MARCO_CODEGEN_CONVERSION_KINSOLCOMMON_LLVMTYPECONVERTER_H

#include "marco/Dialect/KINSOL/IR/KINSOL.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::kinsol {
class LLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  LLVMTypeConverter(mlir::MLIRContext *context,
                    const mlir::LowerToLLVMOptions &options);

  mlir::Type convertInstanceType(InstanceType type);
  mlir::Type convertVariableType(VariableType type);
  mlir::Type convertEquationType(EquationType type);
};
} // namespace mlir::kinsol

#endif // MARCO_CODEGEN_CONVERSION_KINSOLCOMMON_LLVMTYPECONVERTER_H
