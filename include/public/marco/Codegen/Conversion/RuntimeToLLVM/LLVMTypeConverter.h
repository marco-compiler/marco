#ifndef MARCO_CODEGEN_CONVERSION_RUNTIMETOLLVM_LLVMTYPECONVERTER_H
#define MARCO_CODEGEN_CONVERSION_RUNTIMETOLLVM_LLVMTYPECONVERTER_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir::runtime {
class LLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  LLVMTypeConverter(mlir::MLIRContext *context,
                    const mlir::LowerToLLVMOptions &options);
};
} // namespace mlir::runtime

#endif // MARCO_CODEGEN_CONVERSION_RUNTIMETOLLVM_LLVMTYPECONVERTER_H
