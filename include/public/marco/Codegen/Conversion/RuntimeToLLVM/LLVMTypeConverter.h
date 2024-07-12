#ifndef MARCO_CODEGEN_CONVERSION_RUNTIMETOLLVM_LLVMTYPECONVERTER
#define MARCO_CODEGEN_CONVERSION_RUNTIMETOLLVM_LLVMTYPECONVERTER


#include "marco/Dialect/Runtime/IR/Types.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir::runtime
{
  class LLVMTypeConverter : public mlir::LLVMTypeConverter
  {
    public:
      LLVMTypeConverter(
          mlir::MLIRContext* context,
          const mlir::LowerToLLVMOptions& options);

    private:
      mlir::Type convertRuntimeStringType(mlir::runtime::RuntimeStringType type);
  };
} // namespace mlir::runtime



#endif /* end of include guard: MARCO_CODEGEN_CONVERSION_RUNTIMETOLLVM_LLVMTYPECONVERTER */
