#include "marco/Codegen/Conversion/RuntimeToLLVM/LLVMTypeConverter.h"
#include "marco/Dialect/Runtime/IR/Types.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include <mlir/IR/Value.h>

namespace mlir::runtime {

LLVMTypeConverter::LLVMTypeConverter(mlir::MLIRContext *context,
                                     const mlir::LowerToLLVMOptions &options)
    : mlir::LLVMTypeConverter(context, options) {

  addConversion([](RuntimeStringType type) {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });

  addTargetMaterialization([&](mlir::OpBuilder &builder, mlir::Type resultType,
                               mlir::ValueRange inputs,
                               mlir::Location loc) -> mlir::Value {
    if (inputs.size() != 1) {
      return mlir::Value();
    }
    return builder
        .create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });

  addSourceMaterialization([&](mlir::OpBuilder &builder, mlir::Type resultType,
                               mlir::ValueRange inputs,
                               mlir::Location loc) -> mlir::Value {
    if (inputs.size() != 1) {
      return mlir::Value();
    }

    return builder
        .create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
}

} // namespace mlir::runtime
