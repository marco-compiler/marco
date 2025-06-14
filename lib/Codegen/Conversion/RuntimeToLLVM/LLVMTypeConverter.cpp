#include "marco/Codegen/Conversion/RuntimeToLLVM/LLVMTypeConverter.h"
#include "marco/Dialect/Runtime/IR/Types.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace mlir::runtime {
LLVMTypeConverter::LLVMTypeConverter(mlir::MLIRContext *context,
                                     const mlir::LowerToLLVMOptions &options)
    : mlir::LLVMTypeConverter(context, options) {
  addConversion([&](StringType type) { return convertStringType(type); });

  auto addUnrealizedCast = [](mlir::OpBuilder &builder, mlir::Type type,
                              mlir::ValueRange inputs,
                              mlir::Location loc) -> mlir::Value {
    auto castOp =
        builder.create<mlir::UnrealizedConversionCastOp>(loc, type, inputs);

    return castOp.getResult(0);
  };

  addSourceMaterialization(addUnrealizedCast);
  addTargetMaterialization(addUnrealizedCast);
}

mlir::Type LLVMTypeConverter::convertStringType(StringType type) {
  return mlir::LLVM::LLVMPointerType::get(type.getContext());
}
} // namespace mlir::runtime
