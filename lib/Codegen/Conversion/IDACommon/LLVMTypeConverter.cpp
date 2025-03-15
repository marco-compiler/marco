#include "marco/Codegen/Conversion/IDACommon/LLVMTypeConverter.h"

namespace mlir::ida {
LLVMTypeConverter::LLVMTypeConverter(mlir::MLIRContext *context,
                                     const mlir::LowerToLLVMOptions &options)
    : mlir::LLVMTypeConverter(context, options) {
  addConversion([&](InstanceType type) { return convertInstanceType(type); });

  addConversion([&](VariableType type) { return convertVariableType(type); });

  addConversion([&](EquationType type) { return convertEquationType(type); });

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

mlir::Type LLVMTypeConverter::convertInstanceType(InstanceType type) {
  return mlir::LLVM::LLVMPointerType::get(type.getContext());
}

mlir::Type LLVMTypeConverter::convertVariableType(VariableType type) {
  return mlir::IntegerType::get(type.getContext(), 64);
}

mlir::Type LLVMTypeConverter::convertEquationType(EquationType type) {
  return mlir::IntegerType::get(type.getContext(), 64);
}
} // namespace mlir::ida
