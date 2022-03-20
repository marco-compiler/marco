#include "marco/Codegen/Conversion/IDA/TypeConverter.h"

namespace mlir::ida
{
  TypeConverter::TypeConverter(mlir::MLIRContext* context, mlir::LowerToLLVMOptions options)
      : mlir::LLVMTypeConverter(context, options)
  {
    addConversion([&](InstanceType type) {
      return convertInstanceType(type);
    });

    addTargetMaterialization([&](mlir::OpBuilder& builder, mlir::LLVM::LLVMPointerType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value> {
      return opaquePointerTypeTargetMaterialization(builder, resultType, inputs, loc);
    });

    addSourceMaterialization([&](mlir::OpBuilder& builder, InstanceType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value> {
      return instanceTypeSourceMaterialization(builder, resultType, inputs, loc);
    });
  }

  mlir::Type TypeConverter::convertInstanceType(InstanceType type)
  {
    auto integerType = convertType(mlir::IntegerType::get(type.getContext(), 8));
    return mlir::LLVM::LLVMPointerType::get(integerType);
  }

  mlir::Type TypeConverter::convertEquationType(EquationType type)
  {
    return getIndexType();
  }

  llvm::Optional<mlir::Value> TypeConverter::opaquePointerTypeTargetMaterialization(
      mlir::OpBuilder& builder, mlir::LLVM::LLVMPointerType resultType, mlir::ValueRange inputs, mlir::Location loc) const
  {
    if (inputs.size() != 1) {
      return llvm::None;
    }

    if (!inputs[0].getType().isa<InstanceType>()) {
      return llvm::None;
    }

    auto elementType = resultType.getElementType().dyn_cast<mlir::IntegerType>();

    if (!elementType || elementType.getIntOrFloatBitWidth() != 8) {
      return llvm::None;
    }

    return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
  }

  llvm::Optional<mlir::Value> TypeConverter::instanceTypeSourceMaterialization(
      mlir::OpBuilder& builder, InstanceType resultType, mlir::ValueRange inputs, mlir::Location loc) const
  {
    if (inputs.size() != 1) {
      return llvm::None;
    }

    auto pointerType = inputs[0].getType().dyn_cast<mlir::LLVM::LLVMPointerType>();

    if (!pointerType) {
      return llvm::None;
    }

    auto elementType = pointerType.getElementType().dyn_cast<mlir::IntegerType>();

    if (!elementType || elementType.getIntOrFloatBitWidth() != 8) {
      return llvm::None;
    }

    return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
  }
}
