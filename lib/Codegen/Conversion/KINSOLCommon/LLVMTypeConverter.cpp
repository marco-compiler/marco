#include "marco/Codegen/Conversion/KINSOLCommon/LLVMTypeConverter.h"

namespace mlir::kinsol
{
  LLVMTypeConverter::LLVMTypeConverter(
      mlir::MLIRContext* context,
      const mlir::LowerToLLVMOptions& options)
      : mlir::LLVMTypeConverter(context, options)
  {
    addConversion([&](InstanceType type) {
      return convertInstanceType(type);
    });

    addConversion([&](VariableType type) {
      return convertVariableType(type);
    });

    addConversion([&](EquationType type) {
      return convertEquationType(type);
    });

    addTargetMaterialization(
        [&](mlir::OpBuilder& builder,
            mlir::LLVM::LLVMPointerType resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> std::optional<mlir::Value> {
      return opaquePointerTypeTargetMaterialization(
              builder, resultType, inputs, loc);
    });

    addTargetMaterialization(
        [&](mlir::OpBuilder& builder,
            mlir::IntegerType resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> std::optional<mlir::Value> {
      return integerTypeTargetMaterialization(
              builder, resultType, inputs, loc);
    });

    addSourceMaterialization(
        [&](mlir::OpBuilder& builder,
            InstanceType resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> std::optional<mlir::Value> {
      return instanceTypeSourceMaterialization(
              builder, resultType, inputs, loc);
    });

    addSourceMaterialization(
        [&](mlir::OpBuilder& builder,
            VariableType resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> std::optional<mlir::Value> {
      return variableTypeSourceMaterialization(
              builder, resultType, inputs, loc);
    });

    addSourceMaterialization(
        [&](mlir::OpBuilder& builder,
            EquationType resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> std::optional<mlir::Value> {
      return equationTypeSourceMaterialization(
              builder, resultType, inputs, loc);
    });
  }

  mlir::Type LLVMTypeConverter::convertInstanceType(InstanceType type)
  {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  }

  mlir::Type LLVMTypeConverter::convertVariableType(VariableType type)
  {
    return mlir::IntegerType::get(type.getContext(), 64);
  }

  mlir::Type LLVMTypeConverter::convertEquationType(EquationType type)
  {
    return mlir::IntegerType::get(type.getContext(), 64);
  }

  std::optional<mlir::Value>
  LLVMTypeConverter::opaquePointerTypeTargetMaterialization(
      mlir::OpBuilder& builder,
      mlir::LLVM::LLVMPointerType resultType,
      mlir::ValueRange inputs,
      mlir::Location loc) const
  {
    if (inputs.size() != 1) {
      return std::nullopt;
    }

    if (!inputs[0].getType().isa<InstanceType>()) {
      return std::nullopt;
    }

    auto elementType =
        resultType.getElementType().dyn_cast<mlir::IntegerType>();

    if (!elementType || elementType.getIntOrFloatBitWidth() != 8) {
      return std::nullopt;
    }

    return builder.create<mlir::UnrealizedConversionCastOp>(
                      loc, resultType, inputs[0]).getResult(0);
  }

  std::optional<mlir::Value>
  LLVMTypeConverter::integerTypeTargetMaterialization(
      mlir::OpBuilder& builder,
      mlir::IntegerType resultType,
      mlir::ValueRange inputs,
      mlir::Location loc) const
  {
    if (inputs.size() != 1) {
      return std::nullopt;
    }

    if (!inputs[0].getType().isa<VariableType, EquationType>()) {
      return std::nullopt;
    }

    return builder.create<mlir::UnrealizedConversionCastOp>(
                      loc, resultType, inputs[0]).getResult(0);
  }

  std::optional<mlir::Value>
  LLVMTypeConverter::instanceTypeSourceMaterialization(
      mlir::OpBuilder& builder,
      InstanceType resultType,
      mlir::ValueRange inputs,
      mlir::Location loc) const
  {
    if (inputs.size() != 1) {
      return std::nullopt;
    }

    auto pointerType =
        inputs[0].getType().dyn_cast<mlir::LLVM::LLVMPointerType>();

    if (!pointerType) {
      return std::nullopt;
    }

    auto elementType =
        pointerType.getElementType().dyn_cast<mlir::IntegerType>();

    if (!elementType || elementType.getIntOrFloatBitWidth() != 8) {
      return std::nullopt;
    }

    return builder.create<mlir::UnrealizedConversionCastOp>(
                      loc, resultType, inputs[0]).getResult(0);
  }

  std::optional<mlir::Value>
  LLVMTypeConverter::variableTypeSourceMaterialization(
      mlir::OpBuilder& builder,
      VariableType resultType,
      mlir::ValueRange inputs,
      mlir::Location loc) const
  {
    if (inputs.size() != 1) {
      return std::nullopt;
    }

    if (!inputs[0].getType().isa<mlir::IntegerType>()) {
      return std::nullopt;
    }

    return builder.create<mlir::UnrealizedConversionCastOp>(
                      loc, resultType, inputs[0]).getResult(0);
  }

  std::optional<mlir::Value>
  LLVMTypeConverter::equationTypeSourceMaterialization(
      mlir::OpBuilder& builder,
      EquationType resultType,
      mlir::ValueRange inputs,
      mlir::Location loc) const
  {
    if (inputs.size() != 1) {
      return std::nullopt;
    }

    if (!inputs[0].getType().isa<mlir::IntegerType>()) {
      return std::nullopt;
    }

    return builder.create<mlir::UnrealizedConversionCastOp>(
                      loc, resultType, inputs[0]).getResult(0);
  }
}
