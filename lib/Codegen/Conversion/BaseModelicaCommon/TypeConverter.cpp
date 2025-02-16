#include "marco/Codegen/Conversion/BaseModelicaCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace ::mlir::bmodelica;

namespace mlir::bmodelica {
TypeConverter::TypeConverter(mlir::MLIRContext *context,
                             const mlir::DataLayout &dataLayout) {
  integerBitWidth = dataLayout.getTypeSizeInBits(IntegerType::get(context));
  realBitWidth = dataLayout.getTypeSizeInBits(RealType::get(context));

  addConversion([](mlir::Type type) { return type; });

  addConversion([&](BooleanType type) { return convertBooleanType(type); });

  addConversion([&](IntegerType type) { return convertIntegerType(type); });

  addConversion([&](RealType type) { return convertRealType(type); });

  addConversion([&](ArrayType type) { return convertArrayType(type); });

  addConversion(
      [&](UnrankedArrayType type) { return convertUnrankedArrayType(type); });

  addConversion([&](mlir::TensorType type) { return convertTensorType(type); });

  addTargetMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        if (inputs.size() != 1) {
          return std::nullopt;
        }

        auto castOp = builder.create<mlir::UnrealizedConversionCastOp>(
            loc, resultType, inputs);

        return castOp.getResult(0);
      });

  addSourceMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        if (inputs.size() != 1) {
          return std::nullopt;
        }

        auto castOp = builder.create<mlir::UnrealizedConversionCastOp>(
            loc, resultType, inputs);

        return castOp.getResult(0);
      });
}

mlir::Type TypeConverter::convertBooleanType(BooleanType type) {
  return mlir::IntegerType::get(type.getContext(), 1);
}

mlir::Type TypeConverter::convertIntegerType(IntegerType type) {
  return mlir::IntegerType::get(type.getContext(), integerBitWidth);
}

mlir::Type TypeConverter::convertRealType(RealType type) {
  if (realBitWidth <= 16) {
    return mlir::FloatType::getF16(type.getContext());
  }

  if (realBitWidth <= 32) {
    return mlir::FloatType::getF32(type.getContext());
  }

  if (realBitWidth <= 64) {
    return mlir::FloatType::getF64(type.getContext());
  }

  if (realBitWidth <= 80) {
    return mlir::FloatType::getF80(type.getContext());
  }

  return mlir::FloatType::getF128(type.getContext());
  ;
}

mlir::Type TypeConverter::convertArrayType(ArrayType type) {
  auto shape = type.getShape();
  auto elementType = convertType(type.getElementType());
  return mlir::MemRefType::get(shape, elementType);
}

mlir::Type TypeConverter::convertUnrankedArrayType(UnrankedArrayType type) {
  auto elementType = convertType(type.getElementType());
  return mlir::UnrankedMemRefType::get(elementType, type.getMemorySpace());
}

mlir::Type TypeConverter::convertTensorType(mlir::TensorType type) {
  return type.clone(convertType(type.getElementType()));
}
} // namespace mlir::bmodelica
