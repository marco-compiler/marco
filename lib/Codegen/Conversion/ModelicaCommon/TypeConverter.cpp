#include "marco/Codegen/Conversion/ModelicaCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace ::mlir::modelica;

namespace mlir::modelica
{
  TypeConverter::TypeConverter(unsigned int bitWidth)
    : bitWidth(bitWidth)
  {
    addConversion([](mlir::Type type) {
      return type;
    });

    addConversion([&](BooleanType type) {
      return convertBooleanType(type);
    });

    addConversion([&](IntegerType type) {
      return convertIntegerType(type);
    });

    addConversion([&](RealType type) {
      return convertRealType(type);
    });

    addConversion([&](ArrayType type) {
      return convertArrayType(type);
    });

    addConversion([&](UnrankedArrayType type) {
      return convertUnrankedArrayType(type);
    });

    addConversion([&](mlir::IndexType type) {
      return type;
    });

    addTargetMaterialization(
        [&](mlir::OpBuilder& builder,
            mlir::Type resultType,
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
        [&](mlir::OpBuilder& builder,
            mlir::Type resultType,
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

  mlir::Type TypeConverter::convertBooleanType(BooleanType type)
  {
    return mlir::IntegerType::get(type.getContext(), 1);
  }

  mlir::Type TypeConverter::convertIntegerType(IntegerType type)
  {
    return mlir::IntegerType::get(type.getContext(), bitWidth);
  }

  mlir::Type TypeConverter::convertRealType(RealType type)
  {
    if (bitWidth == 16) {
      return mlir::Float16Type::get(type.getContext());
    }

    if (bitWidth == 32) {
      return mlir::Float32Type::get(type.getContext());
    }

    if (bitWidth == 64) {
      return mlir::Float64Type::get(type.getContext());
    }

    llvm_unreachable("Unsupported bit-width for real type");
    return {};
  }

  mlir::Type TypeConverter::convertArrayType(ArrayType type)
  {
    auto shape = type.getShape();
    auto elementType = convertType(type.getElementType());
    return mlir::MemRefType::get(shape, elementType);
  }

  mlir::Type TypeConverter::convertUnrankedArrayType(UnrankedArrayType type)
  {
    auto elementType = convertType(type.getElementType());
    return mlir::UnrankedMemRefType::get(elementType, type.getMemorySpace());
  }
}
