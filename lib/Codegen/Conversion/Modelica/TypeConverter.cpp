#include "marco/Codegen/Conversion/Modelica/TypeConverter.h"

using namespace ::mlir::modelica;

namespace mlir::modelica
{
  TypeConverter::TypeConverter(mlir::MLIRContext* context, mlir::LowerToLLVMOptions options, unsigned int bitWidth)
      : mlir::LLVMTypeConverter(context, options), bitWidth(bitWidth)
  {
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

    addConversion([&](UnsizedArrayType type) {
      return convertUnsizedArrayType(type);
    });

    addTargetMaterialization([&](mlir::OpBuilder& builder, mlir::IntegerType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value> {
      return integerTypeTargetMaterialization(builder, resultType, inputs, loc);
    });

    addTargetMaterialization([&](mlir::OpBuilder& builder, mlir::FloatType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value> {
      return floatTypeTargetMaterialization(builder, resultType, inputs, loc);
    });

    addTargetMaterialization([&](mlir::OpBuilder& builder, mlir::LLVM::LLVMStructType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value> {
      return llvmStructTypeTargetMaterialization(builder, resultType, inputs, loc);
    });

    addSourceMaterialization([&](mlir::OpBuilder& builder, BooleanType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value> {
      return booleanTypeSourceMaterialization(builder, resultType, inputs, loc);
    });

    addSourceMaterialization([&](mlir::OpBuilder& builder, IntegerType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value> {
      return integerTypeSourceMaterialization(builder, resultType, inputs, loc);
    });

    addSourceMaterialization([&](mlir::OpBuilder& builder, RealType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value> {
      return realTypeSourceMaterialization(builder, resultType, inputs, loc);
    });

    addSourceMaterialization([&](mlir::OpBuilder& builder, ArrayType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value> {
      return arrayTypeSourceMaterialization(builder, resultType, inputs, loc);
    });

    addSourceMaterialization([&](mlir::OpBuilder& builder, UnsizedArrayType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value> {
      return unsizedArrayTypeSourceMaterialization(builder, resultType, inputs, loc);
    });
  }

  mlir::Type TypeConverter::convertBooleanType(BooleanType type)
  {
    return mlir::IntegerType::get(&getContext(), 1);
  }

  mlir::Type TypeConverter::convertIntegerType(IntegerType type)
  {
    return mlir::IntegerType::get(&getContext(), bitWidth);
  }

  mlir::Type TypeConverter::convertRealType(RealType type)
  {
    if (bitWidth == 16) {
      return convertType(mlir::Float16Type::get(&getContext()));
    }

    if (bitWidth == 32) {
      return convertType(mlir::Float32Type::get(&getContext()));
    }

    if (bitWidth == 64) {
      return convertType(mlir::Float64Type::get(&getContext()));
    }

    llvm_unreachable("Unsupported bit-width for real type");
    return {};
  }

  mlir::Type TypeConverter::convertArrayType(ArrayType type)
  {
    auto types = getArrayDescriptorFields(type);
    return mlir::LLVM::LLVMStructType::getLiteral(type.getContext(), types);
  }

  mlir::Type TypeConverter::convertUnsizedArrayType(UnsizedArrayType type)
  {
    auto types = getUnsizedArrayDescriptorFields(type);
    return mlir::LLVM::LLVMStructType::getLiteral(type.getContext(), types);
  }

  llvm::Optional<mlir::Value> TypeConverter::integerTypeTargetMaterialization(
      mlir::OpBuilder& builder, mlir::IntegerType resultType, mlir::ValueRange inputs, mlir::Location loc) const
  {
    if (inputs.size() != 1) {
      return llvm::None;
    }

    // Also the BooleanType is admitted, because in MLIR it is represented by the i1 type.
    if (!inputs[0].getType().isa<BooleanType>() && !inputs[0].getType().isa<IntegerType>()) {
      return llvm::None;
    }

    return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
  }

  llvm::Optional<mlir::Value> TypeConverter::floatTypeTargetMaterialization(
      mlir::OpBuilder& builder, mlir::FloatType resultType, mlir::ValueRange inputs, mlir::Location loc) const
  {
    if (inputs.size() != 1) {
      return llvm::None;
    }

    if (!inputs[0].getType().isa<RealType>()) {
      return llvm::None;
    }

    return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
  }

  llvm::Optional<mlir::Value> TypeConverter::llvmStructTypeTargetMaterialization(
      mlir::OpBuilder& builder, mlir::LLVM::LLVMStructType resultType, mlir::ValueRange inputs, mlir::Location loc) const
  {
    if (inputs.size() != 1) {
      return llvm::None;
    }

    if (!inputs[0].getType().isa<ArrayType>() && !inputs[0].getType().isa<UnsizedArrayType>()) {
      return llvm::None;
    }

    return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
  }

  llvm::Optional<mlir::Value> TypeConverter::booleanTypeSourceMaterialization(
      mlir::OpBuilder& builder, BooleanType resultType, mlir::ValueRange inputs, mlir::Location loc) const
  {
    if (inputs.size() != 1) {
      return llvm::None;
    }

    if (!inputs[0].getType().isa<mlir::IntegerType>() || inputs[0].getType().getIntOrFloatBitWidth() != 1) {
      return llvm::None;
    }

    return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
  }

  llvm::Optional<mlir::Value> TypeConverter::integerTypeSourceMaterialization(
      mlir::OpBuilder& builder, IntegerType resultType, mlir::ValueRange inputs, mlir::Location loc) const
  {
    if (inputs.size() != 1) {
      return llvm::None;
    }

    if (!inputs[0].getType().isa<mlir::IntegerType>() || inputs[0].getType().getIntOrFloatBitWidth() != bitWidth) {
      return llvm::None;
    }

    return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
  }

  llvm::Optional<mlir::Value> TypeConverter::realTypeSourceMaterialization(
      mlir::OpBuilder& builder, RealType resultType, mlir::ValueRange inputs, mlir::Location loc) const
  {
    if (inputs.size() != 1) {
      return llvm::None;
    }

    if (!inputs[0].getType().isa<mlir::FloatType>() || inputs[0].getType().getIntOrFloatBitWidth() != bitWidth) {
      return llvm::None;
    }

    return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
  }

  llvm::Optional<mlir::Value> TypeConverter::arrayTypeSourceMaterialization(
      mlir::OpBuilder& builder, ArrayType resultType, mlir::ValueRange inputs, mlir::Location loc) const
  {
    if (inputs.size() != 1) {
      return llvm::None;
    }

    auto isZeroDimensionalPointer = [](mlir::Type type) -> bool {
      if (auto structType = type.dyn_cast<mlir::LLVM::LLVMStructType>()) {
        if (auto types = structType.getBody(); types.size() == 2) {
          if (types[0].isa<mlir::LLVM::LLVMPointerType>() && types[1].isa<mlir::IntegerType>()) {
            return true;
          }
        }
      }

      return false;
    };

    auto isMultiDimensionalPointer = [](mlir::Type type) -> bool {
      if (auto structType = type.dyn_cast<mlir::LLVM::LLVMStructType>()) {
        if (auto types = structType.getBody(); types.size() == 3) {
          if (types[0].isa<mlir::LLVM::LLVMPointerType>() &&
              types[1].isa<mlir::IntegerType>() &&
              types[2].isa<mlir::LLVM::LLVMArrayType>() &&
              types[2].cast<mlir::LLVM::LLVMArrayType>().getElementType().isa<mlir::IntegerType>()) {
            return true;
          }
        }
      }

      return false;
    };

    if (isZeroDimensionalPointer(inputs[0].getType()) || isMultiDimensionalPointer(inputs[0].getType())) {
      return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
    }

    return llvm::None;
  }

  llvm::Optional<mlir::Value> TypeConverter::unsizedArrayTypeSourceMaterialization(
      mlir::OpBuilder& builder, UnsizedArrayType resultType, mlir::ValueRange inputs, mlir::Location loc) const
  {
    if (inputs.size() != 1) {
      return llvm::None;
    }

    if (auto structType = inputs[0].getType().dyn_cast<mlir::LLVM::LLVMStructType>()) {
      if (auto types = structType.getBody(); types.size() == 2) {
        if (types[0].isa<mlir::IntegerType>() && types[1].isa<mlir::LLVM::LLVMPointerType>()) {
          return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
        }
      }
    }

    return llvm::None;
  }

  llvm::SmallVector<mlir::Type, 3> TypeConverter::getArrayDescriptorFields(ArrayType type)
  {
    mlir::Type elementType = type.getElementType();
    elementType = convertType(elementType);

    auto ptrType = mlir::LLVM::LLVMPointerType::get(elementType);
    auto indexType = getIndexType();
    llvm::SmallVector<mlir::Type, 3> results = { ptrType, indexType };

    auto rank = type.getRank();

    if (rank == 0) {
      return results;
    }

    results.insert(results.end(), 1, mlir::LLVM::LLVMArrayType::get(indexType, rank));
    return results;
  }

  llvm::SmallVector<mlir::Type, 3> TypeConverter::getUnsizedArrayDescriptorFields(UnsizedArrayType type)
  {
    auto indexType = getIndexType();
    auto voidPtr = mlir::LLVM::LLVMPointerType::get(convertType(mlir::IntegerType::get(type.getContext(), 8)));

    llvm::SmallVector<mlir::Type, 3> results = { indexType, voidPtr };
    return results;
  }
}
