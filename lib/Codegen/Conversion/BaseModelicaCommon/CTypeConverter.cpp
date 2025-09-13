#include "marco/Codegen/Conversion/BaseModelicaCommon/CTypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace ::mlir::bmodelica;

namespace mlir::bmodelica {
CTypeConverter::CTypeConverter(mlir::MLIRContext *context,
                               const mlir::DataLayout &dataLayout) {
  addConversion([](mlir::Type type) { return type; });

  addConversion([&](BooleanType type) { return convertBooleanType(type); });

  addConversion([&](IntegerType type) { return convertIntegerType(type); });

  addConversion([&](RealType type) { return convertRealType(type); });

  addConversion(
      [&](mlir::IntegerType type) { return convertIntegerType(type); });

  addConversion([&](mlir::FloatType type) { return convertFloatType(type); });

  addConversion([&](RealType type) { return convertRealType(type); });

  addConversion(
      [&](ArrayType type, llvm::SmallVectorImpl<mlir::Type> &converted) {
        return convertArrayType(type, converted);
      });

  auto addUnrealizedCast = [](mlir::OpBuilder &builder, mlir::Type type,
                              mlir::ValueRange inputs,
                              mlir::Location loc) -> mlir::Value {
    return builder.create<CastOp>(loc, type, inputs);
  };

  addSourceMaterialization(addUnrealizedCast);
  addTargetMaterialization(addUnrealizedCast);

  addTargetMaterialization([&](mlir::OpBuilder &builder, mlir::TypeRange types,
                               mlir::ValueRange inputs,
                               mlir::Location loc) -> llvm::SmallVector<Value> {
    llvm::SmallVector<Value> results;

    if (inputs.size() == 1) {
      if (auto arrayType = mlir::dyn_cast<ArrayType>(inputs[0].getType())) {
        mlir::Value array = inputs[0];

        if (mlir::Type convertedElementType =
                convertType(arrayType.getElementType());
            convertedElementType != arrayType.getElementType()) {
          llvm::SmallVector<mlir::Value> dynSizes;

          for (unsigned int dim = 0, rank = arrayType.getRank(); dim < rank;
               ++dim) {
            if (arrayType.isDynamicDim(dim)) {
              mlir::Value size = builder.create<mlir::memref::DimOp>(
                  array.getLoc(), array, dim);

              dynSizes.push_back(size);
            }
          }

          mlir::Value newArray = builder.create<AllocOp>(
              array.getLoc(),
              arrayType.cloneWith(arrayType.getShape(), convertedElementType),
              dynSizes);

          builder.create<ArrayCopyOp>(array.getLoc(), array, newArray);
          array = newArray;
        }

        mlir::Value pointer =
            builder.create<ArrayAddressOp>(array.getLoc(), array);

        pointer = materializeTargetConversion(
            builder, pointer.getLoc(),
            convertType(PointerType::get(array.getContext(),
                                         arrayType.getElementType())),
            pointer);

        results.push_back(pointer);
      }
    }

    return results;
  });
}

mlir::Type CTypeConverter::convertBooleanType(BooleanType type) {
  return mlir::IntegerType::get(type.getContext(), booleanBitWidth);
}

mlir::Type CTypeConverter::convertIntegerType(IntegerType type) {
  return mlir::IntegerType::get(type.getContext(), integerBitWidth);
}

mlir::Type CTypeConverter::convertRealType(RealType type) {
  return mlir::Float64Type::get(type.getContext());
}

mlir::Type
CTypeConverter::convertPointerType(mlir::bmodelica::PointerType type) {
  return mlir::LLVM::LLVMPointerType::get(type.getContext());
}

mlir::Type CTypeConverter::convertIntegerType(mlir::IntegerType type) {
  return mlir::IntegerType::get(type.getContext(), integerBitWidth);
}

mlir::Type CTypeConverter::convertFloatType(mlir::FloatType type) {
  return mlir::Float64Type::get(type.getContext());
}

std::optional<LogicalResult>
CTypeConverter::convertArrayType(ArrayType type,
                                  llvm::SmallVectorImpl<Type> &converted) {
  converted.push_back(
      convertType(PointerType::get(type.getContext(), type.getElementType())));

  converted.append(type.getRank(), mlir::IndexType::get(type.getContext()));
  return mlir::success();
}
} // namespace mlir::bmodelica
