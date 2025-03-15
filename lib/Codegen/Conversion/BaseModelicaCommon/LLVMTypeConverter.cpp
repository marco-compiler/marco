#include "marco/Codegen/Conversion/BaseModelicaCommon/LLVMTypeConverter.h"

using namespace ::mlir::bmodelica;

namespace mlir::bmodelica {
LLVMTypeConverter::LLVMTypeConverter(mlir::MLIRContext *context,
                                     const mlir::DataLayout &dataLayout,
                                     const mlir::LowerToLLVMOptions &options)
    : mlir::LLVMTypeConverter(context, options),
      baseTypeConverter(context, dataLayout),
      llvmTypeConverter(context, options) {
  addConversion([&](BooleanType type) { return forwardConversion(type); });

  addConversion([&](IntegerType type) { return forwardConversion(type); });

  addConversion([&](RealType type) { return forwardConversion(type); });

  addConversion([&](ArrayType type) { return forwardConversion(type); });

  addConversion(
      [&](UnrankedArrayType type) { return forwardConversion(type); });

  addConversion([&](RangeType type) { return convertRangeType(type); });

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

mlir::Type LLVMTypeConverter::forwardConversion(mlir::Type type) {
  return llvmTypeConverter.convertType(baseTypeConverter.convertType(type));
}

mlir::Type LLVMTypeConverter::convertRangeType(RangeType type) {
  mlir::Type inductionType = convertType(type.getInductionType());
  llvm::SmallVector<mlir::Type, 3> structTypes(3, inductionType);

  return mlir::LLVM::LLVMStructType::getLiteral(type.getContext(), structTypes);
}
} // namespace mlir::bmodelica
