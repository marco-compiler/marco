#include "marco/Codegen/Conversion/ModelicaCommon/LLVMTypeConverter.h"

using namespace ::mlir::modelica;

namespace mlir::modelica
{
  LLVMTypeConverter::LLVMTypeConverter(
      mlir::MLIRContext* context,
      const mlir::LowerToLLVMOptions& options,
      unsigned int bitWidth)
      : mlir::LLVMTypeConverter(context, options),
        baseTypeConverter(bitWidth),
        llvmTypeConverter(context, options)
  {
    addConversion([&](BooleanType type) {
      return forwardConversion(type);
    });

    addConversion([&](IntegerType type) {
      return forwardConversion(type);
    });

    addConversion([&](RealType type) {
      return forwardConversion(type);
    });

    addConversion([&](ArrayType type) {
      return forwardConversion(type);
    });

    addConversion([&](UnrankedArrayType type) {
      return forwardConversion(type);
    });

    addTargetMaterialization(
        [&](mlir::OpBuilder& builder,
            mlir::Type resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> std::optional<mlir::Value> {
          if (inputs.size() != 1) {
            return std::nullopt;

          }

          return builder.create<mlir::UnrealizedConversionCastOp>(
                            loc, resultType, inputs).getResult(0);
        });

    addSourceMaterialization(
        [&](mlir::OpBuilder& builder,
            mlir::Type resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> std::optional<mlir::Value> {
          if (inputs.size() != 1) {
            return std::nullopt;
          }

          return builder.create<mlir::UnrealizedConversionCastOp>(
                            loc, resultType, inputs).getResult(0);
        });
  }

  mlir::Type LLVMTypeConverter::forwardConversion(mlir::Type type)
  {
    return llvmTypeConverter.convertType(baseTypeConverter.convertType(type));
  }
}
