#ifndef MARCO_CODEGEN_CONVERSION_KINSOLCOMMON_LLVMTYPECONVERTER_H
#define MARCO_CODEGEN_CONVERSION_KINSOLCOMMON_LLVMTYPECONVERTER_H

#include "marco/Dialect/KINSOL/KINSOLDialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::kinsol
{
  class LLVMTypeConverter : public mlir::LLVMTypeConverter
  {
    public:
      LLVMTypeConverter(
          mlir::MLIRContext* context, const mlir::LowerToLLVMOptions& options);

      mlir::Type convertInstanceType(InstanceType type);
      mlir::Type convertVariableType(VariableType type);
      mlir::Type convertEquationType(EquationType type);

      std::optional<mlir::Value> opaquePointerTypeTargetMaterialization(
          mlir::OpBuilder& builder,
          mlir::LLVM::LLVMPointerType resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) const;

      std::optional<mlir::Value> integerTypeTargetMaterialization(
          mlir::OpBuilder& builder,
          mlir::IntegerType resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) const;

      std::optional<mlir::Value> instanceTypeSourceMaterialization(
          mlir::OpBuilder& builder,
          InstanceType resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) const;

      std::optional<mlir::Value> variableTypeSourceMaterialization(
          mlir::OpBuilder& builder,
          VariableType resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) const;

      std::optional<mlir::Value> equationTypeSourceMaterialization(
          mlir::OpBuilder& builder,
          EquationType resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) const;
  };
}

#endif // MARCO_CODEGEN_CONVERSION_KINSOLCOMMON_LLVMTYPECONVERTER_H
