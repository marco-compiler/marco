#ifndef MARCO_CODEGEN_CONVERSION_MODELICACOMMON_TYPECONVERTER_H
#define MARCO_CODEGEN_CONVERSION_MODELICACOMMON_TYPECONVERTER_H

#include "marco/Dialect/IDA/IDADialect.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::modelica
{
  // We inherit from the LLVMTypeConverter in order to retrieve the converted MLIR index type.
	class TypeConverter : public mlir::LLVMTypeConverter
  {
		public:
      TypeConverter(mlir::MLIRContext* context, mlir::LowerToLLVMOptions options, unsigned int bitWidth);

      mlir::Type convertBooleanType(mlir::modelica::BooleanType type);
      mlir::Type convertIntegerType(mlir::modelica::IntegerType type);
      mlir::Type convertRealType(mlir::modelica::RealType type);
      mlir::Type convertArrayType(mlir::modelica::ArrayType type);

      llvm::Optional<mlir::Value> integerTypeTargetMaterialization(
          mlir::OpBuilder& builder, mlir::IntegerType resultType, mlir::ValueRange inputs, mlir::Location loc) const;

      llvm::Optional<mlir::Value> floatTypeTargetMaterialization(
          mlir::OpBuilder& builder, mlir::FloatType resultType, mlir::ValueRange inputs, mlir::Location loc) const;

      llvm::Optional<mlir::Value> llvmStructTypeTargetMaterialization(
          mlir::OpBuilder& builder, mlir::LLVM::LLVMStructType resultType, mlir::ValueRange inputs, mlir::Location loc) const;

      llvm::Optional<mlir::Value> booleanTypeSourceMaterialization(
          mlir::OpBuilder& builder, mlir::modelica::BooleanType resultType, mlir::ValueRange inputs, mlir::Location loc) const;

      llvm::Optional<mlir::Value> integerTypeSourceMaterialization(
          mlir::OpBuilder& builder, mlir::modelica::IntegerType resultType, mlir::ValueRange inputs, mlir::Location loc) const;

      llvm::Optional<mlir::Value> realTypeSourceMaterialization(
          mlir::OpBuilder& builder, mlir::modelica::RealType resultType, mlir::ValueRange inputs, mlir::Location loc) const;

      llvm::Optional<mlir::Value> arrayTypeSourceMaterialization(
          mlir::OpBuilder& builder, mlir::modelica::ArrayType resultType, mlir::ValueRange inputs, mlir::Location loc) const;

    private:
      llvm::SmallVector<mlir::Type, 3> getArrayDescriptorFields(mlir::modelica::ArrayType type);

    private:
		  unsigned int bitWidth;
	};
}

#endif // MARCO_CODEGEN_CONVERSION_MODELICACOMMON_TYPECONVERTER_H
