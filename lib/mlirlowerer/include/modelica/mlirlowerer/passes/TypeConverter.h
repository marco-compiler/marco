#pragma once

#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/MLIRContext.h>
#include <modelica/mlirlowerer/Type.h>

namespace modelica
{
	class TypeConverter : public mlir::LLVMTypeConverter
	{
		public:
		TypeConverter(mlir::MLIRContext* context, mlir::LowerToLLVMOptions options) : mlir::LLVMTypeConverter(context, options), context(context)
		{
			addConversion([&](BooleanType type) { return convertBooleanType(type); });
			addConversion([&](IntegerType type) { return convertIntegerType(type); });
			addConversion([&](RealType type) { return convertRealType(type); });
			addConversion([&](PointerType type) { return convertPointerType(type); });

			addTargetMaterialization(
					[&](mlir::OpBuilder &builder, mlir::IntegerType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value> {
						if (inputs.size() != 1)
							return llvm::None;

						if (!inputs[0].getType().isa<BooleanType>() && !inputs[0].getType().isa<IntegerType>())
							return llvm::None;

						return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
					});

			addTargetMaterialization(
					[&](mlir::OpBuilder &builder, mlir::FloatType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value> {
						if (inputs.size() != 1)
							return llvm::None;

						if (!inputs[0].getType().isa<RealType>())
							return llvm::None;

						return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
					});

			addTargetMaterialization(
					[&](mlir::OpBuilder &builder, mlir::LLVM::LLVMStructType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value> {
						if (inputs.size() != 1)
							return llvm::None;

						if (!inputs[0].getType().isa<PointerType>())
							return llvm::None;

						return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
					});

			addSourceMaterialization(
					[&](mlir::OpBuilder &builder, BooleanType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value> {
						if (inputs.size() != 1)
							return llvm::None;

						if (!inputs[0].getType().isa<mlir::IntegerType>() || inputs[0].getType().getIntOrFloatBitWidth() != 1)
							return llvm::None;

						return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
			});

			addSourceMaterialization(
					[&](mlir::OpBuilder &builder, IntegerType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value> {
						if (inputs.size() != 1)
							return llvm::None;

						if (!inputs[0].getType().isa<mlir::IntegerType>() || inputs[0].getType().getIntOrFloatBitWidth() != resultType.getBitWidth())
							return llvm::None;

						return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
					});

			addSourceMaterialization(
					[&](mlir::OpBuilder &builder, RealType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value> {
						if (inputs.size() != 1)
							return llvm::None;

						if (!inputs[0].getType().isa<mlir::FloatType>() || inputs[0].getType().getIntOrFloatBitWidth() != resultType.getBitWidth())
							return llvm::None;

						return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
					});

			addSourceMaterialization(
					[&](mlir::OpBuilder &builder, PointerType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value> {
						if (inputs.size() != 1)
							return llvm::None;

						if (!inputs[0].getType().isa<mlir::LLVM::LLVMStructType>())
							return llvm::None;

						return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
					});
		}

		[[nodiscard]] mlir::Type indexType();

		[[nodiscard]] mlir::Type voidPtrType();

		private:
		mlir::Type convertBooleanType(BooleanType type);
		mlir::Type convertIntegerType(IntegerType type);
		mlir::Type convertRealType(RealType type);
		mlir::Type convertPointerType(PointerType type);

		llvm::SmallVector<mlir::Type, 3> getPointerDescriptorFields(PointerType type);

		mlir::MLIRContext* context;
	};
}
