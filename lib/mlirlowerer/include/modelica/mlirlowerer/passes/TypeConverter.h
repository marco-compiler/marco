#pragma once

#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/MLIRContext.h>
#include <modelica/mlirlowerer/Type.h>

namespace modelica
{
	class TypeConverter : public mlir::LLVMTypeConverter
	{
		public:
		TypeConverter(mlir::MLIRContext* context, mlir::LowerToLLVMOptions options) : mlir::LLVMTypeConverter(context, options)
		{
			addConversion([&](BooleanType type) { return convertBooleanType(type); });
			addConversion([&](IntegerType type) { return convertIntegerType(type); });
			addConversion([&](RealType type) { return convertRealType(type); });
			addConversion([&](PointerType type) { return convertPointerType(type); });

			// FIXME: https://reviews.llvm.org/D82831 introduced an automatic
			// materialization of conversion around function calls that is not working
			// well with modelica lowering to llvm (incorrect llvm.mlir.cast are inserted).
			// Workaround until better analysis: register a handler that does not insert
			// any conversions.
			addSourceMaterialization(
					[&](mlir::OpBuilder &builder, mlir::Type resultType,
							mlir::ValueRange inputs,
							mlir::Location loc) -> llvm::Optional<mlir::Value> {
						if (inputs.size() != 1)
							return llvm::None;
						return inputs[0];
					});

			// Similar FIXME workaround here
			addTargetMaterialization(
					[&](mlir::OpBuilder &builder, mlir::Type resultType,
							mlir::ValueRange inputs,
							mlir::Location loc) -> llvm::Optional<mlir::Value> {
						if (inputs.size() != 1)
							return llvm::None;
						return inputs[0];
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
	};
}
