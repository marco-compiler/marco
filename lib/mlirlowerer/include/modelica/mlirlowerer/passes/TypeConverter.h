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
		TypeConverter(mlir::MLIRContext* context, mlir::LowerToLLVMOptions options);

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
