#pragma once

#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/MLIRContext.h>
#include <modelica/mlirlowerer/Type.h>

namespace modelica::codegen
{
	class TypeConverter : public mlir::LLVMTypeConverter
	{
		public:
		TypeConverter(mlir::MLIRContext* context, mlir::LowerToLLVMOptions options, unsigned int bitWidth);

		private:
		mlir::Type convertBooleanType(BooleanType type);
		mlir::Type convertIntegerType(IntegerType type);
		mlir::Type convertRealType(RealType type);
		mlir::Type convertArrayType(ArrayType type);
		mlir::Type convertUnsizedArrayType(UnsizedArrayType type);
		mlir::Type convertOpaquePointerType(OpaquePointerType type);
		mlir::Type convertStructType(StructType type);

		llvm::SmallVector<mlir::Type, 3> getArrayDescriptorFields(ArrayType type);
		llvm::SmallVector<mlir::Type, 3> getUnsizedArrayDescriptorFields(UnsizedArrayType type);

		unsigned int bitWidth;
	};
}
