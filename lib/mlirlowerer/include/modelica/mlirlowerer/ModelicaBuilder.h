#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/Optional.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include "Attribute.h"
#include "Type.h"

namespace modelica::codegen
{
	class ModelicaBuilder : public mlir::OpBuilder
	{
		public:
		ModelicaBuilder(mlir::MLIRContext* context, unsigned int bitWidth);

		BooleanType getBooleanType();
		IntegerType getIntegerType();
		RealType getRealType();
		PointerType getPointerType(BufferAllocationScope allocationScope, mlir::Type elementType, const PointerType::Shape& shape = {});
		OpaquePointerType getOpaquePointerType();
		StructType getStructType(llvm::ArrayRef<mlir::Type> types);

		mlir::IntegerAttr getIndexAttribute(long value);

		BooleanAttribute getBooleanAttribute(bool value);
		//BooleanArrayAttribute getBooleanArrayttribute(llvm::ArrayRef<bool> values);

		IntegerAttribute getIntegerAttribute(long value);
		//IntegerArrayAttribute getIntegerArrayAttribute(llvm::ArrayRef<long> values);

		RealAttribute getRealAttribute(double value);
		//RealArrayAttribute getRealArrayAttribute(llvm::ArrayRef<double> values);

		mlir::Attribute getZeroAttribute(mlir::Type type);

		private:
		unsigned int bitWidth;
	};
}
