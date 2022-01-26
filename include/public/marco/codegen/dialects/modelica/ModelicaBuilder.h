#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "marco/codegen/dialects/modelica/Attribute.h"
#include "marco/codegen/dialects/modelica/Type.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace marco::codegen::modelica
{
	class ModelicaBuilder : public mlir::OpBuilder
	{
		public:
		ModelicaBuilder(mlir::MLIRContext* context);

		BooleanType getBooleanType();
		IntegerType getIntegerType();
		RealType getRealType();
		ArrayType getArrayType(BufferAllocationScope allocationScope, mlir::Type elementType, const Shape& shape = {});
		OpaquePointerType getOpaquePointerType();
		StructType getStructType(llvm::ArrayRef<mlir::Type> types);

		mlir::IntegerAttr getIndexAttribute(long value);

		BooleanAttribute getBooleanAttribute(bool value);
		//BooleanArrayAttribute getBooleanArrayAttribute(llvm::ArrayRef<bool> values);

		IntegerAttribute getIntegerAttribute(long value);
		//IntegerArrayAttribute getIntegerArrayAttribute(llvm::ArrayRef<long> values);

		RealAttribute getRealAttribute(double value);
		//RealArrayAttribute getRealArrayAttribute(llvm::ArrayRef<double> values);

		mlir::Attribute getZeroAttribute(mlir::Type type);

		InverseFunctionsAttribute getInverseFunctionsAttribute(InverseFunctionsAttribute::Map map);
		DerivativeAttribute getDerivativeAttribute(llvm::StringRef name, unsigned int order);
	};
}
