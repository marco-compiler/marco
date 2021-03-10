#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/Optional.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include "Attribute.h"
#include "Type.h"

namespace modelica
{
	class ModelicaBuilder : public mlir::OpBuilder
	{
		public:
		ModelicaBuilder(mlir::MLIRContext* context, unsigned int bitWidth);

		BooleanType getBooleanType();
		IntegerType getIntegerType();
		RealType getRealType();
		PointerType getPointerType(bool heap, mlir::Type elementType, const PointerType::Shape& shape = {});

		mlir::IntegerAttr getIndexAttribute(long value);
		BooleanAttribute getBooleanAttribute(bool value);
		IntegerAttribute getIntegerAttribute(long value);
		RealAttribute getRealAttribute(double value);

		private:
		unsigned int bitWidth;
	};
}
