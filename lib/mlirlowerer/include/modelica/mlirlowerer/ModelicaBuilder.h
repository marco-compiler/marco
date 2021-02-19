#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/Optional.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <modelica/mlirlowerer/Attribute.h>
#include <modelica/mlirlowerer/Type.h>

namespace modelica
{
	class ModelicaBuilder : public mlir::OpBuilder
	{
		public:
		using mlir::OpBuilder::OpBuilder;

		BooleanType getBooleanType();

		IntegerType getIntegerType();

		RealType getRealType();

		PointerType getPointerType(mlir::Type elementType, const PointerType::Shape& shape = {}, mlir::AffineMapAttr map = {});

		//IndexAttribute getIndexAttribute(long value);

	};
}
