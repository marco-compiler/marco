#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include "Attribute.h"
#include "Type.h"

namespace marco::codegen::ida
{
	class IdaBuilder : public mlir::OpBuilder
	{
		public:
		IdaBuilder(mlir::MLIRContext* context);

		BooleanType getBooleanType();
		IntegerType getIntegerType();
		RealType getRealType();

		BooleanAttribute getBooleanAttribute(bool value);
		IntegerAttribute getIntegerAttribute(long value);
		RealAttribute getRealAttribute(double value);
	};
}
