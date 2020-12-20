#include <modelica/mlirlowerer/ModelicaDialect.hpp>
#include <modelica/mlirlowerer/Ops.hpp>

using namespace mlir;
using namespace modelica;

ModelicaDialect::ModelicaDialect(MLIRContext* context)
		: Dialect("modelica", context, TypeID::get<ModelicaDialect>())
{
	// Math operations
	addOperations<AddOp, SubOp, MulOp, DivOp>();

	// Comparison operations
	addOperations<EqOp, NotEqOp, GtOp, GteOp, LtOp, LteOp>();
}

StringRef ModelicaDialect::getDialectNamespace()
{
	return "modelica";
}
