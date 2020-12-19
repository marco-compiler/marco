#include <modelica/mlirlowerer/MathOps.hpp>
#include <modelica/mlirlowerer/ModelicaDialect.hpp>

using namespace mlir;
using namespace modelica;

ModelicaDialect::ModelicaDialect(MLIRContext* context)
		: Dialect("modelica", context, TypeID::get<ModelicaDialect>())
{
	addOperations<AddOp>();
	addOperations<SubOp>();
	addOperations<MulOp>();
	addOperations<DivOp>();
}

StringRef ModelicaDialect::getDialectNamespace()
{
	return "modelica";
}
