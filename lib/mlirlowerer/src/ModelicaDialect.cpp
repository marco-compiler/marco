#include <modelica/mlirlowerer/ConstantOpOld.hpp>
#include <modelica/mlirlowerer/FunctionOp.hpp>
#include <modelica/mlirlowerer/ModelicaDialect.hpp>
#include <modelica/mlirlowerer/ReturnOp.hpp>

using namespace mlir;
using namespace modelica;

ModelicaDialect::ModelicaDialect(MLIRContext* context)
		: Dialect("modelica", context)
{
	addOperations<ConstantOpOld>();
	addOperations<ReturnOp>();
}

StringRef ModelicaDialect::getDialectNamespace()
{
	return "modelica";
}
