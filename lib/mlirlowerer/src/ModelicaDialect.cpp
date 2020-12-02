#include <modelica/mlirlowerer/ModelicaDialect.hpp>
#include <modelica/mlirlowerer/FunctionOp.hpp>

using namespace mlir;
using namespace modelica;

//ModelicaDialect::ModelicaDialect(MLIRContext* context)
//		: Dialect("modelica", context)
//{
//}

StringRef ModelicaDialect::getDialectNamespace()
{
	return "modelica";
}
