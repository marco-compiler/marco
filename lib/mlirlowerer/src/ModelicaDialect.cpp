#include <modelica/mlirlowerer/ModelicaDialect.hpp>

using namespace mlir;
using namespace modelica;

ModelicaDialect::ModelicaDialect(MLIRContext* context)
		: Dialect("modelica", context)
{
}

StringRef ModelicaDialect::getDialectNamespace()
{
	return "modelica";
}
