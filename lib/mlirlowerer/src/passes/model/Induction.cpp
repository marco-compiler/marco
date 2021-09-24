#include <marco/mlirlowerer/dialects/modelica/ModelicaDialect.h>
#include <marco/mlirlowerer/passes/model/Induction.h>

using namespace marco::codegen::model;

Induction::Induction(mlir::BlockArgument argument) : argument(argument)
{
	assert(mlir::isa<modelica::ForEquationOp>(argument.getOwner()->getParentOp()));
}

mlir::BlockArgument Induction::getArgument() const
{
	return argument;
}
