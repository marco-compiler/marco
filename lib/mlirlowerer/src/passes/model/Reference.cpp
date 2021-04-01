#include <modelica/mlirlowerer/passes/model/Reference.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>

using namespace modelica::codegen::model;

Reference::Reference(mlir::Value var) : var(var)
{
	assert(mlir::isa<AllocaOp>(var.getDefiningOp()) ||
	    mlir::isa<AllocOp>(var.getDefiningOp()));
}

mlir::Value Reference::getVar() const
{
	return var;
}

size_t Reference::childrenCount() const
{
	return 0;
}
