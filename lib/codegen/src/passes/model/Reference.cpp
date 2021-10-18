#include <marco/mlirlowerer/passes/model/Reference.h>
#include <marco/mlirlowerer/dialects/modelica/ModelicaDialect.h>

using namespace marco::codegen::model;
using namespace marco::codegen::modelica;

Reference::Reference(mlir::Value var) : var(var)
{
	assert(mlir::isa<AllocaOp>(var.getDefiningOp()) ||
	    mlir::isa<AllocOp>(var.getDefiningOp()));
}

mlir::Value Reference::getVar() const
{
	return var;
}
