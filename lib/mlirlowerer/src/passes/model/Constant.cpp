#include <modelica/mlirlowerer/passes/model/Constant.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>

using namespace modelica::codegen::model;

Constant::Constant(mlir::Value value) : value(value)
{
	assert(mlir::isa<ConstantOp>(value.getDefiningOp()));
}

size_t Constant::childrenCount() const
{
	return 0;
}
