#include <modelica/mlirlowerer/passes/model/Constant.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>

using namespace modelica::codegen::model;

Constant::Constant(mlir::Value value) : value(value)
{
	assert(mlir::isa<ConstantOp>(value.getDefiningOp()));
}
