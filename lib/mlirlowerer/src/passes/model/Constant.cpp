#include <marco/mlirlowerer/passes/model/Constant.h>
#include <marco/mlirlowerer/ModelicaDialect.h>

using namespace marco::codegen::model;

Constant::Constant(mlir::Value value) : value(value)
{
	assert(mlir::isa<ConstantOp>(value.getDefiningOp()));
}

mlir::Value Constant::getValue() const
{
	return value;
}
