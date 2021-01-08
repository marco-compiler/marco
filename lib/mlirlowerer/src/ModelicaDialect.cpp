#include <modelica/mlirlowerer/ModelicaDialect.hpp>
#include <modelica/mlirlowerer/Ops.hpp>

using namespace modelica;

ModelicaDialect::ModelicaDialect(mlir::MLIRContext* context)
		: Dialect("modelica", context, mlir::TypeID::get<ModelicaDialect>())
{
	// Math operations
	addOperations<NegateOp>();
	addOperations<AddOp>();
	addOperations<SubOp>();
	addOperations<MulOp>();
	addOperations<DivOp>();

	// Comparison operations
	addOperations<EqOp>();
	addOperations<NotEqOp>();
	addOperations<GtOp>();
	addOperations<GteOp>();
	addOperations<LtOp>();
	addOperations<LteOp>();

	// Control flow operations
	addOperations<IfOp>();
	addOperations<WhileOp>();
	addOperations<ConditionOp>();
	addOperations<YieldOp>();
	addOperations<BreakOp>();
}

mlir::StringRef ModelicaDialect::getDialectNamespace()
{
	return "modelica";
}
