#include <modelica/mlirlowerer/ModelicaDialect.hpp>
#include <modelica/mlirlowerer/Ops.hpp>

using namespace modelica;

ModelicaDialect::ModelicaDialect(mlir::MLIRContext* context)
		: Dialect("modelica", context, mlir::TypeID::get<ModelicaDialect>())
{
	// Generic operations
	addOperations<AssignmentOp>();
	addOperations<CastOp>();
	addOperations<CastCommonOp>();

	// Math operations
	addOperations<NegateOp>();
	addOperations<AddOp>();
	addOperations<SubOp>();
	addOperations<MulOp>();
	addOperations<CrossProductOp>();
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
	addOperations<ForOp>();
	addOperations<WhileOp>();
	addOperations<ConditionOp>();
	addOperations<YieldOp>();
}

mlir::StringRef ModelicaDialect::getDialectNamespace()
{
	return "modelica";
}
