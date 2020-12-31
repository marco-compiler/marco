#include <modelica/mlirlowerer/ModelicaDialect.hpp>
#include <modelica/mlirlowerer/Ops.hpp>

using namespace mlir;
using namespace modelica;

ModelicaDialect::ModelicaDialect(MLIRContext* context)
		: Dialect("modelica", context, TypeID::get<ModelicaDialect>())
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
	addOperations<WhileOp>();
	addOperations<YieldOp>();
	addOperations<BreakOp>();
}

StringRef ModelicaDialect::getDialectNamespace()
{
	return "modelica";
}
