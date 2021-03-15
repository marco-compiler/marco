#include <modelica/mlirlowerer/ModelicaDialect.h>

using namespace modelica;

ModelicaDialect::ModelicaDialect(mlir::MLIRContext* context)
		: Dialect("modelica", context, mlir::TypeID::get<ModelicaDialect>())
{
	addTypes<BooleanType, IntegerType, RealType, PointerType>();
	addAttributes<BooleanAttribute, IntegerAttribute, RealAttribute>();

	// Basic operations
	addOperations<ConstantOp>();
	addOperations<AssignmentOp>();
	addOperations<CastOp>();
	addOperations<CastCommonOp>();

	// MMemory operations
	addOperations<AllocaOp, AllocOp>();
	addOperations<FreeOp, DimOp, SubscriptionOp>();
	addOperations<LoadOp, StoreOp>();
	addOperations<ArrayCopyOp>();

	// Math operations
	addOperations<NegateOp, AddOp, SubOp, MulOp, DivOp, PowOp>();

	// Logic operations
	addOperations<NotOp, AndOp, OrOp>();
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

	// Builtin operations
	addOperations<NDimsOp, SizeOp>();
}

mlir::StringRef ModelicaDialect::getDialectNamespace()
{
	return "modelica";
}

void ModelicaDialect::printType(mlir::Type type, mlir::DialectAsmPrinter& printer) const {
	return printModelicaType(type, printer);
}

void ModelicaDialect::printAttribute(mlir::Attribute attribute, mlir::DialectAsmPrinter& printer) const {
	return printModelicaAttribute(attribute, printer);
}
