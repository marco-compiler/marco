#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <marco/mlirlowerer/dialects/ida/IdaDialect.h>

using namespace marco::codegen::ida;

IdaDialect::IdaDialect(mlir::MLIRContext* context)
		: Dialect("ida", context, mlir::TypeID::get<IdaDialect>())
{
	addTypes<BooleanType, IntegerType, RealType, OpaquePointerType>();
	addAttributes<BooleanAttribute, IntegerAttribute, RealAttribute>();

	// Allocation, initialization, usage and deletion.
	addOperations<
			ConstantValueOp,
			AllocUserDataOp,
			InitOp,
			StepOp,
			FreeUserDataOp,
			AddTimeOp,
			AddToleranceOp>();

	// Equation setters.
	addOperations<
			AddRowLengthOp,
			AddColumnIndexOp,
			AddEqDimensionOp,
			AddResidualOp,
			AddJacobianOp>();

	// Variable setters.
	addOperations<
			AddVarOffsetOp,
			AddVarDimensionOp,
			AddVarAccessOp,
			SetInitialValueOp,
			SetInitialArrayOp>();

	// Getters.
	addOperations<GetTimeOp, GetVariableOp, GetDerivativeOp>();

	// Lambda constructions.
	addOperations<
			LambdaConstantOp,
			LambdaTimeOp,
			LambdaInductionOp,
			LambdaVariableOp,
			LambdaDerivativeOp>();

	addOperations<
			LambdaAddOp,
			LambdaSubOp,
			LambdaMulOp,
			LambdaDivOp,
			LambdaPowOp,
			LambdaNegateOp,
			LambdaAbsOp,
			LambdaSignOp,
			LambdaSqrtOp,
			LambdaExpOp,
			LambdaLogOp,
			LambdaLog10Op>();

	addOperations<
			LambdaSinOp,
			LambdaCosOp,
			LambdaTanOp,
			LambdaAsinOp,
			LambdaAcosOp,
			LambdaAtanOp,
			LambdaAtan2Op,
			LambdaSinhOp,
			LambdaCoshOp,
			LambdaTanhOp>();
	
	addOperations<LambdaCallOp, LambdaAddressOfOp>();
}

mlir::StringRef IdaDialect::getDialectNamespace()
{
	return "ida";
}

mlir::Type IdaDialect::parseType(mlir::DialectAsmParser& parser) const
{
	return parseIdaType(parser);
}

void IdaDialect::printType(mlir::Type type, mlir::DialectAsmPrinter& printer) const
{
	return printIdaType(type, printer);
}

mlir::Attribute IdaDialect::parseAttribute(mlir::DialectAsmParser& parser, mlir::Type type) const
{
	return parseIdaAttribute(parser, type);
}

void IdaDialect::printAttribute(mlir::Attribute attribute, mlir::DialectAsmPrinter& printer) const
{
	return printIdaAttribute(attribute, printer);
}

mlir::Operation* IdaDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value, mlir::Type type, mlir::Location loc)
{
	return builder.create<ConstantValueOp>(loc, value);
}
