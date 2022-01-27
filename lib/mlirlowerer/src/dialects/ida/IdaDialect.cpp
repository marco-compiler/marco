#include <marco/mlirlowerer/dialects/ida/IdaDialect.h>
#include <marco/mlirlowerer/dialects/modelica/Ops.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

using namespace marco::codegen::ida;

IdaDialect::IdaDialect(mlir::MLIRContext* context)
		: Dialect("ida", context, mlir::TypeID::get<IdaDialect>())
{
	addTypes<OpaquePointerType, IntegerPointerType, RealPointerType>();

	// Allocation, initialization, usage and deletion.
	addOperations<
			AllocDataOp,
			InitOp,
			StepOp,
			FreeDataOp,
			AddTimeOp,
			AddToleranceOp>();

	// Equation setters.
	addOperations<
			AddEquationOp,
			AddResidualOp,
			AddJacobianOp>();

	// Variable setters.
	addOperations<
			AddVariableOp,
			AddVarAccessOp>();

	// Getters.
	addOperations<
			GetVariableAllocOp,
			GetTimeOp>();

	// Residual and Jacobian construction helpers.
	addOperations<
			ResidualFunctionOp,
			JacobianFunctionOp,
			FunctionTerminatorOp,
			FuncAddressOfOp,
			LoadPointerOp>();

	// Statistics.
	addOperations<PrintStatisticsOp>();
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
	return builder.create<modelica::ConstantOp>(loc, value);
}
