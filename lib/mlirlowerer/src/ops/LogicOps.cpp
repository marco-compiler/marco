#include <mlir/IR/Builders.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/StandardTypes.h>
#include <modelica/mlirlowerer/ops/LogicOps.h>

using namespace modelica;
using namespace std;

llvm::StringRef NegateOp::getOperationName()
{
	return "modelica.negate";
}

void NegateOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value operand)
{
	state.addOperands(operand);
	state.addTypes(operand.getType());
}

void NegateOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "neg " << getOperand() << " : " << getOperation()->getResultTypes();
}

llvm::StringRef EqOp::getOperationName()
{
	return "modelica.eq";
}

void EqOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(builder.getI1Type());
	state.addOperands({ lhs, rhs });
}

void EqOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::LogicalResult EqOp::verify()
{
	for (auto operand : getOperands())
		if (operand.getType().isa<mlir::ShapedType>())
			return emitOpError("Comparison operation are only defined for scalar operands of simple types");

	return mlir::success();
}

void EqOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "eq " << getOperands() << " : " << getOperation()->getResultTypes()[0];
}

llvm::StringRef NotEqOp::getOperationName()
{
	return "modelica.neq";
}

void NotEqOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(builder.getI1Type());
	state.addOperands({ lhs, rhs });
}

void NotEqOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::LogicalResult NotEqOp::verify()
{
	for (auto operand : getOperands())
		if (operand.getType().isa<mlir::ShapedType>())
			return emitOpError("Comparison operation are only defined for scalar operands of simple types");

	return mlir::success();
}

void NotEqOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "neq " << getOperands() << " : " << getOperation()->getResultTypes()[0];
}

llvm::StringRef GtOp::getOperationName()
{
	return "modelica.gt";
}

void GtOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(builder.getI1Type());
	state.addOperands({ lhs, rhs });
}

void GtOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::LogicalResult GtOp::verify()
{
	for (auto operand : getOperands())
		if (operand.getType().isa<mlir::ShapedType>())
			return emitOpError("Comparison operation are only defined for scalar operands of simple types");

	return mlir::success();
}

void GtOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "gt " << getOperands() << " : " << getOperation()->getResultTypes()[0];
}

llvm::StringRef GteOp::getOperationName()
{
	return "modelica.gte";
}

void GteOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(builder.getI1Type());
	state.addOperands({ lhs, rhs });
}

void GteOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::LogicalResult GteOp::verify()
{
	for (auto operand : getOperands())
		if (operand.getType().isa<mlir::ShapedType>())
			return emitOpError("Comparison operation are only defined for scalar operands of simple types");

	return mlir::success();
}

void GteOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "gte " << getOperands() << " : " << getOperation()->getResultTypes()[0];
}

llvm::StringRef LtOp::getOperationName()
{
	return "modelica.lt";
}

void LtOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(builder.getI1Type());
	state.addOperands({ lhs, rhs });
}

void LtOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::LogicalResult LtOp::verify()
{
	for (auto operand : getOperands())
		if (operand.getType().isa<mlir::ShapedType>())
			return emitOpError("Comparison operation are only defined for scalar operands of simple types");

	return mlir::success();
}

void LtOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "lt " << getOperands() << " : " << getOperation()->getResultTypes()[0];
}

llvm::StringRef LteOp::getOperationName()
{
	return "modelica.lte";
}

void LteOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(builder.getI1Type());
	state.addOperands({ lhs, rhs });
}

void LteOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::LogicalResult LteOp::verify()
{
	for (auto operand : getOperands())
		if (operand.getType().isa<mlir::ShapedType>())
			return emitOpError("Comparison operation are only defined for scalar operands of simple types");

	return mlir::success();
}

void LteOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "lte " << getOperands() << " : " << getOperation()->getResultTypes()[0];
}
