#include <mlir/IR/Builders.h>
#include <mlir/IR/StandardTypes.h>
#include <modelica/mlirlowerer/Ops.hpp>

using namespace modelica;
using namespace std;

llvm::StringRef NegateOp::getOperationName() {
	return "modelica.negate";
}

void NegateOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value operand)
{
	state.addOperands(operand);
	state.addTypes(operand.getType());
}

llvm::StringRef AddOp::getOperationName() {
	return "modelica.add";
}

void AddOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange operands)
{
	assert(operands.size() >= 2);
	state.addOperands(operands);

	// All operands have same type and shape
	state.addTypes(operands[0].getType());
}

llvm::StringRef SubOp::getOperationName() {
	return "modelica.sub";
}

void SubOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange operands)
{
	assert(operands.size() >= 2);
	state.addOperands(operands);

	// All operands have same type and shape
	state.addTypes(operands[0].getType());
}

llvm::StringRef MulOp::getOperationName() {
	return "modelica.mul";
}

void MulOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange operands)
{
	assert(operands.size() >= 2);
	state.addOperands(operands);

	// All operands have same type and shape
	state.addTypes(operands[0].getType());
}

llvm::StringRef DivOp::getOperationName() {
	return "modelica.div";
}

void DivOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange operands)
{
	assert(operands.size() >= 2);
	state.addOperands(operands);

	// All operands have same type and shape
	state.addTypes(operands[0].getType());
}

llvm::StringRef EqOp::getOperationName() {
	return "modelica.eq";
}

void EqOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addOperands({ lhs, rhs });
	state.addTypes(builder.getI1Type());
}

llvm::StringRef NotEqOp::getOperationName() {
	return "modelica.neq";
}

void NotEqOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addOperands({ lhs, rhs });
	state.addTypes(builder.getI1Type());
}

llvm::StringRef GtOp::getOperationName() {
	return "modelica.gt";
}

void GtOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addOperands({ lhs, rhs });
	state.addTypes(builder.getI1Type());
}

llvm::StringRef GteOp::getOperationName() {
	return "modelica.gte";
}

void GteOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addOperands({ lhs, rhs });
	state.addTypes(builder.getI1Type());
}

llvm::StringRef LtOp::getOperationName() {
	return "modelica.lt";
}

void LtOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addOperands({ lhs, rhs });
	state.addTypes(builder.getI1Type());
}

llvm::StringRef LteOp::getOperationName() {
	return "modelica.lte";
}

void LteOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addOperands({ lhs, rhs });
	state.addTypes(builder.getI1Type());
}
