#include <mlir/IR/Builders.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/OpImplementation.h>
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

llvm::StringRef WhileOp::getOperationName() {
	return "modelica.while";
}

void WhileOp::build(mlir::OpBuilder& builder, mlir::OperationState& state)
{
	auto insertionPoint = builder.saveInsertionPoint();
	builder.createBlock(state.addRegion());	// Condition
	builder.createBlock(state.addRegion());	// Body
	builder.createBlock(state.addRegion());	// Continuation

	builder.create<YieldOp>(state.location);

	builder.restoreInsertionPoint(insertionPoint);
}

mlir::Region& WhileOp::condition()
{
	return getOperation()->getRegion(0);
}

mlir::Region& WhileOp::body()
{
	return getOperation()->getRegion(1);
}

mlir::Region& WhileOp::exit()
{
	return getOperation()->getRegion(2);
}

void WhileOp::print(mlir::OpAsmPrinter& printer) {
	printer << "modelica.while";
	printer.printRegion(condition(), false);
	printer << " do";
	printer.printRegion(body(), false);
	printer << " continuation";
	printer.printRegion(getOperation()->getRegion(2));
}

llvm::StringRef YieldOp::getOperationName() {
	return "modelica.yield";
}

void YieldOp::build(mlir::OpBuilder& builder, mlir::OperationState& staten)
{

}

void YieldOp::print(mlir::OpAsmPrinter& printer) {
	printer << "modelica.yield";
}

llvm::StringRef BreakOp::getOperationName() {
	return "modelica.break";
}

void BreakOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Block* successor)
{
	state.addSuccessors(successor);
}
