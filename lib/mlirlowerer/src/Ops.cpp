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

void NegateOp::print(mlir::OpAsmPrinter& printer) {
	printer << "neg ";
	printer.printOperands(getOperation()->getOperands());
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

void AddOp::print(mlir::OpAsmPrinter& printer) {
	printer << "add ";
	printer.printOperands(getOperation()->getOperands());
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

void SubOp::print(mlir::OpAsmPrinter& printer) {
	printer << "sub ";
	printer.printOperands(getOperation()->getOperands());
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

void MulOp::print(mlir::OpAsmPrinter& printer) {
	printer << "mul ";
	printer.printOperands(getOperation()->getOperands());
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

void DivOp::print(mlir::OpAsmPrinter& printer) {
	printer << "div ";
	printer.printOperands(getOperation()->getOperands());
}

llvm::StringRef EqOp::getOperationName() {
	return "modelica.eq";
}

void EqOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addOperands({ lhs, rhs });
	state.addTypes(builder.getI1Type());
}

void EqOp::print(mlir::OpAsmPrinter& printer) {
	printer << "eq ";
	printer.printOperands(getOperation()->getOperands());
}

llvm::StringRef NotEqOp::getOperationName() {
	return "modelica.neq";
}

void NotEqOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addOperands({ lhs, rhs });
	state.addTypes(builder.getI1Type());
}

void NotEqOp::print(mlir::OpAsmPrinter& printer) {
	printer << "neq ";
	printer.printOperands(getOperation()->getOperands());
}

llvm::StringRef GtOp::getOperationName() {
	return "modelica.gt";
}

void GtOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addOperands({ lhs, rhs });
	state.addTypes(builder.getI1Type());
}

void GtOp::print(mlir::OpAsmPrinter& printer) {
	printer << "gt ";
	printer.printOperands(getOperation()->getOperands());
}

llvm::StringRef GteOp::getOperationName() {
	return "modelica.gte";
}

void GteOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addOperands({ lhs, rhs });
	state.addTypes(builder.getI1Type());
}

void GteOp::print(mlir::OpAsmPrinter& printer) {
	printer << "gte ";
	printer.printOperands(getOperation()->getOperands());
}

llvm::StringRef LtOp::getOperationName() {
	return "modelica.lt";
}

void LtOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addOperands({ lhs, rhs });
	state.addTypes(builder.getI1Type());
}

void LtOp::print(mlir::OpAsmPrinter& printer) {
	printer << "lt ";
	printer.printOperands(getOperation()->getOperands());
}

llvm::StringRef LteOp::getOperationName() {
	return "modelica.lte";
}

void LteOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addOperands({ lhs, rhs });
	state.addTypes(builder.getI1Type());
}

void LteOp::print(mlir::OpAsmPrinter& printer) {
	printer << "lte ";
	printer.printOperands(getOperation()->getOperands());
}

llvm::StringRef IfOp::getOperationName() {
	return "modelica.if";
}

void IfOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value condition, bool withElseRegion)
{
	state.addOperands(condition);
	auto insertionPoint = builder.saveInsertionPoint();

	// "Then" region
	auto* thenRegion = state.addRegion();
	builder.createBlock(thenRegion);

	// "Else" region
	auto* elseRegion = state.addRegion();

	if (withElseRegion)
		builder.createBlock(elseRegion);

	builder.restoreInsertionPoint(insertionPoint);
}

void IfOp::print(mlir::OpAsmPrinter& printer) {
	printer << "if ";
	printer.printOperands(getOperation()->getOperands());
	printer.printRegion(getRegion(0));

	if (!getRegion(1).empty())
	{
		printer << " else";
		printer.printRegion(getRegion(1));
	}
}

mlir::Value IfOp::condition()
{
	return getOperand();
}

mlir::Region& IfOp::thenRegion()
{
	return getRegion(0);
}

mlir::Region& IfOp::elseRegion()
{
	return getRegion(1);
}

llvm::StringRef WhileOp::getOperationName() {
	return "modelica.while";
}

void WhileOp::build(mlir::OpBuilder& builder, mlir::OperationState& state)
{
	auto insertionPoint = builder.saveInsertionPoint();

	// Condition block
	builder.createBlock(state.addRegion());

	// Body block
	builder.createBlock(state.addRegion());

	// Exit block (for break operation)
	builder.createBlock(state.addRegion());
	builder.create<YieldOp>(state.location);

	builder.restoreInsertionPoint(insertionPoint);
}

void WhileOp::print(mlir::OpAsmPrinter& printer) {
	printer << "while";
	printer.printRegion(condition(), false);
	printer << " do";
	printer.printRegion(body(), false);
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

llvm::StringRef YieldOp::getOperationName() {
	return "modelica.yield";
}

void YieldOp::build(mlir::OpBuilder& builder, mlir::OperationState& staten)
{

}

void YieldOp::print(mlir::OpAsmPrinter& printer) {
	printer << "yield";
}

llvm::StringRef BreakOp::getOperationName() {
	return "modelica.break";
}

void BreakOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Block* successor)
{
	state.addSuccessors(successor);
}

void BreakOp::print(mlir::OpAsmPrinter& printer) {
	printer << "break";
}
