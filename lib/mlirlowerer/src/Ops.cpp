#include <mlir/IR/Builders.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/StandardTypes.h>
#include <modelica/mlirlowerer/Ops.hpp>

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
	printer << "neg " << getOperand();
}

llvm::StringRef AddOp::getOperationName()
{
	return "modelica.add";
}

void AddOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange operands)
{
	assert(operands.size() >= 2);
	state.addOperands(operands);

	// All operands have same type and shape
	state.addTypes(operands[0].getType());
}

void AddOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "add " << getOperands();
}

llvm::StringRef SubOp::getOperationName()
{
	return "modelica.sub";
}

void SubOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange operands)
{
	assert(operands.size() >= 2);
	state.addOperands(operands);

	// All operands have same type and shape
	state.addTypes(operands[0].getType());
}

void SubOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "sub " << getOperands();
}

llvm::StringRef MulOp::getOperationName()
{
	return "modelica.mul";
}

void MulOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange operands)
{
	assert(operands.size() >= 2);
	state.addOperands(operands);

	// All operands have same type and shape
	state.addTypes(operands[0].getType());
}

void MulOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "mul " << getOperands();
}

llvm::StringRef DivOp::getOperationName()
{
	return "modelica.div";
}

void DivOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange operands)
{
	assert(operands.size() >= 2);
	state.addOperands(operands);

	// All operands have same type and shape
	state.addTypes(operands[0].getType());
}

void DivOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "div " << getOperands();
}

llvm::StringRef EqOp::getOperationName()
{
	return "modelica.eq";
}

void EqOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addOperands({ lhs, rhs });
	state.addTypes(builder.getI1Type());
}

void EqOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "eq " << getOperands();
}

llvm::StringRef NotEqOp::getOperationName()
{
	return "modelica.neq";
}

void NotEqOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addOperands({ lhs, rhs });
	state.addTypes(builder.getI1Type());
}

void NotEqOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "neq " << getOperands();
}

llvm::StringRef GtOp::getOperationName()
{
	return "modelica.gt";
}

void GtOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addOperands({ lhs, rhs });
	state.addTypes(builder.getI1Type());
}

void GtOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "gt " << getOperands();
}

llvm::StringRef GteOp::getOperationName()
{
	return "modelica.gte";
}

void GteOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addOperands({ lhs, rhs });
	state.addTypes(builder.getI1Type());
}

void GteOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "gte " << getOperands();
}

llvm::StringRef LtOp::getOperationName()
{
	return "modelica.lt";
}

void LtOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addOperands({ lhs, rhs });
	state.addTypes(builder.getI1Type());
}

void LtOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "lt " << getOperands();
}

llvm::StringRef LteOp::getOperationName()
{
	return "modelica.lte";
}

void LteOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addOperands({ lhs, rhs });
	state.addTypes(builder.getI1Type());
}

void LteOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "lte " << getOperands();
}

llvm::StringRef IfOp::getOperationName()
{
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

void IfOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "if " << getOperand();
	printer.printRegion(thenRegion());

	if (!elseRegion().empty())
	{
		printer << " else";
		printer.printRegion(elseRegion());
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

llvm::StringRef ForOp::getOperationName()
{
	return "modelica.for";
}

void ForOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lowerBound, mlir::Value upperBound, mlir::Value step)
{
	state.addOperands({ lowerBound, upperBound, step });

	auto insertionPoint = builder.saveInsertionPoint();

	// Body block
	auto* body = state.addRegion();
	builder.createBlock(body, body->begin(), builder.getIndexType());

	// Exit block (for break operation)
	builder.createBlock(state.addRegion());
	builder.create<YieldOp>(state.location);

	builder.restoreInsertionPoint(insertionPoint);
}

void ForOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "for " << lowerBound() << " to " << upperBound() << " step " << step();
	printer.printRegion(body());
}

mlir::Value ForOp::lowerBound()
{
	return getOperation()->getOperand(0);
}

mlir::Value ForOp::upperBound()
{
	return getOperation()->getOperand(1);
}

mlir::Value ForOp::step()
{
	return getOperation()->getOperand(2);
}

mlir::Region& ForOp::body()
{
	return getOperation()->getRegion(0);
}

mlir::Region& ForOp::exit()
{
	return getOperation()->getRegion(1);
}

llvm::StringRef WhileOp::getOperationName()
{
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

void WhileOp::print(mlir::OpAsmPrinter& printer)
{
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

llvm::StringRef ConditionOp::getOperationName()
{
	return "modelica.condition";
}

void ConditionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value condition)
{
	state.addOperands(condition);
}

void ConditionOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "condition " << condition();
}

mlir::Value ConditionOp::condition()
{
	return getOperand();
}

llvm::StringRef YieldOp::getOperationName()
{
	return "modelica.yield";
}

void YieldOp::build(mlir::OpBuilder& builder, mlir::OperationState& state)
{

}

void YieldOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "yield";
}

llvm::StringRef BreakOp::getOperationName()
{
	return "modelica.break";
}

void BreakOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Block* successor)
{
	state.addSuccessors(successor);
}

void BreakOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "break";
}
