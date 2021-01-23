#include <mlir/IR/Builders.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/StandardTypes.h>
#include <modelica/mlirlowerer/Ops.hpp>

using namespace modelica;
using namespace std;

llvm::StringRef AssignmentOp::getOperationName()
{
	return "modelica.assignment";
}

void AssignmentOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value source, mlir::ValueRange sourceIndexes, mlir::Value destination, mlir::ValueRange destinationIndexes)
{
	state.addOperands({ source, destination });

	state.addAttribute("sourceIndexes", builder.getIndexAttr(sourceIndexes.size()));
	state.addAttribute("destinationIndexes", builder.getIndexAttr(destinationIndexes.size()));

	state.addOperands(sourceIndexes);
	state.addOperands(destinationIndexes);
}

void AssignmentOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "assign " << source() << "[" << sourceIndexes() << "], " << destination() << "[" << destinationIndexes() << "]";
}

mlir::Value AssignmentOp::source()
{
	return getOperand(0);
}

mlir::Value AssignmentOp::destination()
{
	return getOperand(1);
}

mlir::Operation::operand_range AssignmentOp::sourceIndexes()
{
	size_t sourceIndexes = getAttr("sourceIndexes").cast<mlir::IntegerAttr>().getInt();
	return { operand_begin() + 2, operand_begin() + 2 + sourceIndexes };
}

mlir::Operation::operand_range AssignmentOp::destinationIndexes()
{
	size_t sourceIndexes = getAttr("sourceIndexes").cast<mlir::IntegerAttr>().getInt();
	return { operand_begin() + 2 + sourceIndexes, operand_end() };
}

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

void ForOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value breakCondition, mlir::Value returnCondition)
{
	build(builder, state, breakCondition, returnCondition, {});
}

void ForOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value breakCondition, mlir::Value returnCondition, mlir::ValueRange args)
{
	state.addOperands(breakCondition);
	state.addOperands(returnCondition);
	state.addOperands(args);

	auto insertionPoint = builder.saveInsertionPoint();

	// Condition block
	builder.createBlock(state.addRegion(), {}, args.getTypes());

	// Step block
	builder.createBlock(state.addRegion(), {}, args.getTypes());

	// Body block
	builder.createBlock(state.addRegion(), {}, args.getTypes());

	builder.restoreInsertionPoint(insertionPoint);
}

void ForOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "for (break on " << breakCondition() << ", return on " << returnCondition() << ")";

	auto operands = getOperands();

	if (operands.size() > 2)
	{
		auto operandsBegin = operands.begin();
		operandsBegin += 2;
		printer << " ";
		printer.printOperands(operandsBegin, operands.end());
	}

	printer << " condition";
	printer.printRegion(condition(), true);
	printer << " body";
	printer.printRegion(body(), true);
	printer << " step";
	printer.printRegion(step(), true);
}

mlir::Region& ForOp::condition()
{
	return getOperation()->getRegion(0);
}

mlir::Region& ForOp::step()
{
	return getOperation()->getRegion(1);
}

mlir::Region& ForOp::body()
{
	return getOperation()->getRegion(2);
}

mlir::Value ForOp::breakCondition()
{
	return getOperand(0);
}

mlir::Value ForOp::returnCondition()
{
	return getOperand(1);
}

mlir::Operation::operand_range ForOp::args()
{
	return { std::next(getOperation()->operand_begin(), 2), getOperation()->operand_end()};
}

llvm::StringRef WhileOp::getOperationName()
{
	return "modelica.while";
}

void WhileOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value breakCondition, mlir::Value returnCondition)
{
	state.addOperands(breakCondition);
	state.addOperands(returnCondition);

	auto insertionPoint = builder.saveInsertionPoint();

	// Condition block
	builder.createBlock(state.addRegion());

	// Body block
	builder.createBlock(state.addRegion());

	builder.restoreInsertionPoint(insertionPoint);
}

void WhileOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "while (break on " << breakCondition() << ", return on " << returnCondition() << ")";
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

mlir::Value WhileOp::breakCondition()
{
	return getOperand(0);
}

mlir::Value WhileOp::returnCondition()
{
	return getOperand(1);
}

llvm::StringRef ConditionOp::getOperationName()
{
	return "modelica.condition";
}

void ConditionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value condition)
{
	build(builder, state, condition, {});
}

void ConditionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value condition, mlir::ValueRange args)
{
	state.addOperands(condition);
	state.addOperands(args);
}

void ConditionOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "condition (" << condition() << ")";

	for (size_t i = 1; i < getNumOperands(); i++)
		printer << " " << getOperand(i);
}

mlir::Value ConditionOp::condition()
{
	return getOperand(0);
}

mlir::Operation::operand_range ConditionOp::args()
{
	return { std::next(getOperation()->operand_begin()), getOperation()->operand_end()};
}

llvm::StringRef YieldOp::getOperationName()
{
	return "modelica.yield";
}

void YieldOp::build(mlir::OpBuilder& builder, mlir::OperationState& state)
{

}

void YieldOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange operands)
{
	state.addOperands(operands);
}

void YieldOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.yield " << getOperands();
}
