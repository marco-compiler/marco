#include <mlir/IR/Builders.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/StandardTypes.h>
#include <modelica/mlirlowerer/Ops.hpp>

using namespace modelica;
using namespace std;

llvm::StringRef CastOp::getOperationName()
{
	return "modelica.cast";
}

void CastOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value value, mlir::Type destinationType)
{
	state.addOperands(value);
	state.addTypes(destinationType);
}

void CastOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "cast " << value() << " : " << getOperation()->getResultTypes()[0];
}

mlir::Value CastOp::value()
{
	return getOperand();
}

llvm::StringRef CastCommonOp::getOperationName()
{
	return "modelica.cast_common";
}

static mlir::Type getMoreGenericType(mlir::Type x, mlir::Type y)
{
	mlir::Type xBase = x;
	mlir::Type yBase = y;

	while (xBase.isa<mlir::ShapedType>())
		xBase = xBase.cast<mlir::ShapedType>().getElementType();

	while (yBase.isa<mlir::ShapedType>())
		yBase = yBase.cast<mlir::ShapedType>().getElementType();

	if (xBase.isa<mlir::FloatType>())
		return x;

	if (yBase.isa<mlir::FloatType>())
		return y;

	if (xBase.isa<mlir::IntegerType>())
		return x;

	if (yBase.isa<mlir::IntegerType>())
		return y;

	return x;
}

void CastCommonOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange values)
{
	state.addOperands(values);

	mlir::Type resultType = nullptr;
	mlir::Type resultBaseType = nullptr;

	for (const auto& value : values)
	{
		mlir::Type type = value.getType();
		mlir::Type baseType = type;

		if (resultType == nullptr)
		{
			resultType = type;
			resultBaseType = type;

			while (resultBaseType.isa<mlir::ShapedType>())
				resultBaseType = resultBaseType.cast<mlir::ShapedType>().getElementType();

			continue;
		}

		if (type.isa<mlir::ShapedType>())
		{
			mlir::Type result = resultType;

			//assert(result.isa<mlir::ShapedType>() && "Values have different shape");
			//assert(result.cast<mlir::ShapedType>().getShape() == type.cast<mlir::ShapedType>().getShape() && "Values have different shape");

			while (baseType.isa<mlir::ShapedType>())
				baseType = baseType.cast<mlir::ShapedType>().getElementType();
		}

		if (baseType.isIndex())
			continue;

		if (resultBaseType.isIndex())
		{
			resultType = type;
			resultBaseType = baseType;
		}
		else if (resultBaseType.isa<mlir::IntegerType>())
		{
			if (baseType.isa<mlir::FloatType>() || baseType.getIntOrFloatBitWidth() > resultBaseType.getIntOrFloatBitWidth())
			{
				resultType = type;
				resultBaseType = baseType;
			}
		}
		else if (resultBaseType.isa<mlir::FloatType>())
		{
			if (baseType.isa<mlir::FloatType>() && baseType.getIntOrFloatBitWidth() > resultBaseType.getIntOrFloatBitWidth())
			{
				resultType = type;
				resultBaseType = baseType;
			}
		}
	}

	llvm::SmallVector<mlir::Type, 3> types;

	for (const auto& value : values)
	{
		mlir::Type type = value.getType();

		if (type.isa<mlir::ShapedType>())
		{
			auto shape = type.cast<mlir::ShapedType>().getShape();
			types.emplace_back(mlir::VectorType::get(shape, resultBaseType));
		}
		else
			types.emplace_back(resultBaseType);
	}

	state.addTypes(types);
}

void CastCommonOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "cast_common ";
	printer.printOperands(values());
	printer << " : " << getOperation()->getResultTypes();
}

mlir::ValueRange CastCommonOp::values()
{
	return getOperands();
}

mlir::Type CastCommonOp::type()
{
	return getResultTypes()[0];
}

llvm::StringRef AssignmentOp::getOperationName()
{
	return "modelica.assignment";
}

void AssignmentOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value source, mlir::Value destination)
{
	state.addOperands({ source, destination });
}

void AssignmentOp::print(mlir::OpAsmPrinter& printer)
{
	mlir::Value source = this->source();
	mlir::Value destination = this->destination();
	printer << "assign " << source << ", " << destination << " : " << source.getType() << ", " << destination.getType();
}

mlir::Value AssignmentOp::source()
{
	return getOperand(0);
}

mlir::Value AssignmentOp::destination()
{
	return getOperand(1);
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
	printer << "neg " << getOperand() << " : " << getOperation()->getResultTypes();
}

llvm::StringRef AddOp::getOperationName()
{
	return "modelica.add";
}

void AddOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::ValueRange operands)
{
	if (resultType.isa<mlir::ShapedType>())
	{
		auto shapedType = resultType.cast<mlir::ShapedType>();
		resultType = mlir::VectorType::get(shapedType.getShape(), shapedType.getElementType());
	}

	state.addTypes(resultType);

	assert(operands.size() >= 2);
	state.addOperands(operands);
}

void AddOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "add " << getOperands() << " : " << getOperation()->getResultTypes()[0];
}

mlir::ValueRange AddOp::values()
{
	return getOperands();
}

llvm::StringRef SubOp::getOperationName()
{
	return "modelica.sub";
}

void SubOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::ValueRange operands)
{
	if (resultType.isa<mlir::ShapedType>())
	{
		auto shapedType = resultType.cast<mlir::ShapedType>();
		resultType = mlir::VectorType::get(shapedType.getShape(), shapedType.getElementType());
	}

	state.addTypes(resultType);

	assert(operands.size() >= 2);
	state.addOperands(operands);
}

void SubOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "sub " << getOperands() << " : " << getOperation()->getResultTypes()[0];
}

llvm::StringRef MulOp::getOperationName()
{
	return "modelica.mul";
}

void MulOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::ValueRange operands)
{
	if (resultType.isa<mlir::ShapedType>())
	{
		auto shapedType = resultType.cast<mlir::ShapedType>();
		resultType = mlir::VectorType::get(shapedType.getShape(), shapedType.getElementType());
	}

	state.addTypes(resultType);

	assert(operands.size() >= 2);
	state.addOperands(operands);
}

void MulOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "mul " << getOperands() << " : (" << getOperandTypes() << ") -> (" << getOperation()->getResultTypes()[0] << ")";
}

llvm::StringRef CrossProductOp::getOperationName()
{
	return "modelica.cross_product";
}

void CrossProductOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	auto xShapedType = lhs.getType().cast<mlir::ShapedType>();
	auto yShapedType = rhs.getType().cast<mlir::ShapedType>();

	mlir::Type baseType = xShapedType;

	while (baseType.isa<mlir::ShapedType>())
		baseType = baseType.cast<mlir::ShapedType>().getElementType();

	// TODO: add verifier for equality of base types



	state.addTypes(baseType);

	state.addOperands(lhs);
	state.addOperands(rhs);
}

void CrossProductOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "cross_product " << getOperands() << " : (" << getOperandTypes() << ") -> (" << getOperation()->getResultTypes()[0] << ")";
}

mlir::Value CrossProductOp::lhs()
{
	return getOperand(0);
}

mlir::Value CrossProductOp::rhs()
{
	return getOperand(1);
}

llvm::StringRef DivOp::getOperationName()
{
	return "modelica.div";
}

void DivOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::ValueRange operands)
{
	if (resultType.isa<mlir::ShapedType>())
	{
		auto shapedType = resultType.cast<mlir::ShapedType>();
		resultType = mlir::VectorType::get(shapedType.getShape(), shapedType.getElementType());
	}

	state.addTypes(resultType);

	assert(operands.size() >= 2);
	state.addOperands(operands);
}

void DivOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "div " << getOperands() << " : " << getOperation()->getResultTypes()[0];
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
