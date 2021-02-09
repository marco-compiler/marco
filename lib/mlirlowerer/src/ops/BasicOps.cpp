#include <mlir/IR/Builders.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/StandardTypes.h>
#include <modelica/mlirlowerer/ops/BasicOps.h>

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
