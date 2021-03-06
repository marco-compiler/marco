#include <mlir/IR/Builders.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/OpImplementation.h>
#include <modelica/mlirlowerer/ops/MathOps.h>

using namespace modelica;
using namespace std;

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
	printer << "modelica.div " << getOperands() << " : " << getOperation()->getResultTypes()[0];
}
