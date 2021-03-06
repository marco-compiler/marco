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