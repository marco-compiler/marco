#include <modelica/mlirlowerer/MathOps.hpp>

using namespace modelica;
using namespace std;

llvm::StringRef AddOp::getOperationName() {
	return "modelica.add";
}

void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type resultType, mlir::ValueRange operands)
{
	state.addOperands(operands);
	state.addTypes(resultType);
}

llvm::StringRef SubOp::getOperationName() {
	return "modelica.sub";
}

void SubOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type resultType, mlir::ValueRange operands)
{
	state.addOperands(operands);
	state.addTypes(resultType);
}

llvm::StringRef MulOp::getOperationName() {
	return "modelica.mul";
}

void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type resultType, mlir::ValueRange operands)
{
	state.addOperands(operands);
	state.addTypes(resultType);
}

llvm::StringRef DivOp::getOperationName() {
	return "modelica.div";
}

void DivOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type resultType, mlir::ValueRange operands)
{
	state.addOperands(operands);
	state.addTypes(resultType);
}
