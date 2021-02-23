#include <mlir/IR/Builders.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/OpImplementation.h>
#include <modelica/mlirlowerer/ops/BasicOps.h>

using namespace modelica;
using namespace std;

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
	printer << "modelica.assign " << source << " to " << destination << " : " << source.getType() << ", " << destination.getType();
}

mlir::Value AssignmentOp::source()
{
	return getOperand(0);
}

mlir::Value AssignmentOp::destination()
{
	return getOperand(1);
}
