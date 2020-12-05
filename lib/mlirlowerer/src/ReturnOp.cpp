#include <modelica/mlirlowerer/ReturnOp.hpp>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/StandardTypes.h>

using namespace llvm;
using namespace mlir;
using namespace modelica;
using namespace std;

llvm::StringRef ReturnOp::getOperationName()
{
	return "modelica.return";
}

mlir::LogicalResult ReturnOp::verify()
{
	return success();
}

void ReturnOp::build(OpBuilder &builder, OperationState &state, ArrayRef<mlir::Type> types, mlir::ValueRange operands)
{
	state.addOperands(operands);
	state.addTypes(types);
}
