#include <mlir/IR/StandardTypes.h>
#include <modelica/mlirlowerer/ConstantOpOld.hpp>

using namespace llvm;
using namespace mlir;
using namespace modelica;
using namespace std;

llvm::StringRef ConstantOpOld::getOperationName()
{
	return "modelica.constant";
}

mlir::Attribute ConstantOpOld::getValue()
{
	return getAttr("value").cast<Attribute>();
}

mlir::LogicalResult ConstantOpOld::verify()
{
	return success();
}

void ConstantOpOld::build(OpBuilder &builder, OperationState &state, mlir::Type type, Attribute value)
{
	state.addAttribute("value", value);
	state.addTypes(type);
}