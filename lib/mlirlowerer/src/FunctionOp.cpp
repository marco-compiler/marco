#include <modelica/mlirlowerer/FunctionOp.hpp>

using namespace llvm;
using namespace modelica;

StringRef FunctionOp::getOperationName()
{
	return "function";
}