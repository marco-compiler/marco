#include <modelica/mlirlowerer/AssignmentOp.hpp>

using namespace llvm;
using namespace modelica;

StringRef AssignmentOp::getOperationName()
{
	return "assignment";
}
