#include "modelica/frontend/Expression.hpp"

#include "modelica/utils/IRange.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

bool Expression::Operation::operator==(const Operation& other) const
{
	if (kind != other.kind)
		return false;
	if (arguments.size() != other.arguments.size())
		return false;

	return arguments == other.arguments;
}
