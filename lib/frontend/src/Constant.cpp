#include "modelica/frontend/Constant.hpp"

#include "modelica/frontend/Type.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;

void Constant::dump(llvm::raw_ostream& OS, size_t indentLevel) const
{
	OS.indent(indentLevel);
	if (isA<BuiltinType::Integer>())
		OS << get<BuiltinType::Integer>();
	else if (isA<BuiltinType::Float>())
		OS << get<BuiltinType::Float>();
	else if (isA<BuiltinType::Boolean>())
		OS << (get<BuiltinType::Boolean>() ? "true" : "false");
	else if (isA<BuiltinType::String>())
		OS << get<BuiltinType::String>();
	else
		assert(false && "unrechable");
}
