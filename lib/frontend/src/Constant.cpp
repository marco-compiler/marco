#include "modelica/frontend/Constant.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;

void Constant::dump(llvm::raw_ostream& OS, size_t indentLevel) const
{
	OS.indent(indentLevel);
	if (isA<int>())
		OS << get<int>();
	else if (isA<float>())
		OS << get<float>();
	else if (isA<bool>())
		OS << (get<bool>() ? "true" : "false");
	else if (isA<char>())
		OS << get<char>();
	else if (isA<std::string>())
		OS << get<std::string>();
	else
		assert(false && "unrechable");
}
