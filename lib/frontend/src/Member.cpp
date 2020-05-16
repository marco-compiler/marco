#include "modelica/frontend/Member.hpp"

using namespace llvm;
using namespace std;
using namespace modelica;

void Member::dump(llvm::raw_ostream& OS, size_t indents)
{
	OS.indent(indents);
	OS << "member " << name << " type : ";
	type.dump(OS);
	OS << (isParam ? "param" : "");
	OS << "\n";

	if (hasInitializer())
	{
		OS.indent(indents);
		OS << "initializer: \n";
		initializer->dump(OS, indents + 1);
		OS << "\n";
	}

	if (hasStartOverload())
	{
		OS.indent(indents);
		OS << "start overload: ";
		startOverload->dump(OS);
		OS << "\n";
	}
}
