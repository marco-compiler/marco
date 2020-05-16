#include "modelica/frontend/Class.hpp"

using namespace std;
using namespace llvm;
using namespace modelica;

void Class::dump(llvm::raw_ostream& OS, size_t indents)
{
	OS.indent(indents);
	OS << "class " << name << "\n";
	for (auto& mem : members)
	{
		mem.dump(OS, indents + 1);
		OS << "\n";
	}

	for (auto& eq : equations)
	{
		eq.dump(OS, indents + 1);
		OS << "\n";
	}

	for (auto& eq : forEquations)
	{
		eq.dump(OS, indents + 1);
		OS << "\n";
	}
}

Class::Class(
		std::string name,
		llvm::SmallVector<Member, 3> memb,
		llvm::SmallVector<Equation, 3> equs,
		llvm::SmallVector<ForEquation, 3> forEqus)
		: name(std::move(name)),
			members(std::move(memb)),
			equations(std::move(equs)),
			forEquations(std::move(forEqus))
{
}
