#include "modelica/model/ModVariable.hpp"

using namespace llvm;
using namespace std;
using namespace modelica;

IndexSet ModVariable::toIndexSet() const
{
	SmallVector<Interval, 2> intervals;

	const auto& type = getInit().getModType();
	for (size_t a = 0; a < type.getDimensionsCount(); a++)
		intervals.emplace_back(0, type.getDimension(a));

	return IndexSet({ MultiDimInterval(move(intervals)) });
}

void ModVariable::dump(llvm::raw_ostream& OS) const
{
	if (!isState())
		OS << "const ";
	OS << name << " = ";
	getInit().dump(OS);
	OS << '\n';
}
