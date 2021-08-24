#include "marco/model/ModVariable.hpp"

#include "marco/utils/Interval.hpp"

using namespace llvm;
using namespace std;
using namespace marco;

IndexSet ModVariable::toIndexSet() const
{
	return IndexSet({ toMultiDimInterval() });
}
MultiDimInterval ModVariable::toMultiDimInterval() const
{
	SmallVector<Interval, 2> intervals;

	const auto& type = getInit().getModType();
	for (size_t a = 0; a < type.getDimensionsCount(); a++)
		intervals.emplace_back(0, type.getDimension(a));

	return MultiDimInterval(move(intervals));
}

void ModVariable::dump(llvm::raw_ostream& OS) const
{
	if (isState())
		OS << "state ";
	if (isConstant())
		OS << "const ";
	OS << name << " = ";
	getInit().dump(OS);
	OS << '\n';
}
