#include "modelica/model/ModEquation.hpp"

using namespace std;
using namespace llvm;
using namespace modelica;

IndexSet ModEquation::toIndexSet() const
{
	SmallVector<Interval, 2> intervals;

	for (const auto& induction : getInductions())
		intervals.emplace_back(induction.begin(), induction.end());

	return IndexSet({ MultiDimInterval(move(intervals)) });
}
