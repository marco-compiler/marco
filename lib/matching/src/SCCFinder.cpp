#include "SCCFinder.hpp"

#include <algorithm>
#include <boost/graph/detail/adjacency_list.hpp>
#include <map>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "modelica/matching/MatchedEquationLookup.hpp"
#include "modelica/model/EntryModel.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModExpPath.hpp"
#include "modelica/model/ModMatchers.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/model/VectorAccess.hpp"
#include "modelica/utils/IRange.hpp"
#include "modelica/utils/IndexSet.hpp"
#include "modelica/utils/Interval.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;
using namespace boost;

void SCCFinder::populateEdge(
		const MatchedEquationLookup& lookup,
		const IndexesOfEquation& equation,
		const AccessToVar& toVariable)
{
	const auto& variable = model.getVar(toVariable.getVarName());
	const auto usedIndexes = toVariable.getAccess().map(equation.getIndexSet());
	for (const auto& var : lookup.eqsDeterminingVar(variable))
	{
		const auto& setOfVar = var.getIndexSet();
		if (setOfVar.disjoint(usedIndexes))
			continue;

		add_edge(
				nodesLookup[&equation.getEquation()],
				nodesLookup[&var.getEquation()],
				graph);
	}
}

void SCCFinder::populateEq(
		MatchedEquationLookup& lookup, const IndexesOfEquation& equation)
{
	ReferenceMatcher rightHandMatcher;
	rightHandMatcher.visitRight(equation.getEquation());
	for (const auto& toVariable : rightHandMatcher)
	{
		assert(VectorAccess::isCanonical(toVariable.getExp()));
		auto toAccess = AccessToVar::fromExp(toVariable.getExp());
		populateEdge(lookup, equation, toAccess);
	}
}

SCCFinder::SCCFinder(const EntryModel& m): model(m)
{
	MatchedEquationLookup lookUp(m);
	for (const auto& eq : lookUp)
		nodesLookup[&eq.getEquation()] = add_vertex(eq.getEquation(), graph);

	for (const auto& eq : lookUp)
		populateEq(lookUp, eq);
}
