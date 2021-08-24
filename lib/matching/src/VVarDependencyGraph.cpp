#include "marco/matching/VVarDependencyGraph.hpp"

#include <algorithm>
#include <boost/graph/strong_components.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/range/iterator_range_core.hpp>
#if BOOST_VERSION >= 107300
#include <boost/assert/source_location.hpp>
#endif
#include <exception>
#include <llvm/ADT/ArrayRef.h>
#include <map>
#include <sstream>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "marco/matching/MatchedEquationLookup.hpp"
#include "marco/matching/SccLookup.hpp"
#include "marco/model/ModEquation.hpp"
#include "marco/model/ModExp.hpp"
#include "marco/model/ModExpPath.hpp"
#include "marco/model/ModMatchers.hpp"
#include "marco/model/ModVariable.hpp"
#include "marco/model/Model.hpp"
#include "marco/model/VectorAccess.hpp"
#include "marco/utils/IRange.hpp"
#include "marco/utils/IndexSet.hpp"
#include "marco/utils/Interval.hpp"

using namespace marco;
using namespace std;
using namespace llvm;
using namespace boost;

void VVarDependencyGraph::populateEdge(
		const IndexesOfEquation& equation,
		const AccessToVar& toVariable,
		EqToVert& eqToVert)
{
	const auto& variable = model.getVar(toVariable.getVarName());
	const auto usedIndexes =
			toVariable.getAccess().map(equation.getEquation().getInductions());
	for (const auto& var : lookUp.eqsDeterminingVar(variable))
	{
		const auto& setOfVar = var.getInterval();
		if (areDisjoint(usedIndexes, setOfVar))
			continue;

		add_edge(
				eqToVert[&equation.getEquation()],
				eqToVert[&var.getEquation()],
				toVariable.getAccess(),
				graph);
	}
}

void VVarDependencyGraph::populateEq(
		const IndexesOfEquation& equation, EqToVert& eqToVert)
{
	ReferenceMatcher rightHandMatcher;
	rightHandMatcher.visit(equation.getEquation(), true);
	for (const auto& toVariable : rightHandMatcher)
	{
		assert(VectorAccess::isCanonical(toVariable.getExp()));
		auto toAccess = AccessToVar::fromExp(toVariable.getExp());
		populateEdge(equation, toAccess, eqToVert);
	}
}

void VVarDependencyGraph::create(ArrayRef<ModEquation> equs)
{
	EqToVert eqToVert;

	for (const auto& eq : lookUp)
		eqToVert[&eq.getEquation()] = add_vertex(&eq, graph);

	for (const auto& eq : lookUp)
		populateEq(eq, eqToVert);
}

VVarDependencyGraph::VVarDependencyGraph(
		const Model& m, ArrayRef<ModEquation> equs)
		: model(m), lookUp(m, equs)
{
	create(equs);
}

VVarDependencyGraph::VVarDependencyGraph(const Model& m): model(m), lookUp(m)
{
	create(model.getEquations());
}

void VVarDependencyGraph::dump(llvm::raw_ostream& OS) const
{
	OS << "digraph G {";
	for (auto vertex : make_iterator_range(vertices(graph)))
	{
		OS << vertex << "[label=\"" << vertex << "\"]";
	}

	for (auto edge : make_iterator_range(edges(graph)))
	{
		auto from = source(edge);
		auto to = target(edge);

		OS << from << "->" << to << "[label=\"";
		graph[edge].dump(OS);
		OS << "\"];\n";
	}

	OS << "}";
}

void boost::throw_exception(const std::exception& e)
{
	errs() << e.what();
	abort();
}

#if BOOST_VERSION >= 107300 /* source_location only exists in boost >= 1.73 */

void boost::throw_exception(const std::exception& e, const boost::source_location& loc)
{
	errs() << e.what();
	abort();
}

#endif
