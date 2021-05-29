#include "marco/matching/VVarDependencyGraph.hpp"

#include <algorithm>
#include <boost/graph/strong_components.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/range/iterator_range_core.hpp>
#if BOOST_VERSION >= 107300
#	include <boost/assert/source_location.hpp>
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
#include "modelica/model/ModBltBlock.hpp"
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

/**
 * For every node, add an edge to every other node that depend on the matched
 * variable of that node.
 */
void VVarDependencyGraph::populateEdge(
		const IndexesOfEquation* content,
		const AccessToVar& toVariable,
		EqToVert& eqToVert)
{
	const ModVariable& variable = model.getVar(toVariable.getVarName());

	for (const ModEquation& equation : content->getEquations())
	{
		const MultiDimInterval usedIndexes =
				toVariable.getAccess().map(equation.getInductions());

		for (const IndexesOfEquation* var : lookUp.eqsDeterminingVar(variable))
		{
			for (const MultiDimInterval& setOfVar : var->getIntervals())
			{
				if (areDisjoint(usedIndexes, setOfVar))
					continue;

				add_edge(
						eqToVert[&content->getContent()],
						eqToVert[&var->getContent()],
						toVariable.getAccess(),
						graph);
			}
		}
	}
}

/**
 * Search for all dependencies among nodes and add an edge between them.
 */
void VVarDependencyGraph::populateEq(
		const IndexesOfEquation* content, EqToVert& eqToVert)
{
	ReferenceMatcher rightHandMatcher;

	// Search for dependencies among the non-matched variables
	rightHandMatcher.visit(content->getContent(), true);
	for (const ModExpPath& toVariable : rightHandMatcher)
	{
		assert(VectorAccess::isCanonical(toVariable.getExp()));
		AccessToVar toAccess = AccessToVar::fromExp(toVariable.getExp());

		// Add an edge for each dependency
		populateEdge(content, toAccess, eqToVert);
	}
}

/**
 * Add all nodes to the graph. A node is an IndexesOfEquation which contains
 * either an Equation or a BltBlock. Then add all edges among these nodes.
 */
void VVarDependencyGraph::create()
{
	EqToVert eqToVert;

	// Add all nodes
	for (const IndexesOfEquation* content : lookUp)
	{
		// Do not add duplicate IndexesOfEquation to the map.
		// This could be caused by multiple equations in a ModBltBlock.
		if (eqToVert.find(&content->getContent()) == eqToVert.end())
			eqToVert[&content->getContent()] = add_vertex(content, graph);
	}

	// Add all edges
	for (const IndexesOfEquation* content : lookUp)
		populateEq(content, eqToVert);
}

VVarDependencyGraph::VVarDependencyGraph(
		const Model& m, ArrayRef<ModEquation> equs)
		: model(m), lookUp(m, equs)
{
	create();
}

VVarDependencyGraph::VVarDependencyGraph(const Model& m): model(m), lookUp(m)
{
	create();
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

void boost::throw_exception(
		const std::exception& e, const boost::source_location& loc)
{
	errs() << e.what();
	abort();
}

#endif
