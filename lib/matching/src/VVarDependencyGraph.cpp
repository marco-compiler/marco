#include "modelica/matching/VVarDependencyGraph.hpp"

#include <algorithm>
#include <boost/graph/detail/adjacency_list.hpp>
#include <boost/graph/strong_components.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <exception>
#include <map>
#include <sstream>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "modelica/matching/MatchedEquationLookup.hpp"
#include "modelica/matching/SccLookup.hpp"
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

void VVarDependencyGraph::populateEdge(
		const IndexesOfEquation& equation, const AccessToVar& toVariable)
{
	const auto& variable = model.getVar(toVariable.getVarName());
	const auto usedIndexes =
			toVariable.getAccess().map(equation.getEquation().toInterval());
	for (const auto& var : lookUp.eqsDeterminingVar(variable))
	{
		const auto& setOfVar = var.getInterval();
		if (areDisjoint(usedIndexes, setOfVar))
			continue;

		add_edge(
				nodesLookup[&equation.getEquation()],
				nodesLookup[&var.getEquation()],
				toVariable.getAccess(),
				graph);
	}
}

void VVarDependencyGraph::populateEq(const IndexesOfEquation& equation)
{
	ReferenceMatcher rightHandMatcher;
	rightHandMatcher.visitRight(equation.getEquation());
	for (const auto& toVariable : rightHandMatcher)
	{
		assert(VectorAccess::isCanonical(toVariable.getExp()));
		auto toAccess = AccessToVar::fromExp(toVariable.getExp());
		populateEdge(equation, toAccess);
	}
}

VVarDependencyGraph::VVarDependencyGraph(const EntryModel& m)
		: model(m), lookUp(m)
{
	for (const auto& eq : lookUp)
		nodesLookup[&eq.getEquation()] = add_vertex(&eq, graph);

	for (const auto& eq : lookUp)
		populateEq(eq);
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
		auto from = source(edge, graph);
		auto to = target(edge, graph);

		OS << from << "->" << to << "[label=\"";
		graph[edge].dump(OS);
		OS << "\"];\n";
	}

	OS << "}";
}

void boost::throw_exception(const std::exception& e)
{
	errs() << e.what();
	assert(false);
}

SccLookup<VVarDependencyGraph::VertexIndex> VVarDependencyGraph::getSCC() const
{
	SccLookup<VertexIndex>::InputVector components(count());

	auto componentsCount = strong_components(
			graph,
			make_iterator_property_map(components.begin(), get(vertex_index, graph)));

	return SccLookup<VertexIndex>(components, componentsCount);
}
