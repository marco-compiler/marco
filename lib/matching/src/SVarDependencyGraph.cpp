#include "modelica/matching/SVarDependencyGraph.hpp"

#include <boost/graph/detail/adjacency_list.hpp>

#include "modelica/matching/VVarDependencyGraph.hpp"

using namespace modelica;
using namespace llvm;
using namespace boost;
using namespace std;

SVarDepencyGraph::SVarDepencyGraph(
		const VVarDependencyGraph& collapsedGraph, const VVarScc& scc)
		: scc(scc), collapsedGraph(collapsedGraph)
{
	for (const auto& vertex : scc.range(collapsedGraph))
	{
		const auto& interval = vertex.getInterval();
		for (auto indicies : interval.contentRange())
			add_vertex(SingleEquationReference(vertex, indicies), graph);
	}
}
