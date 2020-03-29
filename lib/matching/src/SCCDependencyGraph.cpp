#include "modelica/matching/SCCDependencyGraph.hpp"

#include <boost/graph/detail/adjacency_list.hpp>
#include <boost/range/iterator_range_core.hpp>

using namespace boost;
using namespace modelica;
using namespace std;

SCCDependencyGraph::SCCDependencyGraph(
		SccLookup<size_t>& lookUp, VVarDependencyGraph& originalGraph)
		: sccLookup(lookUp), originalGraph(originalGraph)
{
	map<const Scc<size_t>*, size_t> insertedVertex;
	for (const auto& scc : lookUp)
		insertedVertex[&scc] = add_vertex(&scc, graph);

	auto edgeIterator = make_iterator_range(edges(originalGraph.getImpl()));
	for (const auto& edge : edgeIterator)
	{
		size_t targetVertex = target(edge, originalGraph.getImpl());
		size_t sourceVertex = source(edge, originalGraph.getImpl());

		if (targetVertex == sourceVertex)
			continue;

		const auto& targetScc = lookUp.sccOf(targetVertex);
		const auto& sourceScc = lookUp.sccOf(sourceVertex);

		add_edge(insertedVertex[&sourceScc], insertedVertex[&targetScc], graph);
	}
}
