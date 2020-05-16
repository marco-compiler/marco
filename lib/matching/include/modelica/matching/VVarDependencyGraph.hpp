#pragma once

#include <map>

#include "boost/graph/adjacency_list.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/matching/MatchedEquationLookup.hpp"
#include "modelica/matching/SccLookup.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/model/Model.hpp"
#include "modelica/model/VectorAccess.hpp"

namespace modelica
{
	class VVarDependencyGraph
	{
		public:
		using GraphImp = boost::adjacency_list<
				boost::vecS,
				boost::vecS,
				boost::directedS,
				const IndexesOfEquation*,
				VectorAccess>;

		using VertexIndex =
				boost::property_map<GraphImp, boost::vertex_index_t>::type::value_type;

		using VertexDesc = boost::graph_traits<GraphImp>::vertex_descriptor;

		using EdgeDesc = boost::graph_traits<GraphImp>::edge_descriptor;

		VVarDependencyGraph(const Model& model);
		void dump(llvm::raw_ostream& OS = llvm::outs()) const;
		[[nodiscard]] size_t count() const { return graph.m_vertices.size(); }
		[[nodiscard]] SccLookup<VertexIndex> getSCC() const;
		[[nodiscard]] const GraphImp& getImpl() const { return graph; }
		[[nodiscard]] GraphImp& getImpl() { return graph; }
		[[nodiscard]] const IndexesOfEquation& operator[](VertexIndex index) const
		{
			return *graph[index];
		}

		[[nodiscard]] auto outEdges(VertexDesc vertex)
		{
			return llvm::make_range(boost::out_edges(vertex, graph));
		}

		[[nodiscard]] auto outEdges(VertexDesc vertex) const
		{
			return llvm::make_range(boost::out_edges(vertex, graph));
		}

		[[nodiscard]] VectorAccess& operator[](EdgeDesc edge)
		{
			return graph[edge];
		}
		[[nodiscard]] const VectorAccess& operator[](EdgeDesc edge) const
		{
			return graph[edge];
		}

		[[nodiscard]] VertexDesc target(EdgeDesc edge) const
		{
			return boost::target(edge, graph);
		}

		[[nodiscard]] VertexDesc source(EdgeDesc edge) const
		{
			return boost::source(edge, graph);
		}

		private:
		void populateEdge(
				const IndexesOfEquation& equation, const AccessToVar& toVariable);
		void populateEq(const IndexesOfEquation& eq);

		const Model& model;
		GraphImp graph;
		std::map<const ModEquation*, VertexDesc> nodesLookup;
		MatchedEquationLookup lookUp;
	};

	template<typename Graph, typename Vertex>
	auto outEdgesRange(Vertex& vertex, Graph& graph)
	{
		return llvm::make_range(boost::out_edges(vertex, graph));
	}

}	 // namespace modelica