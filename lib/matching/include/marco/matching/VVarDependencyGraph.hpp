#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <map>
#include <variant>

#include "boost/graph/adjacency_list.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"
#include "marco/matching/MatchedEquationLookup.hpp"
#include "modelica/model/ModBltBlock.hpp"
#include "marco/model/ModEquation.hpp"
#include "marco/model/ModVariable.hpp"
#include "marco/model/Model.hpp"
#include "marco/model/VectorAccess.hpp"

namespace marco
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

		using VertexDesc = boost::graph_traits<GraphImp>::vertex_descriptor;

		using EdgeDesc = boost::graph_traits<GraphImp>::edge_descriptor;

		VVarDependencyGraph(const Model& model);
		VVarDependencyGraph(const Model& model, llvm::ArrayRef<ModEquation> equs);
		void dump(llvm::raw_ostream& OS = llvm::outs()) const;
		[[nodiscard]] size_t count() const { return graph.m_vertices.size(); }
		[[nodiscard]] const GraphImp& getImpl() const { return graph; }
		[[nodiscard]] GraphImp& getImpl() { return graph; }
		[[nodiscard]] const IndexesOfEquation& operator[](VertexDesc index) const
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

		[[nodiscard]] const Model& getModel() const { return model; }

		[[nodiscard]] VertexDesc target(EdgeDesc edge) const
		{
			return boost::target(edge, graph);
		}

		[[nodiscard]] VertexDesc source(EdgeDesc edge) const
		{
			return boost::source(edge, graph);
		}

		private:
		using EqToVert =
				std::map<const std::variant<ModEquation, ModBltBlock>*, VertexDesc>;
		void populateEdge(
				const IndexesOfEquation* equation,
				const AccessToVar& toVariable,
				EqToVert& eqToVert);
		void populateEq(const IndexesOfEquation* eq, EqToVert& eqToVert);
		void create();

		const Model& model;
		GraphImp graph;
		MatchedEquationLookup lookUp;
	};

	template<typename Graph, typename Vertex>
	auto outEdgesRange(Vertex& vertex, Graph& graph)
	{
		return llvm::make_range(boost::out_edges(vertex, graph));
	}

}	 // namespace marco
