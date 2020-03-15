#pragma once

#include <boost/graph/graph_selectors.hpp>
#include <boost/graph/properties.hpp>
#include <map>

#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/graph_traits.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/matching/MatchedEquationLookup.hpp"
#include "modelica/matching/SccLookup.hpp"
#include "modelica/model/EntryModel.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/model/VectorAccess.hpp"

namespace modelica
{
	using VVarGraph = boost::adjacency_list<
			boost::vecS,
			boost::vecS,
			boost::directedS,
			const IndexesOfEquation*,
			VectorAccess>;

	using VertexIndex =
			boost::property_map<VVarGraph, boost::vertex_index_t>::type::value_type;

	using VVarVertexDesc = boost::graph_traits<VVarGraph>::vertex_descriptor;

	class VVarDependencyGraph
	{
		public:
		VVarDependencyGraph(const EntryModel& model);
		void dump(llvm::raw_ostream& OS = llvm::outs()) const;
		[[nodiscard]] size_t count() const { return graph.m_vertices.size(); }
		[[nodiscard]] SccLookup<VertexIndex> getSCC() const;

		[[nodiscard]] const IndexesOfEquation& operator[](VertexIndex index) const
		{
			return *graph[index];
		}

		private:
		void populateEdge(
				const IndexesOfEquation& equation, const AccessToVar& toVariable);
		void populateEq(const IndexesOfEquation& eq);

		const EntryModel& model;
		VVarGraph graph;
		std::map<const ModEquation*, VVarVertexDesc> nodesLookup;
		MatchedEquationLookup lookUp;
	};

}	 // namespace modelica
