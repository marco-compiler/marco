#pragma once

#include <boost/graph/graph_selectors.hpp>
#include <boost/graph/properties.hpp>
#include <map>

#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/graph_traits.hpp"
#include "modelica/matching/MatchedEquationLookup.hpp"
#include "modelica/model/EntryModel.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/model/VectorAccess.hpp"

namespace modelica
{
	struct EquationPropertyImpl
	{
		using kind = boost::vertex_property_tag;
	};

	class EquationVertex
	{
		public:
		EquationVertex(const ModEquation& eq): equation(&eq) {}
		EquationVertex() = default;

		[[nodiscard]] const ModEquation& getEquation() const { return *equation; }

		private:
		const ModEquation* equation;
	};

	// using EquationProperty =
	// boost::property<EquationPropertyImpl, EquationVertex>;

	using SCCGraph = boost::adjacency_list<
			boost::vecS,
			boost::vecS,
			boost::directedS,
			EquationVertex>;

	// using SCCGraphPropertyMap =
	// boost::property_map<SCCGraph, EquationPropertyImpl>::type;

	using SCCVertexDescriptor = boost::graph_traits<SCCGraph>::vertex_descriptor;

	class SCCFinder
	{
		public:
		SCCFinder(const EntryModel& model);

		private:
		void populateEdge(
				const MatchedEquationLookup& lookup,
				const IndexesOfEquation& equation,
				const AccessToVar& toVariable);
		void populateEq(MatchedEquationLookup& lookup, const IndexesOfEquation& eq);

		const EntryModel& model;
		SCCGraph graph;
		std::map<const ModEquation*, SCCVertexDescriptor> nodesLookup;
	};
}	 // namespace modelica
