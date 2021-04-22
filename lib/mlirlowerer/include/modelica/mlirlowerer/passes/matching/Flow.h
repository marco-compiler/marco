#pragma once

#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/Variable.h>
#include <modelica/utils/IndexSet.hpp>

#include "Matching.h"

namespace modelica::codegen::model
{
	class Edge;

	/**
	 * The flow class is used to search a augmenting path in the
	 * the matching graph. It contains the reference to the underlying edge and
	 * the direction the edge is followed. It keeps track of a index set and the
	 * mapped set at the destination, so querying for both the from and the to
	 * index is a constant time operation.
	 */
	class Flow
	{
		public:
		[[nodiscard]] Edge& getEdge();
		[[nodiscard]] const Edge& getEdge() const;

		[[nodiscard]] Variable getVariable() const;
		[[nodiscard]] Equation getEquation() const;

		[[nodiscard]] const IndexSet& getSet() const;

		[[nodiscard]] bool empty() const;
		[[nodiscard]] size_t size() const;

		[[nodiscard]] const IndexSet& getMappedSet() const;

		[[nodiscard]] bool isForwardEdge() const;

		void addFlowAtEnd(IndexSet& set);

		[[nodiscard]] IndexSet inverseMap(const IndexSet& set) const;

		[[nodiscard]] IndexSet applyAndInvert(IndexSet set);

		[[nodiscard]] static Flow forwardEdge(Edge& edge, IndexSet set);
		[[nodiscard]] static Flow backEdge(Edge& edge, IndexSet set);

		[[nodiscard]] static bool compare(const Flow& l, const Flow& r, const MatchingGraph& g);

		private:
		Flow(IndexSet set, IndexSet mapped, Edge& edge, bool isForward);

		Edge* edge;
		IndexSet set;
		IndexSet mappedFlow;
		bool isForward;
	};

	class FlowCandidates
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using iterator = Container<Flow>::iterator;
		using const_iterator = Container<Flow>::const_iterator;

		FlowCandidates(llvm::SmallVector<Flow, 2> c, const MatchingGraph& g);

		[[nodiscard]] bool empty() const;

		[[nodiscard]] const_iterator begin() const;
		[[nodiscard]] const_iterator end() const;

		[[nodiscard]] Flow& getCurrent();
		[[nodiscard]] const Flow& getCurrent() const;

		[[nodiscard]] Variable getCurrentVariable() const;

		void pop();

		private:
		Container<Flow> choices;
	};

	class AugmentingPath
	{
		public:
		AugmentingPath(
				MatchingGraph& graph,
				size_t maxDepth = std::numeric_limits<size_t>::max());

		[[nodiscard]] size_t size() const;

		[[nodiscard]] const FlowCandidates& getCurrentCandidates() const;
		[[nodiscard]] FlowCandidates& getCurrentCandidates();

		[[nodiscard]] FlowCandidates getBestCandidate() const;

		[[nodiscard]] Flow& getCurrentFlow();
		[[nodiscard]] const Flow& getCurrentFlow() const;

		[[nodiscard]] FlowCandidates selectStartingEdge() const;
		[[nodiscard]] IndexSet possibleBackwardFlow(const Edge& backEdge) const;

		[[nodiscard]] bool valid() const;

		void apply();

		private:
		[[nodiscard]] FlowCandidates getBackwardMatchable() const;
		[[nodiscard]] FlowCandidates getForwardMatchable() const;

		MatchingGraph& graph;
		llvm::SmallVector<FlowCandidates, 2> frontier;
	};
}
