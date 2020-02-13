#pragma once
#include <limits>

#include "llvm/Support/raw_ostream.h"
#include "modelica/matching/Edge.hpp"
#include "modelica/matching/Matching.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/utils/IndexSet.hpp"

namespace modelica
{
	/**
	 * The flow class is used to search a augmenting path in the
	 * the matching graph. It contains the refernce to the underlaying edge and
	 * the direction the edge is followed. It keeps track of a index set and the
	 * mapped set at the destination, so querying for both the from and the to
	 * index is a constant time operation.
	 */
	class Flow
	{
		public:
		static Flow backedge(Edge& edge, IndexSet set)
		{
			IndexSet mapped = edge.invertMap(set);
			return Flow(std::move(mapped), std::move(set), edge, false);
		}
		static Flow forwardedge(Edge& edge, IndexSet set)
		{
			IndexSet mapped = edge.map(set);
			return Flow(std::move(set), std::move(mapped), edge, true);
		}
		[[nodiscard]] const ModVariable& getVariable() const
		{
			return getEdge().getVariable();
		}
		[[nodiscard]] Edge& getEdge() { return *edge; }
		[[nodiscard]] const Edge& getEdge() const { return *edge; }
		[[nodiscard]] const ModEquation& getEquation() const
		{
			return edge->getEquation();
		}
		[[nodiscard]] const IndexSet& getSet() const { return set; }
		[[nodiscard]] const IndexSet& getMappedSet() const { return mappedFlow; }
		[[nodiscard]] size_t size() const { return set.size(); }

		[[nodiscard]] static bool compare(const Flow& l, const Flow& r)
		{
			return l.size() < r.size();
		};
		[[nodiscard]] bool isForwardEdge() const { return isForward; }
		void addFLowAtEnd(IndexSet& set)
		{
			if (isForwardEdge())
				edge->getSet().unite(set);
			else
				edge->getSet().remove(set);
		}
		[[nodiscard]] IndexSet inverseMap(const IndexSet& set) const
		{
			if (isForwardEdge())
				return edge->invertMap(set);
			return edge->map(set);
		}

		[[nodiscard]] IndexSet applyAndInvert(IndexSet set)
		{
			if (isForwardEdge())
				set = inverseMap(set);

			addFLowAtEnd(set);

			if (!isForwardEdge())
				set = inverseMap(set);
			return set;
		}

		[[nodiscard]] bool empty() const { return set.empty(); }

		void dump(llvm::raw_ostream& OS = llvm::outs()) const;

		private:
		Flow(IndexSet set, IndexSet mapped, Edge& edge, bool isForward)
				: edge(&edge),
					set(std::move(set)),
					mappedFlow(std::move(mapped)),
					isForward(isForward)
		{
		}
		Edge* edge;
		IndexSet set;
		IndexSet mappedFlow;
		bool isForward;
	};

	class FlowCandidates
	{
		public:
		FlowCandidates(llvm::SmallVector<Flow, 2> c);

		[[nodiscard]] bool empty() const { return choises.empty(); }

		void pop()
		{
			assert(choises.begin() != choises.end());
			auto last = choises.end();
			last--;
			choises.erase(last, choises.end());
		}
		[[nodiscard]] Flow& getCurrent()
		{
			assert(!choises.empty());
			return choises.back();
		}
		[[nodiscard]] const Flow& getCurrent() const
		{
			assert(!choises.empty());
			return choises.back();
		}
		[[nodiscard]] const ModVariable& getCurrentVariable() const
		{
			return getCurrent().getEdge().getVariable();
		}
		void dump(llvm::raw_ostream& OS = llvm::outs()) const;
		[[nodiscard]] std::string toString() const;
		[[nodiscard]] auto begin() const { return choises.begin(); }
		[[nodiscard]] auto end() const { return choises.end(); }

		private:
		llvm::SmallVector<Flow, 2> choises;
	};

	class AugmentingPath
	{
		public:
		AugmentingPath(
				MatchingGraph& graph,
				size_t maxDepth = std::numeric_limits<size_t>::max());
		[[nodiscard]] bool valid() const;
		[[nodiscard]] FlowCandidates getBestCandidate() const;
		[[nodiscard]] const FlowCandidates& getCurrentCandidates() const
		{
			return frontier.back();
		}
		[[nodiscard]] FlowCandidates& getCurrentCandidates()
		{
			return frontier.back();
		}
		[[nodiscard]] Flow& getCurrentFlow()
		{
			return getCurrentCandidates().getCurrent();
		}
		[[nodiscard]] const Flow& getCurrentFlow() const
		{
			return getCurrentCandidates().getCurrent();
		}
		[[nodiscard]] FlowCandidates selectStartingEdge() const;
		void apply();
		void dump(llvm::raw_ostream& OS = llvm::outs()) const;
		void dumpGraph(
				llvm::raw_ostream& OS,
				bool displayEmptyEdges,
				bool displayMappings,
				bool displayOnlyMatchedCount,
				bool displayOtherOptions) const;
		[[nodiscard]] std::string toString() const;
		[[nodiscard]] size_t size() const { return frontier.size(); }
		[[nodiscard]] IndexSet possibleBackwardFlow(const Edge& backEdge) const;

		private:
		[[nodiscard]] FlowCandidates getBackwardMatchable() const;

		[[nodiscard]] FlowCandidates getForwardMatchable() const;
		MatchingGraph& graph;
		llvm::SmallVector<FlowCandidates, 2> frontier;
	};
}	 // namespace modelica
