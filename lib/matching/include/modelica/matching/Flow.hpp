#pragma once
#include "llvm/Support/raw_ostream.h"
#include "modelica/matching/Edge.hpp"
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
				return edge->map(set);
			return edge->invertMap(set);
		}

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
		[[nodiscard]] auto begin() const { return choises.begin(); }
		[[nodiscard]] auto begin() { return choises.begin(); }
		[[nodiscard]] auto end() const { return choises.end(); }
		[[nodiscard]] auto end() { return choises.end(); }
		FlowCandidates(llvm::SmallVector<Flow, 2> c)
				: choises(std::move(c)), current(0)
		{
			sort();
		}

		void sort() { llvm::sort(begin(), end(), Flow::compare); }
		[[nodiscard]] bool empty() const { return choises.empty(); }
		[[nodiscard]] bool allVisited() const { return current >= choises.size(); }
		void next()
		{
			do
				current++;
			while (current < choises.size() && choises[current].getSet().empty());
		}
		[[nodiscard]] Flow& getCurrent() { return choises[current]; }
		[[nodiscard]] const Flow& getCurrent() const { return choises[current]; }
		[[nodiscard]] const ModVariable& getCurrentVariable() const
		{
			assert(current < choises.size());
			return getCurrent().getEdge().getVariable();
		}
		void dump(llvm::raw_ostream& OS = llvm::outs()) const;
		[[nodiscard]] std::string toString() const;

		private:
		llvm::SmallVector<Flow, 2> choises;
		size_t current;
	};
}	 // namespace modelica
