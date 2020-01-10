#pragma once
#include <map>

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/model/Model.hpp"
#include "modelica/model/VectorAccess.hpp"
#include "modelica/utils/IndexSet.hpp"

namespace modelica
{
	class MatchedEquation
	{
		public:
		MatchedEquation(
				const ModEquation& d, const ModVariable& variable, size_t index)
				: data(&d), variable(&variable), index(index)
		{
		}
		[[nodiscard]] const ModEquation& getEquation() const { return *data; }
		[[nodiscard]] const IndexSet& getSet() const { return set; }
		[[nodiscard]] IndexSet& getSet() { return set; }
		[[nodiscard]] const ModVariable& getVariable() const { return *variable; }
		[[nodiscard]] size_t getIndex() const { return index; }

		private:
		const ModEquation* data;
		IndexSet set;
		const ModVariable* variable;
		size_t index;
	};

	using Matching = llvm::SmallVector<MatchedEquation, 0>;

	class Edge
	{
		public:
		Edge(const ModEquation& eq, const ModVariable& var, size_t index)
				: vectorAccess(var.getName()),
					invertedAccess(vectorAccess.invert()),
					matched(eq, var, index)
		{
		}
		Edge(
				const ModEquation& eq,
				const ModVariable& var,
				size_t index,
				llvm::SmallVector<Displacement, 3> acc)
				: vectorAccess(var.getName(), std::move(acc)),
					invertedAccess(vectorAccess.invert()),
					matched(eq, var, index)
		{
		}
		Edge(
				const ModEquation& eq,
				const ModVariable& var,
				size_t index,
				VectorAccess acc)
				: vectorAccess(std::move(acc)),
					invertedAccess(vectorAccess.invert()),
					matched(eq, var, index)
		{
		}
		[[nodiscard]] const ModEquation& getEquation() const
		{
			return matched.getEquation();
		}
		[[nodiscard]] const ModVariable& getVariable() const
		{
			return matched.getVariable();
		}

		[[nodiscard]] MatchedEquation& getMatchedEquation() { return matched; }
		[[nodiscard]] const MatchedEquation& getMatchedEquation() const
		{
			return matched;
		}
		[[nodiscard]] IndexSet& getSet() { return matched.getSet(); }
		[[nodiscard]] const IndexSet& getSet() const { return matched.getSet(); }
		[[nodiscard]] const VectorAccess& getVectorAccess() const
		{
			return vectorAccess;
		}

		[[nodiscard]] IndexSet map(const IndexSet& set) const
		{
			return vectorAccess.map(set);
		}

		[[nodiscard]] IndexSet invertMap(const IndexSet& set) const
		{
			return invertedAccess.map(set);
		}

		private:
		VectorAccess vectorAccess;
		VectorAccess invertedAccess;
		MatchedEquation matched;
	};

	class EdgeFlow
	{
		public:
		EdgeFlow(Edge& edge, IndexSet set, bool isForward)
				: edge(&edge),
					set(std::move(set)),
					mappedFlow(edge.map(set)),
					isForward(isForward)
		{
		}

		EdgeFlow(Edge& edge, bool isForward): edge(&edge), isForward(isForward) {}

		static EdgeFlow backedge(Edge& edge, IndexSet set)
		{
			return EdgeFlow(std::move(set), edge, false);
		}
		static EdgeFlow forwardedge(Edge& edge, IndexSet set)
		{
			return EdgeFlow(std::move(set), edge, true);
		}

		[[nodiscard]] const Edge& getEdge() const { return *edge; }
		[[nodiscard]] const ModEquation& getEquation() const
		{
			return edge->getEquation();
		}
		[[nodiscard]] const IndexSet& getSet() const { return set; }
		[[nodiscard]] const IndexSet& getMappedSet() const { return mappedFlow; }
		[[nodiscard]] size_t size() const { return set.size(); }

		[[nodiscard]] static bool compare(const EdgeFlow& l, const EdgeFlow& r)
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

		private:
		EdgeFlow(IndexSet set, Edge& edge, bool isForward)
				: edge(&edge),
					set(edge.invertMap(set)),
					mappedFlow(std::move(set)),
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
		FlowCandidates(llvm::SmallVector<EdgeFlow, 2> c)
				: choises(std::move(c)), current(0)
		{
			sort();
		}

		void sort() { llvm::sort(begin(), end(), EdgeFlow::compare); }
		[[nodiscard]] bool empty() const { return choises.empty(); }
		[[nodiscard]] bool allVisited() const { return current >= choises.size(); }
		void next()
		{
			do
				current++;
			while (current < choises.size() && choises[current].getSet().empty());
		}
		[[nodiscard]] EdgeFlow& getCurrent() { return choises[current]; }
		[[nodiscard]] const EdgeFlow& getCurrent() const
		{
			return choises[current];
		}
		[[nodiscard]] const ModVariable& getCurrentVariable() const
		{
			assert(current < choises.size());
			return getCurrent().getEdge().getVariable();
		}

		private:
		llvm::SmallVector<EdgeFlow, 2> choises;
		size_t current;
	};

	class MatchingGraph
	{
		public:
		MatchingGraph(const Model& model): model(model)
		{
			for (const auto& eq : model)
				addEquation(eq);
		}

		template<typename Callable>
		void forAllConnected(const ModEquation& equation, Callable c)
		{
			const ModEquation* eq = &equation;
			auto [begin, end] = equationLookUp.equal_range(eq);
			while (begin != end)
			{
				c(edges[begin->second]);
				begin++;
			}
		}
		template<typename Callable>
		void forAllConnected(const ModVariable& variable, Callable c)
		{
			const ModVariable* var = &variable;
			auto [begin, end] = variableLookUp.equal_range(var);
			while (begin != end)
			{
				c(edges[begin->second]);
				begin++;
			}
		}
		template<typename Callable>
		void forAllConnected(const ModEquation& equation, Callable c) const
		{
			const ModEquation* eq = &equation;
			auto [begin, end] = equationLookUp.equal_range(eq);
			while (begin != end)
			{
				c(edges[begin->second]);
				begin++;
			}
		}
		template<typename Callable>
		void forAllConnected(const ModVariable& variable, Callable c) const
		{
			const ModVariable* var = &variable;
			auto [begin, end] = variableLookUp.equal_range(var);
			while (begin != end)
			{
				c(edges[begin->second]);
				begin++;
			}
		}
		[[nodiscard]] FlowCandidates selectStartingEdge();
		[[nodiscard]] llvm::SmallVector<EdgeFlow, 2> findAugmentingPath();
		[[nodiscard]] bool updatePath(llvm::SmallVector<EdgeFlow, 2> flow);

		[[nodiscard]] IndexSet getUnmatchedSet(const ModVariable& variable) const
		{
			auto set = variable.toIndexSet();
			set.remove(getMatchedSet(variable));
			return set;
		}

		[[nodiscard]] IndexSet getUnmatchedSet(const ModEquation& equation) const
		{
			IndexSet matched;
			const auto unite = [&matched](const auto& edge) {
				matched.unite(edge.getSet());
			};
			forAllConnected(equation, unite);

			auto set = equation.toIndexSet();
			set.dump(llvm::outs());
			matched.dump(llvm::outs());
			set.remove(matched);
			return set;
		}

		[[nodiscard]] IndexSet getMatchedSet(const ModVariable& variable) const
		{
			IndexSet matched;
			const auto unite = [&matched](const auto& edge) {
				matched.unite(edge.getSet());
			};
			forAllConnected(variable, unite);

			return matched;
		}

		[[nodiscard]] auto begin() const { return edges.begin(); }
		[[nodiscard]] auto begin() { return edges.begin(); }
		[[nodiscard]] auto end() { return edges.end(); }
		[[nodiscard]] auto end() const { return edges.end(); }
		[[nodiscard]] const Model& getModel() const { return model; }
		[[nodiscard]] size_t variableCount() const
		{
			return model.getVars().size();
		}
		[[nodiscard]] size_t equationCount() const
		{
			return model.getEquations().size();
		}
		[[nodiscard]] size_t edgesCount() const { return edges.size(); }
		void dumpGraph(llvm::raw_ostream& OS) const;
		void match(int maxIterations);
		[[nodiscard]] Matching toMatch() const;
		[[nodiscard]] Matching extractMatch();

		private:
		void addEquation(const ModEquation& eq);

		void addVariableUsage(
				const ModEquation& eq, const ModExp& use, size_t index);

		template<typename... T>
		void emplace_edge(T&&... args)	// NOLINT
		{
			edges.emplace_back(std::forward<T>(args)...);

			size_t edgeIndex = edges.size() - 1;
			const ModEquation* eq = &(edges.back().getEquation());
			equationLookUp.insert({ eq, edgeIndex });
			auto var = &(edges.back().getVariable());
			variableLookUp.insert({ var, edgeIndex });
		}
		llvm::SmallVector<Edge, 0> edges;
		std::multimap<const ModEquation*, size_t> equationLookUp;
		std::multimap<const ModVariable*, size_t> variableLookUp;
		const Model& model;
	};

}	 // namespace modelica
