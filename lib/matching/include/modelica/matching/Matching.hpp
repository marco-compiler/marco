#pragma once
#include <map>

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/matching/Edge.hpp"
#include "modelica/matching/Flow.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/model/Model.hpp"
#include "modelica/model/VectorAccess.hpp"
#include "modelica/utils/IndexSet.hpp"

namespace modelica
{
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
		void forAllConnected(const ModVariable& var, Callable c)
		{
			const ModVariable* eq = &var;
			auto [begin, end] = variableLookUp.equal_range(eq);
			while (begin != end)
			{
				c(edges[begin->second]);
				begin++;
			}
		}
		template<typename Callable>
		void forAllConnected(const ModVariable& var, Callable c) const
		{
			const ModVariable* eq = &var;
			auto [begin, end] = variableLookUp.equal_range(eq);
			while (begin != end)
			{
				c(edges[begin->second]);
				begin++;
			}
		}

		[[nodiscard]] FlowCandidates selectStartingEdge();
		[[nodiscard]] llvm::SmallVector<FlowCandidates, 2> findAugmentingPath();

		[[nodiscard]] IndexSet getUnmatchedSet(const ModVariable& variable) const
		{
			auto set = variable.toIndexSet();
			set.remove(getMatchedSet(variable));
			return set;
		}

		[[nodiscard]] IndexSet getUnmatchedSet(const ModEquation& equation) const
		{
			IndexSet matched;
			const auto unite = [&matched](const Edge& edge) {
				matched.unite(edge.getSet());
			};
			forAllConnected(equation, unite);

			auto set = equation.toIndexSet();
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

		[[nodiscard]] IndexSet getMatchedSet(const ModEquation& eq) const
		{
			IndexSet matched;
			const auto unite = [&matched](const auto& edge) {
				matched.unite(edge.getSet());
			};
			forAllConnected(eq, unite);

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

		[[nodiscard]] size_t matchedEdgesCount() const;
		void dump(llvm::raw_ostream& OS = llvm::outs()) const;

		private:
		void addEquation(const ModEquation& eq);

		llvm::SmallVector<Edge, 0> edges;
		std::multimap<const ModEquation*, size_t> equationLookUp;
		std::multimap<const ModVariable*, size_t> variableLookUp;
		const Model& model;
	};

}	 // namespace modelica
