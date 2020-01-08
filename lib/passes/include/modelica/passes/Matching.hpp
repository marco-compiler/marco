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

	[[nodiscard]] Matching match(const Model& model);

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

		[[nodiscard]] IndexSet getUnmatchedSet(const ModVariable& variable) const
		{
			auto set = variable.toIndexSet();
			set.remove(getMatchedSet(variable));
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
