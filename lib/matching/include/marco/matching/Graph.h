#ifndef MARCO_MATCHING_MATCHINGGRAPH_H
#define MARCO_MATCHING_MATCHINGGRAPH_H

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <type_traits>
#include <variant>

#include "AccessFunction.h"
#include "IncidenceMatrix.h"
#include "MCIM.h"
#include "MCIS.h"
#include "Range.h"

namespace marco::matching
{
	namespace detail
	{
		template<typename V>
		struct VertexDescriptor
		{
			V* data;
		};

		template<typename V, typename E>
		class EdgeDescriptor
		{
			E* data;
			VertexDescriptor<V> first;
			VertexDescriptor<V> second;
		};

		template<class VariableProperty>
		class VariableVertex
		{
			public:
			using Id = typename VariableProperty::Id;

			VariableVertex(VariableProperty property)
					: property(std::move(property))
			{
				assert(getRank() > 0 && "Scalar variables are not supported");
			}

			VariableProperty& getProperty()
			{
				return property;
			}

			const VariableProperty& getProperty() const
			{
				return property;
			}

			VariableVertex::Id getId() const
			{
				return property.getId();
			}

			unsigned int getRank() const
			{
				return property.getRank();
			}

			long getDimensionSize(size_t index) const
			{
				assert(index < getRank());
				return property.getDimensionSize(index);
			}

			MultidimensionalRange getRanges() const
			{
				llvm::SmallVector<Range, 3> ranges;

				for (size_t i = 0; i < getRank(); ++i)
				{
					long size = getDimensionSize(i);
					assert(size > 0);
					ranges.emplace_back(0, size);
				}

				return MultidimensionalRange(ranges);
			}

			unsigned int flatSize() const
			{
				unsigned int result = 1;

				for (unsigned int i = 0, rank = getRank(); i < rank; ++i)
				{
					long size = getDimensionSize(i);
					assert(size > 0);
					result *= size;
				}

				return result;
			}

			private:
			VariableProperty property;
		};
	}

	template<typename VariableProperty>
	class Access
	{
		public:
		Access(VariableProperty variable, AccessFunction accessFunction)
				: variable(std::move(variable)),
					accessFunction(std::move(accessFunction))
		{
		}

		template<typename... T>
		Access(VariableProperty variable, T&&... accesses)
				: variable(std::move(variable)),
					accessFunction(llvm::ArrayRef<DimensionAccess>({ std::forward<T>(accesses)... }))
		{
		}

		VariableProperty getVariable() const
		{
			return variable;
		}

		AccessFunction getAccessFunction() const
		{
			return accessFunction;
		}

		private:
		VariableProperty variable;
		AccessFunction accessFunction;
	};

	namespace detail
	{
		template<class EquationProperty, class VariableProperty>
		class EquationVertex
		{
			public:
			using Id = typename EquationProperty::Id;

			EquationVertex(EquationProperty property)
					: property(std::move(property))
			{
			}

			EquationProperty& getProperty()
			{
				return property;
			}

			const EquationProperty& getProperty() const
			{
				return property;
			}

			Id getId() const
			{
				return property.getId();
			}

			unsigned int getNumOfIterationVars() const
			{
				return property.getNumOfIterationVars();
			}

			Range getIterationRange(size_t index) const
			{
				assert(index < getNumOfIterationVars());
				return property.getIterationRange(index);
			}

			MultidimensionalRange getIterationRanges() const
			{
				llvm::SmallVector<Range, 3> ranges;

				for (unsigned int i = 0, e = getNumOfIterationVars(); i < e; ++i)
					ranges.push_back(getIterationRange(i));

				return MultidimensionalRange(ranges);
			}

			unsigned int flatSize() const
			{
				return getIterationRanges().flatSize();
			}

			void getVariableAccesses(llvm::SmallVectorImpl<Access<VariableProperty>>& accesses) const
			{
				property.getVariableAccesses(accesses);
			}

			private:
			EquationProperty property;
		};

		class Edge
		{
			public:
			Edge(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
					: equations(equationRanges.flatSize()),
						variables(variableRanges.flatSize()),
						incidenceMatrix(equationRanges, variableRanges),
						matchMatrix(equationRanges, variableRanges),
						visible(true)
			{
			}

			unsigned int getNumberOfEquations() const
			{
				return equations;
			}

			unsigned int getNumberOfVariables() const
			{
				return variables;
			}

			llvm::ArrayRef<AccessFunction> getAccessFunctions() const
			{
				return accessFunctions;
			}

			void addAccessFunction(AccessFunction accessFunction)
			{
				accessFunctions.push_back(accessFunction);
				incidenceMatrix.apply(accessFunction);
			}

			const IncidenceMatrix& getIncidenceMatrix() const
			{
				return incidenceMatrix;
			}

			void addMatch(IncidenceMatrix match)
			{
				matchMatrix += match;
			}

			bool isVisible() const
			{
				return visible;
			}

			void setVisibility(bool visibility)
			{
				visible = visibility;
			}

			private:
			unsigned int equations;
			unsigned int variables;
			llvm::SmallVector<AccessFunction, 3> accessFunctions;
			IncidenceMatrix incidenceMatrix;
			IncidenceMatrix matchMatrix;
			bool visible;
		};
	}

	template<
			class VariableProperty,
			class EquationProperty>
	class MatchingGraph
	{
		public:
		using Variable = detail::VariableVertex<VariableProperty>;
		using Equation = detail::EquationVertex<EquationProperty, VariableProperty>;
		using Vertex = std::variant<Variable, Equation>;
		using Edge = detail::Edge;

		private:
		using Graph = boost::adjacency_list<
				boost::setS,        // OutEdgeList = set (no multigraph)
				boost::listS,       // VertexList  = list (efficient node removal)
				boost::undirectedS, // Graph is undirected
				Vertex,
				Edge>;

		public:
		using VertexDescriptor = typename boost::graph_traits<Graph>::vertex_descriptor;
		using EdgeDescriptor = typename boost::graph_traits<Graph>::edge_descriptor;

		private:
		using VertexIterator = typename boost::graph_traits<Graph>::vertex_iterator;
		using EdgeIterator = typename boost::graph_traits<Graph>::edge_iterator;

		public:
		bool hasVariable(typename Variable::Id id) const
		{
			return hasVertex<Variable>(id);
		}

		VertexDescriptor getVariableVertex(typename Variable::Id id) const
		{
			auto search = findVertex<Variable>(id);
			assert(search.first && "Variable not found");
			return *search.second;
		}

		VariableProperty& getVariable(typename Variable::Id id)
		{
			auto vertex = getVariableVertex(id);
			return std::get<Variable>(graph[vertex]).getProperty();
		}

		const VariableProperty& getVariable(typename Variable::Id id) const
		{
			auto vertex = getVariableVertex(id);
			return std::get<Variable>(graph[vertex]).getProperty();
		}

		void addVariable(VariableProperty property)
		{
			Variable variable(std::move(property));
			assert(!hasVariable(variable.getId()) && "Already existing variable");
			boost::add_vertex(std::move(variable), graph);
		}

		bool hasEquation(typename Equation::Id id) const
		{
			return hasVertex<Equation>(id);
		}

		VertexDescriptor getEquationVertex(typename Equation::Id id) const
		{
			auto search = findVertex<Equation>(id);
			assert(search.first && "Equation not found");
			return *search.second;
		}

		EquationProperty& getEquation(typename Equation::Id id)
		{
			auto vertex = getEquationVertex(id);
			return std::get<Equation>(graph[vertex]).getProperty();
		}

		const EquationProperty& getEquation(typename Equation::Id id) const
		{
			auto vertex = getEquationVertex(id);
			return std::get<Equation>(graph[vertex]).getProperty();
		}

		void addEquation(EquationProperty property)
		{
			Equation equation(std::move(property));
			assert(!hasEquation(equation.getId()) && "Already existing equation");
			auto equationVertex = boost::add_vertex(equation, graph);

			llvm::SmallVector<Access<VariableProperty>, 3> accesses;
			equation.getVariableAccesses(accesses);

			// The equation may access multiple variables or even multiple indexes
			// of the same variable. Add an edge to the graph for each of those
			// accesses.

			for (const auto& access : accesses)
			{
				auto variableVertex = getVariableVertex(access.getVariable().getId());
				unsigned int numberOfEquations = equation.getIterationRanges().flatSize();
				auto& variable = std::get<Variable>(graph[variableVertex]);
				unsigned int numberOfVariables = variable.flatSize();

				Edge edge(equation.getIterationRanges(), variable.getRanges());
				auto edgeDescriptor = boost::add_edge(equationVertex, variableVertex, edge, graph);

				graph[edgeDescriptor.first].addAccessFunction(access.getAccessFunction());
			}
		}

		size_t getNumberOfScalarVariables() const
		{
			size_t result = 0;

			for (const auto& v : boost::make_iterator_range(boost::vertices(graph)))
			{
				if (auto& vertex = graph[v]; std::holds_alternative<Variable>(vertex))
				{
					auto& variable = std::get<Variable>(vertex);
					result += variable.flatSize();
				}
			}

			return result;
		}

		size_t getNumberOfScalarEquations() const
		{
			size_t result = 0;

			for (const auto& v : boost::make_iterator_range(boost::vertices(graph)))
			{
				if (auto& vertex = graph[v]; std::holds_alternative<Equation>(vertex))
				{
					auto& equation = std::get<Equation>(vertex);
					result += equation.flatSize();
				}
			}

			return result;
		}

		auto getVertices()
		{
			return boost::make_iterator_range(boost::vertices(graph));
		}

		unsigned int getVertexVisibilityDegree(VertexDescriptor vertex) const
		{
			return boost::out_degree(vertex, graph);
		}

		bool hasEdge(typename EquationProperty::Id equationId, typename VariableProperty::Id variableId) const
		{
			return findEdge<Equation, Variable>(equationId, variableId).first;
		}

		Edge& getEdge(EdgeDescriptor edge)
		{
			return graph[edge];
		}

		const Edge& getEdge(EdgeDescriptor edge) const
		{
			return graph[edge];
		}

		EdgeDescriptor getFirstOutEdge(VertexDescriptor vertex) const
		{
			auto edges = boost::out_edges(vertex, graph);
			assert(edges.first != edges.second && "Vertex doesn't belong to any edge");
			return *edges.first;
		}

		std::pair<VertexDescriptor, VertexDescriptor> getEdgeVertices(EdgeDescriptor edge)
		{
			return std::make_pair(boost::source(edge, graph), boost::target(edge, graph));
		}

		private:
		template<typename T>
		bool hasVertex(typename T::Id id) const
		{
			return findVertex<T>(id).first;
		}

		template<typename T>
		std::pair<bool, VertexIterator> findVertex(typename T::Id id) const
		{
			VertexIterator begin, end;
			std::tie(begin, end) = boost::vertices(graph);

			auto it = std::find_if(begin, end, [&](const VertexDescriptor& v) {
				const auto& vertex = graph[v];

				if (!std::holds_alternative<T>(vertex))
					return false;

				return std::get<T>(vertex).getId() == id;
			});

			return std::make_pair(it != end, it);
		}

		template<typename From, typename To>
		std::pair<bool, EdgeIterator> findEdge(typename From::Id from, typename To::Id to) const
		{
			EdgeIterator begin, end;
			std::tie(begin, end) = boost::edges(graph);

			auto it = std::find_if(begin, end, [&](const EdgeDescriptor& e) {
				auto& source = graph[boost::source(e, graph)];
				auto& target = graph[boost::target(e, graph)];

				if (!std::holds_alternative<From>(source) || !std::holds_alternative<To>(target))
					return false;

				return std::get<From>(source).getId() == from && std::get<To>(target).getId() == to;
			});

			return std::make_pair(it != end, it);
		}

		Graph graph;
	};
}

#endif	// MARCO_MATCHING_MATCHINGGRAPH_H
