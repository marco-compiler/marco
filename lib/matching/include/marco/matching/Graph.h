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
#include "MCIM.h"
#include "MCIS.h"
#include "Range.h"

namespace marco::matching
{
	namespace detail
	{
		template<class VariableDescriptor>
		class VariableVertex
		{
			public:
			using Id = typename VariableDescriptor::Id;

			VariableVertex(VariableDescriptor descriptor)
					: descriptor(std::move(descriptor))
			{
				assert(getRank() > 0 && "Scalar variables are not supported");
			}

			VariableVertex::Id getId() const
			{
				return descriptor.getId();
			}

			unsigned int getRank() const
			{
				return descriptor.getRank();
			}

			long getDimensionSize(size_t index) const
			{
				assert(index < getRank());
				return descriptor.getDimensionSize(index);
			}

			/*
			RangeSet getRanges() const
			{
				llvm::SmallVector<Range, 3> ranges;

				for (size_t i = 0; i < getRank(); ++i)
				{
					long size = getDimensionSize(i);
					assert(size > 0);
					ranges.emplace_back(0, size);
				}

				return RangeSet(ranges);
			}
			 */

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
			VariableDescriptor descriptor;
		};
	}

	template<typename VariableDescriptor>
	class Access
	{
		public:
		Access(VariableDescriptor variable, AccessFunction accessFunction)
				: variable(std::move(variable)),
					accessFunction(std::move(accessFunction))
		{
		}

		template<typename... T>
		Access(VariableDescriptor variable, T&&... accesses)
				: variable(std::move(variable)),
					accessFunction(llvm::ArrayRef<SingleDimensionAccess>({ std::forward<T>(accesses)... }))
		{
		}

		VariableDescriptor getVariable() const
		{
			return variable;
		}

		AccessFunction getAccessFunction() const
		{
			return accessFunction;
		}

		private:
		VariableDescriptor variable;
		AccessFunction accessFunction;
	};

	namespace detail
	{
		template<class EquationDescriptor, class VariableDescriptor>
		class EquationVertex
		{
			public:
			using Id = typename EquationDescriptor::Id;

			EquationVertex(EquationDescriptor descriptor)
					: descriptor(std::move(descriptor))
			{
			}

			Id getId() const
			{
				return descriptor.getId();
			}

			unsigned int getNumOfIterationVars() const
			{
				return descriptor.getNumOfIterationVars();
			}

			Range getIterationRange(size_t index) const
			{
				assert(index < getNumOfIterationVars());
				return descriptor.getIterationRange(index);
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

			void getVariableAccesses(llvm::SmallVectorImpl<Access<VariableDescriptor>>& accesses) const
			{
				descriptor.getVariableAccesses(accesses);
			}

			private:
			EquationDescriptor descriptor;
		};

		class EdgeProperty
		{
			public:
			EdgeProperty(unsigned int equations, unsigned int variables, llvm::ArrayRef<AccessFunction> accessFunctions = llvm::None)
					: equations(equations),
						variables(variables),
						accessFunctions(accessFunctions.begin(), accessFunctions.end()),
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
			//MCIM incidenceMatrix;
			//MCIM matchingMatrix;
			bool visible;
		};
	}

	template<
			class VariableDescriptor,
			class EquationDescriptor>
	class MatchingGraph
	{
		public:
		using Variable = detail::VariableVertex<VariableDescriptor>;
		using Equation = detail::EquationVertex<EquationDescriptor, VariableDescriptor>;

		private:
		using Graph = boost::adjacency_list<
				boost::setS,        // OutEdgeList = set (no multigraph)
				boost::listS,       // VertexList  = list (efficient node removal)
				boost::undirectedS, // Graph is undirected
				std::variant<Variable, Equation>,
				detail::EdgeProperty>;

		using Vertex = typename boost::graph_traits<Graph>::vertex_descriptor;
		using VertexIterator = typename boost::graph_traits<Graph>::vertex_iterator;
		using Edge = typename boost::graph_traits<Graph>::edge_descriptor;
		using EdgeIterator = typename boost::graph_traits<Graph>::edge_iterator;

		public:
		bool hasVariable(typename Variable::Id id) const
		{
			return hasVertex<Variable>(id);
		}

		Variable getVariable(typename Variable::Id id) const
		{
			auto vertex = getVariableVertex(id);
			return std::get<Variable>(graph[vertex]);
		}

		void addVariable(VariableDescriptor descriptor)
		{
			Variable variable(descriptor);
			assert(!hasVariable(variable.getId()) && "Already existing variable");
			boost::add_vertex(std::move(variable), graph);
		}

		bool hasEquation(typename Equation::Id id) const
		{
			return hasVertex<Equation>(id);
		}

		Equation getEquation(typename Equation::Id id) const
		{
			auto vertex = getEquationVertex(id);
			return std::get<Equation>(graph[vertex]);
		}

		void addEquation(EquationDescriptor descriptor)
		{
			Equation equation(descriptor);
			assert(!hasEquation(equation.getId()) && "Already existing equation");
			auto equationVertex = boost::add_vertex(equation, graph);

			llvm::SmallVector<Access<VariableDescriptor>, 3> accesses;
			equation.getVariableAccesses(accesses);

			for (const auto& access : accesses)
			{
				auto variableVertex = getVariableVertex(access.getVariable().getId());
				unsigned int numberOfEquations = equation.getIterationRanges().flatSize();
				unsigned int numberOfVariables = std::get<Variable>(graph[variableVertex]).flatSize();

				detail::EdgeProperty edge(numberOfEquations, numberOfVariables, access.getAccessFunction());
				boost::add_edge(equationVertex, variableVertex, edge, graph);
			}
		}

		bool hasEdge(typename EquationDescriptor::Id equationId, typename VariableDescriptor::Id variableId) const
		{
			return findEdge<Equation, Variable>(equationId, variableId).first;
		}

		unsigned int getNumberOfScalarEquations() const
		{
			unsigned int result = 0;

			for (Vertex v : boost::make_iterator_range(boost::vertices(graph)))
			{
				if (auto& vertex = graph[v]; std::holds_alternative<Equation>(vertex))
				{
					auto& equation = std::get<Equation>(vertex);
					result += equation.flatSize();
				}
			}

			return result;
		}

		unsigned int getNumberOfScalarVariables() const
		{
			unsigned int result = 0;

			for (Vertex v : boost::make_iterator_range(boost::vertices(graph)))
			{
				if (auto& vertex = graph[v]; std::holds_alternative<Variable>(vertex))
				{
					auto& variable = std::get<Variable>(vertex);
					result += variable.flatSize();
				}
			}

			return result;
		}

		bool simplify()
		{
			// Vertices that are candidate for the first simplification phase.
			// They are the ones having only one incident edge.
			std::list<Vertex> candidates;

			for (Vertex& vertex : boost::make_iterator_range(boost::vertices(graph)))
			{
				auto incidentEdges = getVertexVisibilityDegree(vertex);

				if (incidentEdges == 0)
					return false;

				if (incidentEdges == 1)
					candidates.push_back(vertex);
			}

			while (!candidates.empty())
			{
				Vertex v1 = candidates.front();
				candidates.pop_front();

				bool shouldRemoveV2 = false;
				Edge edge = getFirstOutEdge(v1);

			}

			return true;
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

			auto it = std::find_if(begin, end, [&](const Vertex& v) {
				const auto& vertex = graph[v];

				if (!std::holds_alternative<T>(vertex))
					return false;

				return std::get<T>(vertex).getId() == id;
			});

			return std::make_pair(it != end, it);
		}

		Vertex getVariableVertex(typename Variable::Id id) const
		{
			auto search = findVertex<Variable>(id);
			assert(search.first && "Variable not found");
			return *search.second;
		}

		Vertex getEquationVertex(typename Equation::Id id) const
		{
			auto search = findVertex<Equation>(id);
			assert(search.first && "Equation not found");
			return *search.second;
		}

		template<typename From, typename To>
		std::pair<bool, EdgeIterator> findEdge(typename From::Id from, typename To::Id to) const
		{
			EdgeIterator begin, end;
			std::tie(begin, end) = boost::edges(graph);

			auto it = std::find_if(begin, end, [&](const Edge& e) {
				auto& source = graph[boost::source(e, graph)];
				auto& target = graph[boost::target(e, graph)];

				if (!std::holds_alternative<From>(source) || !std::holds_alternative<To>(target))
					return false;

				return std::get<From>(source).getId() == from && std::get<To>(target).getId() == to;
			});

			return std::make_pair(it != end, it);
		}

		unsigned int getVertexVisibilityDegree(const Vertex& vertex) const
		{
			return boost::out_degree(vertex, graph);
		}

		Edge getFirstOutEdge(const Vertex& vertex) const
		{
			for (Edge edge : boost::make_iterator_range(boost::out_edges(vertex, graph)))
				return edge;

			assert(false && "Unreachable");
		}

		Graph graph;
	};
}

#endif	// MARCO_MATCHING_MATCHINGGRAPH_H
