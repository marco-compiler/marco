#ifndef MARCO_MATCHING_MATCHINGGRAPH_H
#define MARCO_MATCHING_MATCHINGGRAPH_H

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <type_traits>
#include <variant>

#include "AccessFunction.h"
#include "IndexSet.h"

namespace marco::matching
{
	using Matrix = boost::numeric::ublas::matrix<int>;

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

			void getVariableAccesses(llvm::SmallVectorImpl<Access<VariableDescriptor>>& accesses) const
			{
				descriptor.getVariableAccesses(accesses);
			}

			private:
			EquationDescriptor descriptor;
		};
	}

	class EdgeProperty
	{
		public:


		private:
		// Amount of equations
		unsigned int equations;

		// Amount of variables
		unsigned int variables;

		// Vector access functions
		llvm::SmallVector<AccessFunction, 3> vafc;

		Matrix u;
		Matrix m;

		bool hidden;
	};

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
				EdgeProperty>;

		using Vertex = typename boost::graph_traits<Graph>::vertex_descriptor;
		using VertexIterator = typename boost::graph_traits<Graph>::vertex_iterator;
		//using Edge = typename boost::graph_traits<Graph>::edge_descriptor;

		public:
		void addVariable(VariableDescriptor descriptor)
		{
			boost::add_vertex(Variable(descriptor), graph);
		}

		void addEquation(EquationDescriptor descriptor)
		{
			Equation equation(descriptor);
			auto equationVertex = boost::add_vertex(equation, graph);

			llvm::SmallVector<Access<VariableDescriptor>, 3> accesses;
			equation.getVariableAccesses(accesses);

			for (const auto& access : accesses)
			{
				auto var = getVariable(access.getVariable().getId());
				boost::add_edge(equationVertex, var, EdgeProperty(), graph);
			}
		}

		bool hasVariable(typename Variable::Id id) const
		{
			return hasVertex<Variable>(id);
		}

		Variable getVariable(typename Variable::Id id) const
		{
			assert(hasVertex<Variable>(id));
			auto& vertex = *getVertex<Variable>(id);
			return std::get<Variable>(graph[vertex]);
		}

		private:
		template<typename T>
		bool hasVertex(typename T::Id id) const
		{
			VertexIterator begin, end;
			std::tie(begin, end) = boost::vertices(graph);
			auto it = getVertex<T>(id);
			return it != end;
		}

		template<typename T>
		VertexIterator getVertex(typename T::Id id) const
		{
			VertexIterator begin, end;
			std::tie(begin, end) = boost::vertices(graph);

			return std::find_if(begin, end, [&](const Vertex& v) {
				const auto& vertex = graph[v];

				if (!std::holds_alternative<T>(vertex))
					return false;

				return std::get<T>(vertex).getId() == id;
			});
		}

		Graph graph;
	};
}

#endif	// MARCO_MATCHING_MATCHINGGRAPH_H
