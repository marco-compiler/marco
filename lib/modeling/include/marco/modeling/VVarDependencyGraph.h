#ifndef MARCO_MODELING_VVARDEPENDENCYGRAPH_H
#define MARCO_MODELING_VVARDEPENDENCYGRAPH_H

#include <llvm/ADT/GraphTraits.h>
#include <llvm/ADT/SCCIterator.h>

#include "AccessFunction.h"
#include "Dumpable.h"
#include "Graph.h"
#include "MultidimensionalRange.h"

namespace marco::modeling
{
  namespace internal::scc
  {
    class EmptyAccessProperty
    {
    };

    template<class T>
    struct get_access_property
    {
      template<class U, typename = typename U::AccessProperty>
      static typename U::AccessProperty property(int);

      template<class U>
      static EmptyAccessProperty property(...);

      using type = decltype(property<T>(0));
    };
  }

  namespace scc
  {
    template<typename VariableProperty, typename AccessProperty = internal::scc::EmptyAccessProperty>
    class Access
    {
      public:
      Access(VariableProperty variable, AccessProperty property = AccessProperty())
              : variable(std::move(variable)), property(std::move(property))
      {
      }

      const VariableProperty& getVariable() const
      {
        return variable;
      }

      const AccessProperty& getProperty() const
      {
        return property;
      }

      private:
      VariableProperty variable;
      AccessProperty property;
    };
  }

  namespace internal::scc
  {
    template<typename VariableProperty>
    class VariableWrapper
    {
      public:
      using Id = typename VariableProperty::Id;
      using Property = VariableProperty;

      VariableWrapper(VariableProperty property)
              : property(property)
      {
      }

      VariableWrapper::Id getId() const
      {
        return property.getId();
      }

      private:
      // Custom equation property
      VariableProperty property;
    };

    template<class EquationProperty, class VariableProperty>
    class EquationVertex
    {
      public:
      using Id = typename EquationProperty::Id;
      using Property = EquationProperty;

      using Access = marco::modeling::scc::Access<
              VariableProperty,
              typename internal::scc::get_access_property<EquationProperty>::type>;

      EquationVertex(EquationProperty property)
              : property(property)
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

      size_t getNumOfIterationVars() const
      {
        return getNumOfIterationVars(property);
      }

      Range getIterationRange(size_t index) const
      {
        return getIterationRange(property, index);
      }

      MultidimensionalRange getIterationRanges() const
      {
        return getIterationRanges(property);
      }

      Access getWrite() const
      {
        return getWrite(property);
      }

      void getReads(llvm::SmallVectorImpl<Access>& accesses) const
      {
        getVariableAccesses(property, accesses);
      }

      private:
      static size_t getNumOfIterationVars(const EquationProperty& p)
      {
        return p.getNumOfIterationVars();
      }

      static Range getIterationRange(const EquationProperty& p, size_t index)
      {
        assert(index < getNumOfIterationVars(p));
        return Range(p.getRangeStart(index), p.getRangeEnd(index));
      }

      static MultidimensionalRange getIterationRanges(const EquationProperty& p)
      {
        llvm::SmallVector<Range, 3> ranges;

        for (unsigned int i = 0, e = getNumOfIterationVars(p); i < e; ++i)
          ranges.push_back(getIterationRange(p, i));

        return MultidimensionalRange(ranges);
      }

      static void getVariableAccesses(
              const EquationProperty& p,
              llvm::SmallVectorImpl<Access>& accesses)
      {
        p.getReads(accesses);
      }

      static Access getWrite(const EquationProperty& p)
      {
        return p.getWrite();
      }

      // Custom equation property
      EquationProperty property;
    };

    class Edge
    {
    };
  }

  template<typename VariableProperty, typename EquationProperty>
  class VVarDependencyGraph
  {
    public:
    using Equation = internal::scc::EquationVertex<EquationProperty, VariableProperty>;
    using Edge = internal::scc::Edge;

    using Graph = internal::DirectedGraph<Equation, Edge>;
    using EquationDescriptor = typename Graph::VertexDescriptor;

    VVarDependencyGraph(llvm::ArrayRef<EquationProperty> equations)
    {
      Graph graph;

      for (const auto& equationProperty : equations)
      {
        Equation equation(equationProperty);
        auto equationDescriptor = graph.addVertex(std::move(equation));
        const auto& write = graph[equationDescriptor].getWrite();
        writes.emplace(write.getVariable().getId(), equationDescriptor);
      }

      for (const auto& equationDescriptor : graph.getVertices())
      {
        const Equation& equation = graph[equationDescriptor];

        llvm::SmallVector<scc::Access<VariableProperty>> reads;
        equation.getReads(reads);

        for (auto& read : reads)
        {
          auto writingEquations = writes.equal_range(read.getVariable().getId());
          assert(writingEquations.first != writingEquations.second);

          for (auto& writingEquation : llvm::make_range(writingEquations.first, writingEquations.second))
            graph.addEdge(equationDescriptor, writingEquation.second, Edge());
        }
      }

      auto subGraphs = graph.getConnectedComponents();
      connectedGraphs.insert(connectedGraphs.begin(), subGraphs.begin(), subGraphs.end());
    }

    void findSCCs()
    {
      for (const auto& graph : connectedGraphs)
      {
        for (auto SCC : llvm::make_range(llvm::scc_begin(graph), llvm::scc_end(graph)))
          std::cout << SCC.size() << "\n";

        for (auto it = llvm::scc_begin(graph); it != llvm::scc_end(graph); ++it)
        {
          std::cout << "Found SCC " << it.hasCycle() << " \n";

          for (const auto& node : *it)
          {
            std::cout << "ID: " << graph[node].getId() << "\n";
          }
        }
      }
    }

    private:
    std::vector<Graph> connectedGraphs;
    std::multimap<typename VariableProperty::Id, EquationDescriptor> writes;
  };
}

#endif // MARCO_MODELING_VVARDEPENDENCYGRAPH_H
