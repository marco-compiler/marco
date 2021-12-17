#ifndef MARCO_MATCHING_VVARDEPENDENCYGRAPH_H
#define MARCO_MATCHING_VVARDEPENDENCYGRAPH_H

#include <llvm/ADT/SCCIterator.h>

#include "AccessFunction.h"
#include "Dumpable.h"
#include "Graph.h"
#include "MultidimensionalRange.h"

namespace marco::scc
{
  using Range = matching::Range;
  using MultidimensionalRange = matching::MultidimensionalRange;

  namespace detail
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

  template<typename VariableProperty, typename AccessProperty = detail::EmptyAccessProperty>
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

  namespace detail
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

      using Access = marco::scc::Access<
              VariableProperty,
              typename detail::get_access_property<EquationProperty>::type>;

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
    using Equation = detail::EquationVertex<EquationProperty, VariableProperty>;
    using Edge = detail::Edge;

    using Graph = matching::detail::DirectedGraph<Equation, Edge>;
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

        llvm::SmallVector<Access<VariableProperty>> reads;
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

    /*
    ~VVarDependencyGraph()
    {
      for (auto* variable : variables)
        delete variable;
    }

    Variable* getVariable(typename Variable::Id id)
    {
      return *llvm::find_if(variables, [id](auto* variable) {
        return variable->getId() == id;
      });
    }
     */

    /*
    bool hasVariable(const VariableProperty& variable)
    {
      auto requestedId = variable.getId();

      auto it = llvm::find_if(variablesId, [&](const typename VariableProperty::Id& id) {
        return id == requestedId;
      });

      return it != variablesId.end();
    }
     */

    void findSCCs()
    {
      for (const auto& graph : connectedGraphs)
      for (auto it = llvm::scc_begin(graph); it != llvm::scc_end(graph); ++it)
      {
        std::cout << "Found\n";
      }
    }

    private:
    std::vector<Graph> connectedGraphs;
    //llvm::SmallVector<Variable*, 3> variables;
    std::multimap<typename VariableProperty::Id, EquationDescriptor> writes;
  };
}

#endif // MARCO_MATCHING_VVARDEPENDENCYGRAPH_H
