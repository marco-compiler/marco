#ifndef MARCO_MODELING_SCC_H
#define MARCO_MODELING_SCC_H

#include "llvm/ADT/STLExtras.h"
#include <vector>

namespace marco::modeling
{
  namespace dependency
  {
    template<typename SCC>
    struct SCCTraits
    {
      // Elements to provide:
      //
      // typedef ElementRef : the type of the elements composing the SCC, which should be cheap to copy.
      //
      // static bool hasCycle(const SCC* scc);
      //    return whether the SCC contains any cycle.
      //
      // static std::vector<ElementRef> getElements(const SCC* scc);
      //    return the elements composing the SCC.
      //
      // static std::vector<ElementRef> getDependencies(const Impl* SCC, ElementRef element);
      //    return the dependencies of an element, which may belong to other SCCs.
    };
  }

  namespace internal::dependency
  {
    /// List of the equations composing an SCC.
    /// All the equations belong to a given graph.
    template<typename Graph>
    class SCC
    {
      public:
        using EquationDescriptor = typename Graph::VertexDescriptor;

      private:
        using Equation = typename Graph::VertexProperty;
        using Container = std::vector<EquationDescriptor>;

      public:
        using iterator = typename Container::iterator;
        using const_iterator = typename Container::const_iterator;

        template<typename It>
        SCC(const Graph& graph, bool cycle, It equationsBegin, It equationsEnd)
            : graph(&graph), cycle(cycle), equations(equationsBegin, equationsEnd)
        {
        }

        const Graph& getGraph() const
        {
          assert(graph != nullptr);
          return *graph;
        }

        /// Get whether the SCC present a cycle.
        /// Note that only SCCs with just one element may not have cycles.
        bool hasCycle() const
        {
          return cycle;
        }

        /// Get the number of equations composing the SCC.
        size_t size() const
        {
          return equations.size();
        }

        const Equation& operator[](size_t index) const
        {
          assert(index < equations.size());
          return (*this)[equations[index]];
        }

        /// @name Forwarded methods
        /// {

        const Equation& operator[](EquationDescriptor descriptor) const
        {
          return (*graph)[descriptor];
        }

        /// }
        /// @name Iterators
        /// {

        iterator begin()
        {
          return equations.begin();
        }

        const_iterator begin() const
        {
          return equations.begin();
        }

        iterator end()
        {
          return equations.end();
        }

        const_iterator end() const
        {
          return equations.end();
        }

      /// }

      private:
        const Graph* graph;
        bool cycle;
        Container equations;
    };
  }

  namespace dependency
  {
    // Traits specialization for the internal SCC class.
    template<typename Graph>
    class SCCTraits<internal::dependency::SCC<Graph>>
    {
      private:
        using Impl = internal::dependency::SCC<Graph>;

      public:
        using ElementRef = typename Impl::EquationDescriptor;

        static bool hasCycle(const Impl* SCC)
        {
          return SCC->hasCycle();
        }

        static std::vector<ElementRef> getElements(const Impl* SCC)
        {
          std::vector<ElementRef> result(SCC->begin(), SCC->end());
          return result;
        }

        static std::vector<ElementRef> getDependencies(
            const Impl* SCC, ElementRef element)
        {
          std::vector<ElementRef> result;
          const auto& graph = SCC->getGraph();

          auto edges = llvm::make_range(
              graph.outgoingEdgesBegin(element),
              graph.outgoingEdgesEnd(element));

          for (const auto& edge : edges) {
            result.push_back(edge.to);
          }

          return result;
        }
    };
  }
}

#endif // MARCO_MODELING_SCC_H
