#ifndef MARCO_MODELING_SCCSDEPENDENCYGRAPH_H
#define MARCO_MODELING_SCCSDEPENDENCYGRAPH_H

#include "marco/Modeling/SCC.h"
#include "marco/Modeling/SingleEntryWeaklyConnectedDigraph.h"
#include "llvm/ADT/PostOrderIterator.h"
#include <set>
#include <vector>

namespace marco::modeling
{
  template<typename SCC>
  class SCCsDependencyGraph
  {
    public:
      using Graph =
          internal::dependency::SingleEntryWeaklyConnectedDigraph<SCC>;

      using SCCDescriptor = typename Graph::VertexDescriptor;
      using SCCTraits = typename ::marco::modeling::dependency::SCCTraits<SCC>;
      using ElementRef = typename SCCTraits::ElementRef;

      /// @name Forwarded methods
      /// {

      SCC& operator[](SCCDescriptor descriptor)
      {
        return graph[descriptor];
      }

      const SCC& operator[](SCCDescriptor descriptor) const
      {
        return graph[descriptor];
      }

      /// }

      void addSCCs(llvm::ArrayRef<SCC> SCCs)
      {
        // Internalize the SCCs and keep track of the parent-children
        // relationships.
        llvm::DenseMap<ElementRef, SCCDescriptor> parentSCC;

        for (const auto& scc : SCCs) {
          SCCDescriptor sccDescriptor = graph.addVertex(scc);

          for (const auto& element :
               SCCTraits::getElements(&graph[sccDescriptor])) {
            parentSCC[element] = sccDescriptor;
          }
        }

        // Connect the SCCs.
        for (const auto& sccDescriptor : llvm::make_range(
                 graph.verticesBegin(), graph.verticesEnd())) {
          const SCC& scc = graph[sccDescriptor];

          // The set of SCCs that have already been connected to the current
          // SCC. This allows to avoid duplicated edges.
          llvm::DenseSet<SCCDescriptor> connectedSCCs;

          for (const auto& source : SCCTraits::getElements(&scc)) {
            for (const auto& destination :
                 SCCTraits::getDependentElements(&scc, source)) {
              SCCDescriptor destinationSCC =
                  parentSCC.find(destination)->second;

              if (!connectedSCCs.contains(destinationSCC)) {
                graph.addEdge(sccDescriptor, destinationSCC);
                connectedSCCs.insert(destinationSCC);
              }
            }
          }
        }
      }

      /// Perform a post-order visit of the dependency graph and get the
      /// ordered SCC descriptors.
      std::vector<SCCDescriptor> postOrder() const
      {
        std::vector<SCCDescriptor> result;
        std::set<SCCDescriptor> set;

        for (SCCDescriptor scc : llvm::post_order_ext(&graph, set)) {
          // Ignore the entry node.
          if (scc != graph.getEntryNode()) {
            result.push_back(scc);
          }
        }

        return result;
      }

      /// Perform a reverse post-order visit of the dependency graph
      /// and get the ordered SCC descriptors.
      std::vector<SCCDescriptor> reversePostOrder() const
      {
        auto result = postOrder();
        std::reverse(result.begin(), result.end());
        return result;
      }

      /// @name Forwarded methods
      /// {

      auto SCCsBegin() const
      {
        return graph.verticesBegin();
      }

      auto SCCsEnd() const
      {
        return graph.verticesEnd();
      }

      auto dependentSCCsBegin(SCCDescriptor scc) const
      {
        return graph.linkedVerticesBegin(std::move(scc));
      }

      auto dependentSCCsEnd(SCCDescriptor scc) const
      {
        return graph.linkedVerticesEnd(std::move(scc));
      }

      /// }

    private:
      Graph graph;
  };
}

#endif // MARCO_MODELING_SCCSDEPENDENCYGRAPH_H
