#ifndef MARCO_MODELING_SCCSDEPENDENCYGRAPH_H
#define MARCO_MODELING_SCCSDEPENDENCYGRAPH_H

#include "marco/Modeling/SCC.h"
#include "marco/Modeling/SingleEntryWeaklyConnectedDigraph.h"
#include "llvm/ADT/PostOrderIterator.h"
#include <set>
#include <vector>

namespace marco::modeling
{
  template<typename SCCProperty>
  class SCCsDependencyGraph
  {
    public:
      using Graph =
          internal::dependency::SingleEntryWeaklyConnectedDigraph<SCCProperty>;

      using SCCDescriptor = typename Graph::VertexDescriptor;

      using SCCTraits =
          typename ::marco::modeling::dependency::SCCTraits<SCCProperty>;

      using ElementRef = typename SCCTraits::ElementRef;

    private:
      // Keep track of the parent-children relationships.
      llvm::DenseMap<ElementRef, SCCDescriptor> parentSCC;

      Graph graph;

    public:
      /// @name Forwarded methods
      /// {

      SCCProperty& operator[](SCCDescriptor descriptor)
      {
        return graph[descriptor];
      }

      const SCCProperty& operator[](SCCDescriptor descriptor) const
      {
        return graph[descriptor];
      }

      /// }

      void addSCCs(llvm::ArrayRef<SCCProperty> SCCs)
      {
        // Internalize the SCCs.
        llvm::SmallVector<SCCDescriptor> sccDescriptors;

        for (const auto& scc : SCCs) {
          SCCDescriptor sccDescriptor = graph.addVertex(scc);
          sccDescriptors.push_back(sccDescriptor);

          for (const auto& element :
               SCCTraits::getElements(&graph[sccDescriptor])) {
            parentSCC[element] = sccDescriptor;
          }
        }

        // Connect the SCCs.
        for (SCCDescriptor sccDescriptor : sccDescriptors) {
          const SCCProperty& scc = graph[sccDescriptor];

          // The set of SCCs that have already been connected to the current
          // SCC. This allows to avoid duplicated edges.
          llvm::DenseSet<SCCDescriptor> connectedSCCs;

          for (const auto& source : SCCTraits::getElements(&scc)) {
            for (const auto& destination :
                 SCCTraits::getDependencies(&scc, source)) {
              SCCDescriptor sourceSCC = parentSCC.find(destination)->second;

              if (!connectedSCCs.contains(sourceSCC)) {
                graph.addEdge(sourceSCC, sccDescriptor);
                connectedSCCs.insert(sourceSCC);
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

      /*
      std::vector<std::vector<SCCDescriptor>> getPaths() const
      {
        std::vector<std::vector<SCCDescriptor>> result;
        std::stack<std::vector<SCCDescriptor>> paths;

        SCCDescriptor entryNode = graph.getEntryNode();

        for (SCCDescriptor root : llvm::make_range(
                 graph.linkedVerticesBegin(entryNode),
                 graph.linkedVerticesEnd(entryNode))) {
          std::vector<SCCDescriptor> newPath;
          newPath.push_back(root);
          paths.push(std::move(newPath));
        }

        while (!paths.empty()) {
          llvm::SmallVector<std::vector<SCCDescriptor>> newPaths;
          auto& top = paths.top();

          for (SCCDescriptor child : llvm::make_range(
                   graph.linkedVerticesBegin(top.back()),
                   graph.linkedVerticesEnd(top.back()))) {
            auto newPath = newPaths.emplace_back(top);
            newPath.push_back(child);
          }

          if (newPaths.empty()) {
            result.push_back(std::move(top));
            paths.pop();
          } else {
            paths.pop();

            for (auto& newPath : newPaths) {
              paths.push(std::move(newPath));
            }
          }
        }

        return result;
      }
       */
  };
}

#endif // MARCO_MODELING_SCCSDEPENDENCYGRAPH_H
