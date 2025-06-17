#ifndef MARCO_MODELING_SCCSDEPENDENCYGRAPH_H
#define MARCO_MODELING_SCCSDEPENDENCYGRAPH_H

#include "marco/Modeling/SCC.h"
#include "marco/Modeling/SingleEntryWeaklyConnectedDigraph.h"
#include "llvm/ADT/PostOrderIterator.h"
#include <set>
#include <vector>

namespace marco::modeling {
/// Graph storing the dependencies between SCCs.
/// An edge from SCC A to SCC B is created if the computations inside A depends
/// on B, meaning that B needs to be computed first. The order of computations
/// is therefore given by a post-order visit of the graph.
template <typename SCCProperty>
class SCCsDependencyGraph {
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

  SCCProperty &operator[](SCCDescriptor descriptor) {
    return graph[descriptor];
  }

  const SCCProperty &operator[](SCCDescriptor descriptor) const {
    return graph[descriptor];
  }

  /// }

  void addSCCs(llvm::ArrayRef<SCCProperty> SCCs) {
    // Internalize the SCCs.
    llvm::SmallVector<SCCDescriptor> sccDescriptors;

    for (const auto &scc : SCCs) {
      SCCDescriptor sccDescriptor = graph.addVertex(scc);
      sccDescriptors.push_back(sccDescriptor);

      for (const auto &element :
           SCCTraits::getElements(&graph[sccDescriptor])) {
        parentSCC[element] = sccDescriptor;
      }
    }

    // Connect the SCCs.
    for (SCCDescriptor sccDescriptor : sccDescriptors) {
      const SCCProperty &scc = graph[sccDescriptor];

      // The set of SCCs that have already been connected to the current
      // SCC. This allows to avoid duplicated edges.
      llvm::DenseSet<SCCDescriptor> connectedSCCs;

      for (const auto &source : SCCTraits::getElements(&scc)) {
        for (const auto &destination :
             SCCTraits::getDependencies(&scc, source)) {
          SCCDescriptor destinationSCC = parentSCC.find(destination)->second;

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
  std::vector<SCCDescriptor> postOrder() const {
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
  std::vector<SCCDescriptor> reversePostOrder() const {
    auto result = postOrder();
    std::reverse(result.begin(), result.end());
    return result;
  }

  /// @name Forwarded methods
  /// {

  auto SCCsBegin() const { return graph.verticesBegin(); }

  auto SCCsEnd() const { return graph.verticesEnd(); }

  /// }
};
} // namespace marco::modeling

#endif // MARCO_MODELING_SCCSDEPENDENCYGRAPH_H
