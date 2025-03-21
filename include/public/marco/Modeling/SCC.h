#ifndef MARCO_MODELING_SCC_H
#define MARCO_MODELING_SCC_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <vector>

namespace marco::modeling {
namespace dependency {
template <typename SCC>
struct SCCTraits {
  // Elements to provide:
  //
  // typedef ElementRef : the type of the elements composing the SCC, which
  // should be cheap to copy.
  //
  // static std::vector<ElementRef> getElements(const SCC* scc);
  //    return the elements composing the SCC.
  //
  // static std::vector<ElementRef> getDependencies(const SCC* scc, ElementRef
  // element);
  //    return the elements (which may belong to other SCCs) on which a given
  //    one depends on.

  using Error = typename SCC::UnknownSCCTypeError;
};
} // namespace dependency

namespace internal::dependency {
/// List of the equations composing an SCC.
/// All the equations belong to a given graph.
template <typename Graph>
class SCC {
public:
  using ElementDescriptor = typename Graph::VertexDescriptor;

private:
  using Property = typename Graph::VertexProperty;
  using Container = llvm::SmallVector<ElementDescriptor>;

public:
  using iterator = typename Container::iterator;
  using const_iterator = typename Container::const_iterator;

private:
  const Graph *graph;
  Container equations;

public:
  template <typename It>
  SCC(const Graph &graph, It equationsBegin, It equationsEnd)
      : graph(&graph), equations(equationsBegin, equationsEnd) {}

  friend llvm::hash_code hash_value(const SCC &val) {
    return llvm::hash_combine_range(val.equations.begin(), val.equations.end());
  }

  const Graph &getGraph() const {
    assert(graph != nullptr);
    return *graph;
  }

  /// Get the number of equations composing the SCC.
  [[nodiscard]] size_t size() const { return equations.size(); }

  const Property &operator[](size_t index) const {
    assert(index < equations.size());
    return (*this)[equations[index]];
  }

  /// @name Forwarded methods
  /// {

  const Property &operator[](ElementDescriptor descriptor) const {
    return (*graph)[descriptor];
  }

  /// }
  /// @name Iterators
  /// {

  iterator begin() { return equations.begin(); }

  const_iterator begin() const { return equations.begin(); }

  iterator end() { return equations.end(); }

  const_iterator end() const { return equations.end(); }

  /// }
};
} // namespace internal::dependency

namespace dependency {
// Traits specialization for the internal SCC class.
template <typename Graph>
class SCCTraits<internal::dependency::SCC<Graph>> {
private:
  using Impl = internal::dependency::SCC<Graph>;

public:
  using ElementRef = typename Impl::ElementDescriptor;

  static std::vector<ElementRef> getElements(const Impl *SCC) {
    std::vector<ElementRef> result(SCC->begin(), SCC->end());
    return result;
  }

  static std::vector<ElementRef> getDependencies(const Impl *SCC,
                                                 ElementRef element) {
    std::vector<ElementRef> result;
    const auto &graph = SCC->getGraph();

    for (ElementRef connectedElement :
         llvm::make_range(graph.linkedVerticesBegin(element),
                          graph.linkedVerticesEnd(element))) {
      result.push_back(connectedElement);
    }

    return result;
  }
};
} // namespace dependency
} // namespace marco::modeling

#endif // MARCO_MODELING_SCC_H
