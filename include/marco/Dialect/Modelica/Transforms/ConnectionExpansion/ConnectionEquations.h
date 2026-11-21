#ifndef MARCO_DIALECT_MODELICA_TRANSFORMS_CONNECTIONEXPANSION_CONNECTIONEQUATIONS_H
#define MARCO_DIALECT_MODELICA_TRANSFORMS_CONNECTIONEXPANSION_CONNECTIONEQUATIONS_H

#include "marco/Dialect/Modelica/Transforms/ConnectionExpansion/ConnectionGraph.h"
#include "marco/Modeling/AccessFunction.h"
#include "marco/Modeling/IndexSet.h"
#include "marco/Modeling/MultidimensionalRange.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace mlir::modelica::connection {
/// vertex[i] = root[toRoot(i)] for each i in indices.
struct PotentialFamily {
  marco::modeling::IndexSet indices;
  VertexId vertex, root;
  std::unique_ptr<marco::modeling::AccessFunction> toRoot;
};

/// Term vertex[fromRoot(r)] of the sum of the set whose root is r.
struct FlowTerm {
  VertexId vertex;
  std::unique_ptr<marco::modeling::AccessFunction> fromRoot;
};

/// Sum over the iterators of vertex[access(r, iterators)].
struct FlowReduction {
  VertexId vertex;
  marco::modeling::MultidimensionalRange iterators;
  std::unique_ptr<marco::modeling::AccessFunction> access;
};

/// sum(terms) + sum(reductions) = 0 for each root index r in rootIndices.
struct FlowFamily {
  marco::modeling::IndexSet rootIndices;
  VertexId root;
  llvm::SmallVector<FlowTerm> terms;
  llvm::SmallVector<FlowReduction> reductions;
};

struct ConnectionEquations {
  llvm::SmallVector<PotentialFamily> potentials;
  llvm::SmallVector<FlowFamily> flows;
};

/// Counters of the work done by 'computeEquations'.
struct EquationsStatistics {
  /// Number of contributions to the flow sums visited while merging them and
  /// while partitioning the root indices. It grows linearly with the number
  /// of elements processed point by point.
  size_t numOfVisitedContributions = 0;
};

/// Compute the equations of the connection sets of a graph whose
/// representatives have already been computed. Each element of a potential
/// vertex that is not a representative is equated to its representative,
/// and the members of each flow set are summed to zero. The equations
/// between the inside and outside sides of the same element are omitted.
ConnectionEquations computeEquations(const ConnectionGraph &graph,
                                     EquationsStatistics *statistics = nullptr);
} // namespace mlir::modelica::connection

#endif // MARCO_DIALECT_MODELICA_TRANSFORMS_CONNECTIONEXPANSION_CONNECTIONEQUATIONS_H
