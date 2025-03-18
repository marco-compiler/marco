#ifndef MARCO_MODELING_CYCLES_H
#define MARCO_MODELING_CYCLES_H

#include "marco/Modeling/ArrayEquationsDependencyGraph.h"
#include "marco/Modeling/Dumpable.h"
#include "marco/Modeling/IndexSet.h"
#include "marco/Modeling/SCC.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/SCCIterator.h"
#include <list>

namespace marco::modeling {
namespace internal::dependency {
template <typename EquationDescriptor, typename Access>
class PathDependency {
public:
  PathDependency(EquationDescriptor equation, IndexSet equationIndices,
                 Access writeAccess, IndexSet writtenVariableIndices,
                 Access readAccess, IndexSet readVariableIndices)
      : equation(equation), equationIndices(std::move(equationIndices)),
        writeAccess(std::move(writeAccess)),
        writtenVariableIndices(std::move(writtenVariableIndices)),
        readAccess(std::move(readAccess)),
        readVariableIndices(std::move(readVariableIndices)) {}

  EquationDescriptor equation;
  IndexSet equationIndices;
  Access writeAccess;
  IndexSet writtenVariableIndices;
  Access readAccess;
  IndexSet readVariableIndices;
};

template <typename EquationDescriptor, typename Access>
class Path {
private:
  using Dependency = PathDependency<EquationDescriptor, Access>;
  using Container = std::list<Dependency>;

public:
  using iterator = typename Container::iterator;
  using const_iterator = typename Container::const_iterator;

  using reverse_iterator = typename Container::reverse_iterator;

  using const_reverse_iterator = typename Container::const_reverse_iterator;

  [[nodiscard]] size_t size() const { return equations.size(); }

  [[nodiscard]] iterator begin() { return equations.begin(); }

  [[nodiscard]] const_iterator begin() const { return equations.begin(); }

  [[nodiscard]] iterator end() { return equations.end(); }

  [[nodiscard]] const_iterator end() const { return equations.end(); }

  [[nodiscard]] reverse_iterator rbegin() { return equations.rbegin(); }

  [[nodiscard]] const_reverse_iterator rbegin() const {
    return equations.rbegin();
  }

  [[nodiscard]] reverse_iterator rend() { return equations.rend(); }

  [[nodiscard]] const_reverse_iterator rend() const { return equations.rend(); }

  Dependency &back() { return equations.back(); }

  const Dependency &back() const { return equations.back(); }

  Path operator+(Dependency equation) {
    Path result(*this);
    result += std::move(equation);
    return result;
  }

  Path &operator+=(Dependency equation) {
    equations.push_back(std::move(equation));
    return *this;
  }

  Path withoutLast(size_t n) const {
    assert(n <= equations.size());

    Path result;
    auto it = equations.begin();

    for (size_t i = 0, e = equations.size() - n; i < e; ++i) {
      result.equations.push_back(*it);
      ++it;
    }

    return result;
  }

private:
  Container equations;
};

template <typename EquationDescriptor, typename Access>
class CyclicEquation {
public:
  CyclicEquation(EquationDescriptor equation, IndexSet equationIndices,
                 Access writeAccess, IndexSet writtenVariableIndices,
                 Access readAccess, IndexSet readVariableIndices)
      : equation(equation), equationIndices(std::move(equationIndices)),
        writeAccess(std::move(writeAccess)),
        writtenVariableIndices(std::move(writtenVariableIndices)),
        readAccess(std::move(readAccess)),
        readVariableIndices(std::move(readVariableIndices)) {}

  EquationDescriptor equation;
  IndexSet equationIndices;
  Access writeAccess;
  IndexSet writtenVariableIndices;
  Access readAccess;
  IndexSet readVariableIndices;
};

template <typename EquationDescriptor>
class ReducedEquationView {
public:
  ReducedEquationView(EquationDescriptor equation, IndexSet indices)
      : equation(equation), indices(std::move(indices)) {}

  [[nodiscard]] EquationDescriptor operator*() const { return equation; }

  [[nodiscard]] const IndexSet &getIndices() const { return indices; }

private:
  EquationDescriptor equation;
  IndexSet indices;
};
} // namespace internal::dependency

namespace internal::dependency_graph {
template <typename EquationView>
class SCC {
public:
  using Container = llvm::SmallVector<EquationView>;
  using const_iterator = typename Container::const_iterator;

  llvm::ArrayRef<EquationView> getEquations() const { return equations; }

  void addEquation(EquationView equation) {
    equations.push_back(std::move(equation));
  }

  const_iterator begin() const { return equations.begin(); }

  const_iterator end() const { return equations.end(); }

private:
  Container equations;
};
} // namespace internal::dependency_graph

template <typename VariableProperty, typename EquationProperty>
class DependencyGraph {
public:
  using ArrayDependencyGraph =
      ArrayEquationsDependencyGraph<VariableProperty, EquationProperty>;

  using Variable = typename ArrayDependencyGraph::Variable;
  using Equation = typename ArrayDependencyGraph::Equation;

  using EquationDescriptor = typename ArrayDependencyGraph::EquationDescriptor;

  using AccessProperty = typename ArrayDependencyGraph::AccessProperty;
  using Access = typename ArrayDependencyGraph::Access;

  using WritesMap = typename ArrayDependencyGraph::WritesMap;

  using PathDependency =
      internal::dependency::PathDependency<EquationDescriptor, Access>;

  using Path = internal::dependency::Path<EquationDescriptor, Access>;

  using CyclicEquation =
      internal::dependency::CyclicEquation<EquationDescriptor, Access>;

  using Cycle = llvm::SmallVector<CyclicEquation>;

  using EquationView =
      internal::dependency::ReducedEquationView<EquationDescriptor>;

  using SCC = internal::dependency_graph::SCC<EquationView>;

  using CachedAccesses =
      llvm::DenseMap<EquationDescriptor, llvm::SmallVector<Access>>;

private:
  mlir::MLIRContext *context;
  ArrayDependencyGraph arrayDependencyGraph;

public:
  explicit DependencyGraph(mlir::MLIRContext *context)
      : context(context), arrayDependencyGraph(context) {}

  [[nodiscard]] mlir::MLIRContext *getContext() const {
    assert(context != nullptr);
    return context;
  }

  void addEquations(llvm::ArrayRef<EquationProperty> equations) {
    arrayDependencyGraph.addEquations(equations);
  }

  EquationProperty &operator[](EquationDescriptor descriptor) {
    return arrayDependencyGraph[descriptor];
  }

  const EquationProperty &operator[](EquationDescriptor descriptor) const {
    return arrayDependencyGraph[descriptor];
  }

  std::vector<Cycle> getEquationsCycles() const {
    std::vector<Cycle> result;
    std::mutex resultMutex;

    auto SCCs = arrayDependencyGraph.getSCCs();

    auto processFn = [&](const typename ArrayDependencyGraph::SCC &scc) {
      auto writesMap =
          arrayDependencyGraph.getWritesMap(scc.begin(), scc.end());

      CachedAccesses cachedWriteAccesses;
      CachedAccesses cachedReadAccesses;

      for (EquationDescriptor equationDescriptor : scc) {
        const auto &equation =
            arrayDependencyGraph.getEquation(equationDescriptor);

        for (auto &access : equation.getWrites()) {
          cachedWriteAccesses[equationDescriptor].push_back(std::move(access));
        }

        // Prefer affine write access functions.
        llvm::sort(cachedWriteAccesses[equationDescriptor],
                   [](const Access &first, const Access &second) {
                     return first.getAccessFunction().isAffine() &&
                            !second.getAccessFunction().isAffine();
                   });

        for (auto &access : equation.getReads()) {
          cachedReadAccesses[equationDescriptor].push_back(std::move(access));
        }

        // Prefer invertible read access functions.
        llvm::sort(cachedWriteAccesses[equationDescriptor],
                   [](const Access &first, const Access &second) {
                     return first.getAccessFunction().isInvertible() &&
                            !second.getAccessFunction().isInvertible();
                   });
      }

      llvm::DenseMap<EquationDescriptor, IndexSet> visitedEquationIndices;

      for (const EquationDescriptor &equationDescriptor : scc) {
        llvm::SmallVector<Path> paths;

        getEquationsCycles(paths, writesMap, cachedWriteAccesses,
                           cachedReadAccesses, visitedEquationIndices,
                           equationDescriptor);

        std::lock_guard<std::mutex> lockGuard(resultMutex);

        for (Path &path : paths) {
          Cycle cyclicEquations;

          for (PathDependency &dependency : path) {
            cyclicEquations.push_back(CyclicEquation(
                dependency.equation, std::move(dependency.equationIndices),
                std::move(dependency.writeAccess),
                std::move(dependency.writtenVariableIndices),
                std::move(dependency.readAccess),
                std::move(dependency.readVariableIndices)));
          }

          result.push_back(std::move(cyclicEquations));
        }
      }
    };

    mlir::parallelForEach(getContext(), SCCs, processFn);
    return result;
  }

  void getSCCs(llvm::SmallVectorImpl<SCC> &result) const {
    std::vector<Cycle> cycles = getEquationsCycles();

    // Function to search for an SCC into which a cycle should be merged.
    auto searchSCCFn = [&](const Cycle &cycle) {
      for (const CyclicEquation &cyclicEquation : cycle) {
        auto sccIt = llvm::find_if(result, [&](const SCC &scc) {
          for (const EquationView &sccEquation : scc) {
            if (*sccEquation == cyclicEquation.equation) {
              return true;
            }
          }

          return false;
        });

        if (sccIt != result.end()) {
          return sccIt;
        }
      }

      return result.end();
    };

    // Keep track of all the indices belonging to SCCs.
    llvm::DenseMap<EquationDescriptor, IndexSet> processedIndices;

    for (const auto &cycle : cycles) {
      auto sccIt = searchSCCFn(cycle);

      if (sccIt == result.end()) {
        // New SCC.
        SCC newSCC;

        for (const CyclicEquation &cyclicEquation : cycle) {
          newSCC.addEquation(EquationView(cyclicEquation.equation,
                                          cyclicEquation.equationIndices));

          processedIndices[cyclicEquation.equation] +=
              cyclicEquation.equationIndices;
        }

        result.push_back(std::move(newSCC));
      } else {
        // Merge the cycle into an SCC having some common equations.
        llvm::DenseMap<EquationDescriptor, IndexSet> mergedEquations;

        for (const CyclicEquation &cyclicEquation : cycle) {
          mergedEquations[cyclicEquation.equation] +=
              cyclicEquation.equationIndices;

          processedIndices[cyclicEquation.equation] +=
              cyclicEquation.equationIndices;
        }

        for (const EquationView &equation : *sccIt) {
          mergedEquations[*equation] += equation.getIndices();
          processedIndices[*equation] += equation.getIndices();
        }

        SCC mergedSCC;

        for (const auto &equation : mergedEquations) {
          mergedSCC.addEquation(
              EquationView(equation.getFirst(), equation.getSecond()));
        }

        *sccIt = std::move(mergedSCC);
      }
    }

    // Create an SCC for each remaining equation.
    for (EquationDescriptor equation :
         llvm::make_range(arrayDependencyGraph.equationsBegin(),
                          arrayDependencyGraph.equationsEnd())) {
      IndexSet allIndices =
          arrayDependencyGraph.getEquation(equation).getIterationRanges();

      IndexSet remainingIndices = allIndices - processedIndices[equation];

      if (!remainingIndices.empty()) {
        SCC newSCC;

        newSCC.addEquation(EquationView(equation, std::move(remainingIndices)));

        result.push_back(std::move(newSCC));
      }
    }
  }

private:
  void getEquationsCycles(
      llvm::SmallVectorImpl<Path> &cycles, const WritesMap &writesMap,
      CachedAccesses &cachedWriteAccesses, CachedAccesses &cachedReadAccesses,
      llvm::DenseMap<EquationDescriptor, IndexSet> &visitedEquationIndices,
      EquationDescriptor equation) const {
    // The first equation starts with the full range, as it has no
    // predecessors.
    IndexSet equationIndices(
        arrayDependencyGraph.getEquation(equation).getIterationRanges());

    getEquationsCycles(cycles, writesMap, cachedWriteAccesses,
                       cachedReadAccesses, visitedEquationIndices, equation,
                       equationIndices - visitedEquationIndices[equation], {});
  }

  void getEquationsCycles(
      llvm::SmallVectorImpl<Path> &cycles, const WritesMap &writesMap,
      CachedAccesses &cachedWriteAccesses, CachedAccesses &cachedReadAccesses,
      llvm::DenseMap<EquationDescriptor, IndexSet> &visitedEquationIndices,
      EquationDescriptor equation, const IndexSet &equationIndices,
      Path path) const {
    llvm::SmallVector<Path> newCycles;

    // Visit all the write accesses. When the indices of the equation get
    // restricted, the written indices may differ and even not overlap anymore.
    // We keep track of the visited indices to avoid going through access
    // functions that would lead to the same cycles.
    IndexSet visitedWrittenIndices;

    for (const Access &currentEquationWriteAccess :
         cachedWriteAccesses[equation]) {
      IndexSet currentEquationWrittenIndices =
          currentEquationWriteAccess.getAccessFunction().map(equationIndices);

      if (visitedWrittenIndices.contains(currentEquationWrittenIndices)) {
        continue;
      }

      visitedWrittenIndices += currentEquationWrittenIndices;

      // The set of read accesses needs to be recomputed according to the
      // current write access being considered. Restrict the set of indices
      // considered for the equation may indeed lead some write accesses to
      // become read accesses. For example, consider the accesses x[i] and
      // x[10 - i], and the original equation indices [0 to 10]. In this case,
      // both the accesses write to the indices [0 to 10] of x. However, if the
      // indices get restricted to [3, 5], they would respectively access
      // x[3 to 5] and x[7 to 5]. When considering the first access as a write
      // access, the second access should be considered a write access for the
      // equation indices [5 to 5], which lead to x[5], while it should be
      // considered as a read access for the equation indices [3 to 4], which
      // lead to x[7 to 6].

      llvm::SmallVector<Access> currentEquationReadAccesses;

      // The original read accesses remain read accesses. At most, they access
      // to a reduced set of indices, but they will not overlap with written
      // indices.
      llvm::append_range(currentEquationReadAccesses,
                         cachedReadAccesses[equation]);

      // Write accesses, on the contrary, may overlap.
      for (const Access &access : cachedWriteAccesses[equation]) {
        IndexSet accessedVariableIndices =
            access.getAccessFunction().map(equationIndices);

        IndexSet remainingIndices =
            accessedVariableIndices - currentEquationWrittenIndices;

        if (!remainingIndices.empty()) {
          currentEquationReadAccesses.push_back(access);
        }
      }

      for (const Access &readAccess : currentEquationReadAccesses) {
        IndexSet readVariableIndices =
            readAccess.getAccessFunction().map(equationIndices);

        if (readAccess.getVariable() ==
            currentEquationWriteAccess.getVariable()) {
          // Ensure that the written indices are excluded in case of read
          // accesses obtained from write accesses as a consequence of indices
          // restriction.
          readVariableIndices -= currentEquationWrittenIndices;
        }

        auto writingEquations = writesMap.equal_range(readAccess.getVariable());

        for (const auto &[variableId, writeInfo] :
             llvm::make_range(writingEquations)) {
          EquationDescriptor writingEquation = writeInfo.getEquation();

          const IndexSet &writtenVariableIndices =
              writeInfo.getWrittenVariableIndexes();

          // If the ranges do not overlap, then there is no loop involving
          // the writing equation.
          if (!readVariableIndices.overlaps(writtenVariableIndices)) {
            continue;
          }

          // The indices of the read variable that are also written by the
          // identified writing equation.
          auto variableIndicesIntersection =
              readVariableIndices.intersect(writtenVariableIndices);

          // Determine the indices of the writing equation that lead to the
          // requested access.
          IndexSet writingEquationIndices = getWritingEquationIndices(
              writingEquation, cachedWriteAccesses[writingEquation],
              variableIndicesIntersection);

          // Avoid visiting the same indices multiple times.
          writingEquationIndices -= visitedEquationIndices[writingEquation];

          if (writingEquationIndices.empty()) {
            continue;
          }

          // Append the current equation to the traversal path.
          Path extendedPath =
              path + PathDependency(equation, equationIndices,
                                    currentEquationWriteAccess,
                                    currentEquationWrittenIndices, readAccess,
                                    variableIndicesIntersection);

          // Visit the path backwards and restrict the equation indices, if
          // necessary.
          restrictPathIndices(extendedPath);

          // Check if the writing equation leads to a cycle.
          if (auto cycle = extractCycleIfAny(extendedPath, writingEquation,
                                             writingEquationIndices)) {
            newCycles.push_back(std::move(*cycle));
            continue;
          }

          getEquationsCycles(newCycles, writesMap, cachedWriteAccesses,
                             cachedReadAccesses, visitedEquationIndices,
                             writingEquation, writingEquationIndices,
                             std::move(extendedPath));
        }
      }

      for (Path &cycle : newCycles) {
        for (const PathDependency &pathDependency : cycle) {
          visitedEquationIndices[pathDependency.equation] +=
              pathDependency.equationIndices;
        }

        cycles.push_back(std::move(cycle));
      }
    }
  }

  /// Compute the indices of an equation that lead to a write to the given
  /// indices of a variable.
  IndexSet getWritingEquationIndices(EquationDescriptor equation,
                                     llvm::ArrayRef<Access> writeAccesses,
                                     const IndexSet &variableIndices) const {
    IndexSet allEquationIndices(
        arrayDependencyGraph.getEquation(equation).getIterationRanges());

    IndexSet writingEquationIndices;

    for (const Access &access : writeAccesses) {
      writingEquationIndices += access.getAccessFunction().inverseMap(
          variableIndices, allEquationIndices);
    }

    return writingEquationIndices;
  }

  void restrictPathIndices(Path &path) const {
    auto it = path.rbegin();
    auto endIt = path.rend();

    if (it == endIt) {
      return;
    }

    // Traverse the path backwards.
    auto prevIt = it;

    while (++it != endIt) {
      // Propagate the indices written by the current equation as the indices
      // read by the equation preceding in the path.
      it->readVariableIndices = prevIt->writtenVariableIndices;

      // Compute the indices of the equation that lead to that read access.
      const Access &readAccess = it->readAccess;

      IndexSet newEquationIndices = readAccess.getAccessFunction().inverseMap(
          it->readVariableIndices, it->equationIndices);

      if (newEquationIndices == it->equationIndices) {
        // Stop iterating through the path.
        // No further modifications will be performed.
        return;
      }

      it->equationIndices = std::move(newEquationIndices);

      it->writtenVariableIndices =
          it->writeAccess.getAccessFunction().map(it->equationIndices);

      prevIt = it;
    }
  }

  std::optional<Path>
  extractCycleIfAny(const Path &path, EquationDescriptor nextEquation,
                    const IndexSet &nextEquationIndices) const {
    if (auto length = path.size(); length <= 1) {
      return std::nullopt;
    }

    // Search along the restricted path if the next equation has
    // already been visited with some of the given indices.
    auto dependencyIt =
        llvm::find_if(path, [&](const PathDependency &dependency) {
          if (dependency.equation != nextEquation) {
            return false;
          }

          return dependency.equationIndices.overlaps(nextEquationIndices);
        });

    if (dependencyIt != path.end()) {
      // Check if the path involves more than one array equation.
      bool isSelfLoop = std::all_of(
          dependencyIt, path.end(), [&](const PathDependency &dependency) {
            return dependency.equation == path.begin()->equation;
          });

      if (!isSelfLoop) {
        Path cycle;

        for (auto it = dependencyIt; it != path.end(); ++it) {
          cycle = cycle + std::move(*it);
        }

        return cycle;
      }
    }

    return std::nullopt;
  }
};
} // namespace marco::modeling

#endif // MARCO_MODELING_CYCLES_H
