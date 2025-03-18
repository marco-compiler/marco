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

        for (auto &access : equation.getReads()) {
          cachedReadAccesses[equationDescriptor].push_back(std::move(access));
        }
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

    // Explore the current equation accesses.
    const auto &currentEquationWriteAccesses = cachedWriteAccesses[equation];
    const auto &currentEquationReadAccesses = cachedReadAccesses[equation];

    // Prefer affine access functions.
    const Access &currentEquationWriteAccess = getAccessWithProperty(
        currentEquationWriteAccesses, [](const Access &access) {
          return access.getAccessFunction().isAffine();
        });

    IndexSet currentEquationWrittenIndices =
        currentEquationWriteAccess.getAccessFunction().map(equationIndices);

    for (const Access &readAccess : currentEquationReadAccesses) {
      const auto &accessFunction = readAccess.getAccessFunction();
      auto readVariableIndices = accessFunction.map(equationIndices);

      auto writingEquations = writesMap.equal_range(readAccess.getVariable());

      for (const auto &[variableId, writeInfo] :
           llvm::make_range(writingEquations.first, writingEquations.second)) {
        const IndexSet &writtenVariableIndices =
            writeInfo.getWrittenVariableIndexes();

        // If the ranges do not overlap, then there is no loop involving
        // the writing equation.
        if (!readVariableIndices.overlaps(writtenVariableIndices)) {
          continue;
        }

        // Determine the indices of the writing equation that lead to the
        // requested access.
        auto variableIndicesIntersection =
            readVariableIndices.intersect(writtenVariableIndices);

        EquationDescriptor writingEquation = writeInfo.getEquation();

        IndexSet allWritingEquationIndices(
            arrayDependencyGraph.getEquation(writingEquation)
                .getIterationRanges());

        const auto &writingEquationWriteAccesses =
            cachedWriteAccesses[writingEquation];

        // Prefer invertible access functions.
        const Access &writingEquationWriteAccess = getAccessWithProperty(
            writingEquationWriteAccesses, [](const Access &access) {
              return access.getAccessFunction().isInvertible();
            });

        const AccessFunction &writingEquationAccessFunction =
            writingEquationWriteAccess.getAccessFunction();

        IndexSet usedWritingEquationIndices =
            writingEquationAccessFunction.inverseMap(
                variableIndicesIntersection, allWritingEquationIndices);

        // Avoid visiting the same indices multiple times.
        usedWritingEquationIndices -= visitedEquationIndices[writingEquation];

        if (usedWritingEquationIndices.empty()) {
          continue;
        }

        Path extendedPath =
            path + PathDependency(equation, equationIndices,
                                  currentEquationWriteAccess,
                                  currentEquationWrittenIndices, readAccess,
                                  readVariableIndices);

        restrictPathIndices(extendedPath);

        if (auto pathLength = extendedPath.size(); pathLength > 1) {
          // Search along the restricted path if the current equation has
          // already been visited with some of the current indices.
          auto dependencyIt = llvm::find_if(
              extendedPath, [&](const PathDependency &dependency) {
                if (dependency.equation != writingEquation) {
                  return false;
                }

                return dependency.equationIndices.overlaps(
                    usedWritingEquationIndices);
              });

          if (dependencyIt != extendedPath.end()) {
            // Check if the path involves more than one array equation.
            bool isSelfLoop = std::all_of(
                dependencyIt, extendedPath.end(),
                [&](const PathDependency &dependency) {
                  return dependency.equation == extendedPath.begin()->equation;
                });

            if (!isSelfLoop) {
              Path cycle;

              for (auto it = dependencyIt; it != extendedPath.end(); ++it) {
                cycle = cycle + std::move(*it);
              }

              cycles.push_back(cycle);
            }

            continue;
          }
        }

        getEquationsCycles(newCycles, writesMap, cachedWriteAccesses,
                           cachedReadAccesses, visitedEquationIndices,
                           writingEquation, usedWritingEquationIndices,
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

  const Access &getAccessWithProperty(
      llvm::ArrayRef<Access> accesses,
      std::function<bool(const Access &)> preferenceFn) const {
    assert(!accesses.empty());
    auto it = llvm::find_if(accesses, preferenceFn);

    if (it == accesses.end()) {
      it = accesses.begin();
    }

    return *it;
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
};
} // namespace marco::modeling

#endif // MARCO_MODELING_CYCLES_H
