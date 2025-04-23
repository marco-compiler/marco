#ifndef MARCO_MODELING_CYCLES_H
#define MARCO_MODELING_CYCLES_H

#include "marco/Modeling/ArrayEquationsDependencyGraph.h"
#include "marco/Modeling/Dumpable.h"
#include "marco/Modeling/IndexSet.h"
#include "marco/Modeling/SCC.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SCCIterator.h"

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
  using Container = llvm::SmallVector<Dependency, 3>;

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

  Dependency &operator[](size_t index) { return equations[index]; }

  const Dependency &operator[](size_t index) const { return equations[index]; }

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
} // namespace marco::modeling

namespace llvm {
template <typename EquationDescriptor>
struct DenseMapInfo<marco::modeling::internal::dependency::ReducedEquationView<
    EquationDescriptor>> {
  using Key = marco::modeling::internal::dependency::ReducedEquationView<
      EquationDescriptor>;

  static Key getEmptyKey() {
    return {DenseMapInfo<EquationDescriptor>::getEmptyKey(),
            DenseMapInfo<marco::modeling::IndexSet>::getEmptyKey()};
  }

  static Key getTombstoneKey() {
    return {DenseMapInfo<EquationDescriptor>::getTombstoneKey(),
            DenseMapInfo<marco::modeling::IndexSet>::getTombstoneKey()};
  }

  static unsigned getHashValue(const Key &val) {
    return llvm::hash_combine(*val, val.getIndices());
  }

  static bool isEqual(const Key &lhs, const Key &rhs) {
    return DenseMapInfo<EquationDescriptor>::isEqual(*lhs, *rhs) &&
           DenseMapInfo<marco::modeling::IndexSet>::isEqual(lhs.getIndices(),
                                                            rhs.getIndices());
  }
};
} // namespace llvm

namespace marco::modeling {
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

  using WriteInfo = typename ArrayDependencyGraph::WriteInfo;
  using WritesMap = typename ArrayDependencyGraph::WritesMap;

  using PathDependency =
      internal::dependency::PathDependency<EquationDescriptor, Access>;

  using Path = internal::dependency::Path<EquationDescriptor, Access>;

  using CyclicEquation =
      internal::dependency::CyclicEquation<EquationDescriptor, Access>;

  using Cycle = llvm::SmallVector<CyclicEquation, 3>;

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

  llvm::SmallVector<Cycle, 3> getEquationsCycles() const {
    llvm::SmallVector<Cycle, 3> result;
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

      llvm::SmallVector<Path, 3> allPaths;
      llvm::DenseMap<EquationDescriptor, IndexSet> visitedEquationIndices;

      for (const EquationDescriptor &equationDescriptor : scc) {
        getEquationsCycles(allPaths, writesMap, cachedWriteAccesses,
                           cachedReadAccesses, visitedEquationIndices,
                           equationDescriptor);
      }

      std::lock_guard<std::mutex> lockGuard(resultMutex);

      for (Path &path : allPaths) {
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
    };

    mlir::parallelForEach(getContext(), SCCs, processFn);
    return result;
  }

  llvm::SmallVector<SCC> getSCCs() const {
    llvm::SmallVector<SCC> result;
    std::mutex resultMutex;

    const auto SCCs = arrayDependencyGraph.getSCCs();

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

      llvm::SmallVector<Path, 3> cycles;
      llvm::DenseMap<EquationDescriptor, IndexSet> visitedEquationIndices;

      for (const EquationDescriptor &equationDescriptor : scc) {
        getEquationsCycles(cycles, writesMap, cachedWriteAccesses,
                           cachedReadAccesses, visitedEquationIndices,
                           equationDescriptor);
      }

      llvm::SmallVector<Path, 3> partitionedCycles;

      for (Path &cycle : cycles) {
        addCycleAndSplit(partitionedCycles, std::move(cycle));
      }

      assert(checkEquationsPartitioning(partitionedCycles) &&
             "Found overlapping and non-equal equation indices");

      using PartitionsGraph =
          internal::dependency::SingleEntryWeaklyConnectedDigraph<EquationView>;

      using Partition = typename PartitionsGraph::VertexDescriptor;
      PartitionsGraph partitionsGraph;
      llvm::DenseMap<EquationView, Partition> partitionsMap;

      // Add the nodes.
      for (const Path &path : partitionedCycles) {
        for (const PathDependency &pathDependency : path) {
          EquationView equationView(pathDependency.equation,
                                    pathDependency.equationIndices);

          if (!partitionsMap.contains(equationView)) {
            auto vertexDescriptor = partitionsGraph.addVertex(equationView);
            partitionsMap[equationView] = vertexDescriptor;
          }
        }
      }

      // Add the edges.
      for (const Path &path : partitionedCycles) {
        EquationView first(path[0].equation, path[0].equationIndices);
        EquationView source = first;

        for (size_t i = 1, e = path.size(); i < e; ++i) {
          EquationView destination(path[i].equation, path[i].equationIndices);
          partitionsGraph.addEdge(partitionsMap[source],
                                  partitionsMap[destination]);
          source = std::move(destination);
        }

        partitionsGraph.addEdge(partitionsMap[source], partitionsMap[first]);
      }

      // Create the cyclic SCCs.
      auto sccBeginIt = llvm::scc_begin(
          static_cast<const PartitionsGraph *>(&partitionsGraph));

      auto sccEndIt =
          llvm::scc_end(static_cast<const PartitionsGraph *>(&partitionsGraph));

      for (auto it = sccBeginIt; it != sccEndIt; ++it) {
        std::unique_lock<std::mutex> lock(resultMutex);
        SCC &partitionsSCC = result.emplace_back();
        lock.unlock();

        for (auto partitionDescriptor : *it) {
          if (partitionDescriptor != partitionsGraph.getEntryNode()) {
            partitionsSCC.addEquation(partitionsGraph[partitionDescriptor]);
          }
        }
      }

      // Create the SCCs for the remaining indices.
      for (const EquationDescriptor &equation : scc) {
        IndexSet remainingEquationIndices =
            arrayDependencyGraph.getEquation(equation).getIterationRanges();

        remainingEquationIndices -= visitedEquationIndices[equation];

        if (!remainingEquationIndices.empty()) {
          std::lock_guard<std::mutex> lock(resultMutex);
          SCC &remainingIndicesSCC = result.emplace_back();

          remainingIndicesSCC.addEquation(
              EquationView(equation, std::move(remainingEquationIndices)));
        }
      }
    };

    mlir::parallelForEach(getContext(), SCCs, processFn);
    return result;
  }

private:
  [[maybe_unused, nodiscard]] bool
  checkEquationsPartitioning(llvm::ArrayRef<Path> cycles) const {
    auto checkEquationFn = [&](EquationDescriptor equation,
                               const IndexSet &indices) {
      for (const Path &path : cycles) {
        for (const PathDependency &pathDependency : path) {
          if (pathDependency.equation == equation) {
            if (pathDependency.equationIndices.overlaps(indices) &&
                pathDependency.equationIndices != indices) {
              return false;
            }
          }
        }
      }

      return true;
    };

    for (const Path &path : cycles) {
      for (const PathDependency &pathDependency : path) {
        if (!checkEquationFn(pathDependency.equation,
                             pathDependency.equationIndices)) {
          return false;
        }
      }
    }

    return true;
  }

  void getEquationsCycles(
      llvm::SmallVectorImpl<Path> &cycles, const WritesMap &writesMap,
      CachedAccesses &cachedWriteAccesses, CachedAccesses &cachedReadAccesses,
      llvm::DenseMap<EquationDescriptor, IndexSet> &visitedEquationIndices,
      EquationDescriptor equation) const {
    // The first equation starts with the full range, as it has no
    // predecessors.
    IndexSet equationIndices(
        arrayDependencyGraph.getEquation(equation).getIterationRanges());

    IndexSet unvisitedEquationIndices =
        equationIndices - visitedEquationIndices[equation];

    if (unvisitedEquationIndices.empty()) {
      return;
    }

    getEquationsCycles(cycles, writesMap, cachedWriteAccesses,
                       cachedReadAccesses, visitedEquationIndices, equation,
                       unvisitedEquationIndices, {});
  }

  void getEquationsCycles(
      llvm::SmallVectorImpl<Path> &cycles, const WritesMap &writesMap,
      CachedAccesses &cachedWriteAccesses, CachedAccesses &cachedReadAccesses,
      llvm::DenseMap<EquationDescriptor, IndexSet> &visitedEquationIndices,
      EquationDescriptor equation, const IndexSet &equationIndices,
      Path path) const {
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
      // current write access being considered. Restricting the set of indices
      // considered for the equation may indeed lead some write accesses to
      // become read accesses. For example, consider the accesses x[i] and
      // x[10 - i], and the original equation indices [0 to 10]. In this case,
      // both the accesses write to the indices [0 to 10] of x. However, if the
      // indices get restricted to [3, 5], they would respectively access
      // x[3 to 5] and x[7 to 5]. When considering the first access as a write
      // access, the second access should be considered as a write access for
      // the equation indices [5 to 5], which lead to x[5], while it should be
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

        auto writingEquationsIt = writesMap.find(readAccess.getVariable());

        if (writingEquationsIt == writesMap.end()) {
          continue;
        }

        for (const MultidimensionalRange &readVariableIndicesRange :
             llvm::make_range(readVariableIndices.rangesBegin(),
                              readVariableIndices.rangesEnd())) {
          writingEquationsIt->second.walkOverlappingObjects(
              readVariableIndicesRange, [&](const WriteInfo &writeInfo) {
                EquationDescriptor writingEquation = writeInfo.getEquation();

                const MultidimensionalRange &writtenVariableIndices =
                    writeInfo.getWrittenVariableIndices();

                // If the ranges do not overlap, then there is no loop involving
                // the writing equation.
                if (!readVariableIndices.overlaps(writtenVariableIndices)) {
                  return;
                }

                // The indices of the read variable that are also written by the
                // identified writing equation.
                auto variableIndicesIntersection =
                    readVariableIndices.intersect(writtenVariableIndices);

                // Determine the indices of the writing equation that lead to
                // the requested access.
                IndexSet writingEquationIndices = getWritingEquationIndices(
                    writingEquation, cachedWriteAccesses[writingEquation],
                    variableIndicesIntersection);

                // Avoid visiting the same indices multiple times.
                writingEquationIndices -=
                    visitedEquationIndices[writingEquation];

                if (writingEquationIndices.empty()) {
                  return;
                }

                // Append the current equation to the traversal path.
                Path extendedPath =
                    path + PathDependency(equation, equationIndices,
                                          currentEquationWriteAccess,
                                          currentEquationWrittenIndices,
                                          readAccess,
                                          variableIndicesIntersection);

                // Visit the path backwards and restrict the equation indices,
                // if necessary.
                restrictPathIndicesBackward(extendedPath,
                                            extendedPath.size() - 1);

                // Check if the writing equation leads to a cycle.
                if (auto cycle =
                        extractCycleIfAny(extendedPath, writingEquation,
                                          writingEquationIndices)) {
                  for (const PathDependency &pathDependency : *cycle) {
                    visitedEquationIndices[pathDependency.equation] +=
                        pathDependency.equationIndices;
                  }

                  cycles.push_back(std::move(*cycle));
                }

                getEquationsCycles(cycles, writesMap, cachedWriteAccesses,
                                   cachedReadAccesses, visitedEquationIndices,
                                   writingEquation, writingEquationIndices,
                                   std::move(extendedPath));
              });
        }
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

  void restrictPathIndicesForward(Path &path, size_t startingPoint) const {
    PathDependency *previous = &path[startingPoint];

    for (size_t i = startingPoint + 1, e = path.size(); i < e; ++i) {
      PathDependency *current = &path[i];

      // Propagate the indices read by the previous equation.
      current->writtenVariableIndices = previous->readVariableIndices;

      // Compute the indices of the equation that lead to that write access.
      IndexSet newEquationIndices =
          current->writeAccess.getAccessFunction().inverseMap(
              current->writtenVariableIndices, current->equationIndices);

      if (newEquationIndices == current->equationIndices) {
        // Stop iterating through the path.
        // No further modifications will be performed.
        return;
      }

      current->equationIndices = std::move(newEquationIndices);

      current->readVariableIndices =
          current->readAccess.getAccessFunction().map(current->equationIndices);

      previous = current;
    }
  }

  void restrictPathIndicesBackward(Path &path, size_t startingPoint) const {
    PathDependency *previous = &path[startingPoint];

    for (size_t i = 1, e = startingPoint; i <= e; ++i) {
      PathDependency *current = &path[e - i];

      // Propagate the indices written by the current equation as the indices
      // read by the equation preceding in the path.
      current->readVariableIndices = previous->writtenVariableIndices;

      // Compute the indices of the equation that lead to that read access.
      const Access &readAccess = current->readAccess;

      IndexSet newEquationIndices = readAccess.getAccessFunction().inverseMap(
          current->readVariableIndices, current->equationIndices);

      if (newEquationIndices == current->equationIndices) {
        // Stop iterating through the path.
        // No further modifications will be performed.
        return;
      }

      current->equationIndices = std::move(newEquationIndices);

      current->writtenVariableIndices =
          current->writeAccess.getAccessFunction().map(
              current->equationIndices);

      previous = current;
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

  void addCycleAndSplit(llvm::SmallVectorImpl<Path> &cycles,
                        Path newCycle) const {
    // Split the existing cycles.
    for (const PathDependency &pathDependency : newCycle) {
      llvm::SmallVector<Path, 3> newCycles;

      for (Path &existingCycle : cycles) {
        splitCycle(newCycles, std::move(existingCycle), pathDependency.equation,
                   pathDependency.equationIndices);
      }

      cycles = std::move(newCycles);
    }

    // Split the new cycle.
    llvm::SmallVector<Path, 3> newCycles;
    newCycles.push_back(std::move(newCycle));

    for (const Path &cycle : cycles) {
      for (const PathDependency &pathDependency : cycle) {
        llvm::SmallVector<Path, 3> split;

        for (Path &currentNewCycle : newCycles) {
          splitCycle(split, std::move(currentNewCycle), pathDependency.equation,
                     pathDependency.equationIndices);
        }

        newCycles = std::move(split);
      }
    }

    llvm::append_range(cycles, std::move(newCycles));
  }

  /// Split a cycle so that its components either exactly match or do not
  /// overlap the given equation.
  void splitCycle(llvm::SmallVectorImpl<Path> &result, Path cycle,
                  EquationDescriptor equation,
                  const IndexSet &equationIndices) const {
    llvm::SmallVector<Path, 3> workList;
    workList.push_back(std::move(cycle));

    auto findOverlappingPos = [&](const Path &path) -> std::optional<size_t> {
      for (const auto &pathDependency : llvm::enumerate(path)) {
        if (pathDependency.value().equation == equation &&
            pathDependency.value().equationIndices.overlaps(equationIndices) &&
            !equationIndices.contains(pathDependency.value().equationIndices)) {
          return pathDependency.index();
        }
      }

      return std::nullopt;
    };

    while (!workList.empty()) {
      Path currentCycle = workList.pop_back_val();

      // Search for an equation that has overlapping but fully contained
      // indices.
      std::optional<size_t> overlappingPos = findOverlappingPos(currentCycle);

      if (!overlappingPos) {
        // No need for a split.
        result.push_back(std::move(currentCycle));
        continue;
      }

      const PathDependency &node = currentCycle[*overlappingPos];
      IndexSet commonIndices = node.equationIndices.intersect(equationIndices);
      IndexSet difference = node.equationIndices - commonIndices;

      // Copy the original cycle and set the indices obtained as difference.
      Path differenceCycle(currentCycle);
      setEquationIndices(differenceCycle[*overlappingPos], difference);
      restrictPathIndicesForward(differenceCycle, *overlappingPos);
      restrictPathIndicesBackward(differenceCycle, *overlappingPos);
      workList.push_back(std::move(differenceCycle));

      // Create the cycle with the common indices. Reuse the original instance
      // to reduce the copy overhead.
      setEquationIndices(currentCycle[*overlappingPos], commonIndices);
      restrictPathIndicesForward(currentCycle, *overlappingPos);
      restrictPathIndicesBackward(currentCycle, *overlappingPos);
      workList.push_back(std::move(currentCycle));
    }
  }

  void setEquationIndices(PathDependency &node, const IndexSet &indices) const {
    node.equationIndices = indices;
    node.writtenVariableIndices =
        node.writeAccess.getAccessFunction().map(indices);
    node.readVariableIndices = node.readAccess.getAccessFunction().map(indices);
  }
};
} // namespace marco::modeling

#endif // MARCO_MODELING_CYCLES_H
