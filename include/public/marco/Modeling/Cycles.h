#ifndef MARCO_MODELING_CYCLES_H
#define MARCO_MODELING_CYCLES_H

#include "marco/Diagnostic/TreeOStream.h"
#include "marco/Modeling/ArrayVariablesDependencyGraph.h"
#include "marco/Modeling/Dumpable.h"
#include "marco/Modeling/IndexSet.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/SCCIterator.h"
#include <list>

namespace marco::modeling
{
  namespace dependency
  {
    namespace impl {
      template<typename EquationDescriptor, typename Access>
      class PathDependency
      {
        public:
          PathDependency(
            EquationDescriptor equation,
            IndexSet equationIndices,
            Access writeAccess,
            IndexSet writtenVariableIndices,
            Access readAccess,
            IndexSet readVariableIndices)
              : equation(equation),
                equationIndices(std::move(equationIndices)),
                writeAccess(std::move(writeAccess)),
                writtenVariableIndices(std::move(writtenVariableIndices)),
                readAccess(std::move(readAccess)),
                readVariableIndices(std::move(readVariableIndices))
          {
          }

          EquationDescriptor equation;
          IndexSet equationIndices;
          Access writeAccess;
          IndexSet writtenVariableIndices;
          Access readAccess;
          IndexSet readVariableIndices;
      };

      template<typename EquationDescriptor, typename Access>
      class Path
      {
        private:
          using Dependency = PathDependency<EquationDescriptor, Access>;
          using Container = std::list<Dependency>;

        public:
          using iterator = typename Container::iterator;
          using const_iterator = typename Container::const_iterator;

          using reverse_iterator = typename Container::reverse_iterator;

          using const_reverse_iterator =
              typename Container::const_reverse_iterator;

          size_t size() const
          {
            return equations.size();
          }

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

          reverse_iterator rbegin()
          {
            return equations.rbegin();
          }

          const_reverse_iterator rbegin() const
          {
            return equations.rbegin();
          }

          reverse_iterator rend()
          {
            return equations.rend();
          }

          const_reverse_iterator rend() const
          {
            return equations.rend();
          }

          Dependency& back()
          {
            return equations.back();
          }

          const Dependency& back() const
          {
            return equations.back();
          }

          Path operator+(Dependency equation)
          {
            Path result(*this);
            result += std::move(equation);
            return result;
          }

          Path& operator+=(Dependency equation)
          {
            equations.push_back(std::move(equation));
            return *this;
          }

          Path withoutLast(size_t n) const
          {
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
    }

    template<typename EquationProperty, typename Access>
    class CyclicEquation
    {
      public:
        CyclicEquation(
            EquationProperty equation,
            IndexSet equationIndices,
            Access writeAccess,
            IndexSet writtenVariableIndices,
            Access readAccess,
            IndexSet readVariableIndices)
            : equation(equation),
              equationIndices(std::move(equationIndices)),
              writeAccess(std::move(writeAccess)),
              writtenVariableIndices(std::move(writtenVariableIndices)),
              readAccess(std::move(readAccess)),
              readVariableIndices(std::move(readVariableIndices))
        {
        }

        EquationProperty equation;
        IndexSet equationIndices;
        Access writeAccess;
        IndexSet writtenVariableIndices;
        Access readAccess;
        IndexSet readVariableIndices;
    };

    template<typename EquationProperty, typename Access>
    class Cycle
    {
      private:
        using Equation = CyclicEquation<EquationProperty, Access>;
        using Container = llvm::SmallVector<Equation, 3>;

      public:
        using iterator = typename Container::iterator;
        using const_iterator = typename Container::const_iterator;

        using reverse_iterator = typename Container::reverse_iterator;

        using const_reverse_iterator =
            typename Container::const_reverse_iterator;

        Cycle(llvm::ArrayRef<Equation> equations)
        {
          for (const Equation& equation : equations) {
            this->equations.push_back(equation);
          }
        }

        size_t size() const
        {
          return equations.size();
        }

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

        reverse_iterator rbegin()
        {
          return equations.rbegin();
        }

        const_reverse_iterator rbegin() const
        {
          return equations.rbegin();
        }

        reverse_iterator rend()
        {
          return equations.rend();
        }

        const_reverse_iterator rend() const
        {
          return equations.rend();
        }

      private:
        Container equations;
    };
  }
}

namespace marco::modeling
{
  template<typename VariableProperty, typename EquationProperty>
  class CyclesFinder
  {
    public:
      using DependencyGraph = ArrayVariablesDependencyGraph<
          VariableProperty, EquationProperty>;

      using Variable = typename DependencyGraph::Variable;
      using Equation = typename DependencyGraph::Equation;

      using EquationDescriptor = typename DependencyGraph::EquationDescriptor;
      using AccessProperty = typename DependencyGraph::AccessProperty;
      using Access = typename DependencyGraph::Access;

      using WritesMap = typename DependencyGraph::WritesMap;

      using PathDependency =
          dependency::impl::PathDependency<EquationDescriptor, Access>;

      using Path = dependency::impl::Path<EquationDescriptor, Access>;

      using CyclicEquation =
          dependency::CyclicEquation<EquationProperty, Access>;

      using Cycle = dependency::Cycle<EquationProperty, Access>;

      CyclesFinder(mlir::MLIRContext* context)
          : context(context),
            vectorDependencyGraph(context)
      {
      }

      mlir::MLIRContext* getContext() const
      {
        assert(context != nullptr);
        return context;
      }

      void addEquations(llvm::ArrayRef<EquationProperty> equations)
      {
        vectorDependencyGraph.addEquations(equations);
      }

      std::vector<Cycle> getEquationsCycles() const
      {
        std::vector<Cycle> result;
        std::mutex resultMutex;

        auto SCCs = vectorDependencyGraph.getSCCs();

        auto processFn = [&](const typename DependencyGraph::SCC& scc) {
          auto writesMap =
              vectorDependencyGraph.getWritesMap(scc.begin(), scc.end());

          if (scc.hasCycle()) {
            for (const EquationDescriptor& equationDescriptor : scc) {
              llvm::SmallVector<Path> paths;
              getEquationsCycles(paths, writesMap, equationDescriptor);

              std::lock_guard<std::mutex> lockGuard(resultMutex);

              for (Path& path : paths) {
                llvm::SmallVector<CyclicEquation, 3> cyclicEquations;

                for (PathDependency& dependency : path) {
                  cyclicEquations.push_back(CyclicEquation(
                      vectorDependencyGraph[dependency.equation].getProperty(),
                      std::move(dependency.equationIndices),
                      std::move(dependency.writeAccess),
                      std::move(dependency.writtenVariableIndices),
                      std::move(dependency.readAccess),
                      std::move(dependency.readVariableIndices)));
                }

                result.emplace_back(Cycle(cyclicEquations));
              }
            }
          }
        };

        mlir::parallelForEach(getContext(), SCCs, processFn);
        return result;
      }

    private:
      void getEquationsCycles(
          llvm::SmallVectorImpl<Path>& cycles,
          const WritesMap& writesMap,
          EquationDescriptor equation) const
      {
        // The first equation starts with the full range, as it has no
        // predecessors.
        IndexSet equationIndices(
            vectorDependencyGraph[equation].getIterationRanges());

        getEquationsCycles(cycles, writesMap, equation, equationIndices, {});
      }

      void getEquationsCycles(
          llvm::SmallVectorImpl<Path>& cycles,
          const WritesMap& writesMap,
          EquationDescriptor equation,
          const IndexSet& equationIndices,
          Path path) const
      {
        const Access& currentEquationWriteAccess =
            vectorDependencyGraph[equation].getWrite();

        IndexSet currentEquationWrittenIndices =
            currentEquationWriteAccess.getAccessFunction()
                .map(equationIndices);

        for (const Access& readAccess :
             vectorDependencyGraph[equation].getReads()) {
          const auto& accessFunction = readAccess.getAccessFunction();
          auto readVariableIndices = accessFunction.map(equationIndices);

          auto writingEquations = writesMap.equal_range(readAccess.getVariable());

          for (const auto& [variableId, writeInfo] : llvm::make_range(
                   writingEquations.first, writingEquations.second)) {
            const IndexSet& writtenVariableIndices =
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
                vectorDependencyGraph[writingEquation].getIterationRanges());

            Access writingEquationWriteAccess =
                vectorDependencyGraph[writingEquation].getWrite();

            const AccessFunction& writingEquationAccessFunction =
                writingEquationWriteAccess.getAccessFunction();

            IndexSet usedWritingEquationIndices =
                writingEquationAccessFunction.inverseMap(
                    variableIndicesIntersection, allWritingEquationIndices);

            Path extendedPath = path +
                PathDependency(equation, equationIndices,
                               currentEquationWriteAccess,
                               currentEquationWrittenIndices,
                               readAccess, readVariableIndices);

            restrictPathIndices(extendedPath);

            if (auto pathLength = extendedPath.size(); pathLength > 1) {
              // Search along the restricted path if the current equation has
              // already been visited with some of the current indices.
              auto dependencyIt = llvm::find_if(
                  extendedPath, [&](const PathDependency& dependency) {
                    if (dependency.equation != writingEquation) {
                      return false;
                    }

                    return dependency.equationIndices.overlaps(
                        usedWritingEquationIndices);
                  });

              if (dependencyIt == extendedPath.begin()) {
                cycles.push_back(extendedPath);
                continue;
              }

              if (dependencyIt != extendedPath.end()) {
                // Sub-cycle detected.
                continue;
              }
            }

            getEquationsCycles(cycles, writesMap,
                               writingEquation, usedWritingEquationIndices,
                               extendedPath);
          }
        }
      }

      void restrictPathIndices(Path& path) const
      {
        auto it = path.rbegin();
        auto endIt = path.rend();

        if (it == endIt) {
          return;
        }

        auto prevIt = it;

        while (++it != endIt) {
          it->readVariableIndices = prevIt->writtenVariableIndices;
          const Access& readAccess = it->readAccess;

          it->equationIndices = readAccess.getAccessFunction().inverseMap(
              it->readVariableIndices, it->equationIndices);

          it->writtenVariableIndices =
              it->writeAccess.getAccessFunction().map(it->equationIndices);

          prevIt = it;
        }
      }

    private:
      mlir::MLIRContext* context;
      DependencyGraph vectorDependencyGraph;
  };
}

#endif // MARCO_MODELING_CYCLES_H
