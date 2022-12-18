#ifndef MARCO_MODELING_CYCLES_H
#define MARCO_MODELING_CYCLES_H

#include "marco/Diagnostic/TreeOStream.h"
#include "marco/Modeling/AccessFunction.h"
#include "marco/Modeling/Dependency.h"
#include "marco/Modeling/Dumpable.h"
#include "marco/Modeling/Graph.h"
#include "marco/Modeling/IndexSet.h"
#include "marco/Modeling/MultidimensionalRange.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Support/ThreadPool.h"
#include <list>
#include <stack>

namespace marco::modeling
{
  namespace internal::scc
  {
    template<typename Graph, typename EquationDescriptor, typename Access>
    class DFSStep : public Dumpable
    {
      public:
        DFSStep(const Graph& graph, EquationDescriptor equation, IndexSet equationIndexes, Access read)
          : graph(&graph),
            equation(std::move(equation)),
            equationIndexes(std::move(equationIndexes)),
            read(std::move(read))
        {
        }

        using Dumpable::dump;

        void dump(std::ostream& stream) const override
        {
          using namespace marco::utils;

          TreeOStream os(stream);
          os << "DFS step\n";
          os << tree_property << "Written variable: " << (*graph)[equation].getWrite().getVariable() << "\n";
          os << tree_property << "Writing equation: " << (*graph)[equation].getId() << "\n";
          os << tree_property << "Filtered equation indexes: " << equationIndexes << "\n";
          os << tree_property << "Read access: " << read.getAccessFunction() << "\n";
        }

        EquationDescriptor getEquation() const
        {
          return equation;
        }

        const IndexSet& getEquationIndexes() const
        {
          return equationIndexes;
        }

        void setEquationIndexes(IndexSet indexes)
        {
          equationIndexes = std::move(indexes);
        }

        const Access& getRead() const
        {
          return read;
        }

      private:
        const Graph* graph;
        EquationDescriptor equation;
        IndexSet equationIndexes;
        Access read;
    };

    template<
        typename Graph,
        typename EquationDescriptor,
        typename Equation,
        typename Access>
    class FilteredEquation : public Dumpable
    {
      private:
        class Dependency : public Dumpable
        {
          public:
            Dependency(Access access, std::unique_ptr<FilteredEquation> equation)
              : access(std::move(access)), equation(std::move(equation))
            {
            }

            Dependency(const Dependency& other)
                : access(other.access), equation(std::make_unique<FilteredEquation>(*other.equation))
            {
            }

            Dependency(Dependency&& other) = default;

            ~Dependency() = default;

            friend void swap(Dependency& first, Dependency& second)
            {
              using std::swap;
              swap(first.access, second.access);
              swap(first.equation, second.equation);
            }

            Dependency& operator=(const Dependency& other)
            {
              Dependency result(other);
              swap(*this, result);
              return *this;
            }

            using Dumpable::dump;

            void dump(std::ostream& stream) const override
            {
              using namespace marco::utils;

              TreeOStream os(stream);
              os << "Access function: " << access.getAccessFunction() << "\n";
              os << tree_property;
              equation->dump(os);
            }

            const Access& getAccess() const
            {
              return access;
            }

            FilteredEquation& getNode()
            {
              assert(equation != nullptr);
              return *equation;
            }

            const FilteredEquation& getNode() const
            {
              assert(equation != nullptr);
              return *equation;
            }

          private:
            Access access;
            std::unique_ptr<FilteredEquation> equation;
        };

        class Interval : public Dumpable
        {
          public:
            using Container = std::vector<Dependency>;

          public:
            Interval(MultidimensionalRange range, llvm::ArrayRef<Dependency> destinations = llvm::None)
                : range(std::move(range)), destinations(destinations.begin(), destinations.end())
            {
            }

            Interval(MultidimensionalRange range, Access access, std::unique_ptr<FilteredEquation> destination)
                : range(std::move(range))
            {
              destinations.emplace_back(std::move(access), std::move(destination));
            }

            using Dumpable::dump;

            void dump(std::ostream& stream) const override
            {
              using namespace marco::utils;

              TreeOStream os(stream);
              os << tree_property << "Indexes: " << range << "\n";

              for (const auto& destination : destinations) {
                os << tree_property;
                destination.dump(os);
                os << "\n";
              }
            }

            const MultidimensionalRange& getRange() const
            {
              return range;
            }

            llvm::ArrayRef<Dependency> getDestinations() const
            {
              return destinations;
            }

            void addDestination(Access access, std::unique_ptr<FilteredEquation> destination)
            {
              destinations.emplace_back(std::move(access), std::move(destination));
            }

          private:
            MultidimensionalRange range;
            Container destinations;
        };

        using Container = std::vector<Interval>;

      public:
        using const_iterator = typename Container::const_iterator;

        FilteredEquation(const Graph& graph, EquationDescriptor equation)
            : graph(&graph), equation(std::move(equation))
        {
        }

        using Dumpable::dump;

        void dump(std::ostream& stream) const override
        {
          using namespace marco::utils;

          TreeOStream os(stream);
          os << "Equation (" <<  (*graph)[equation].getId() << ")\n";

          for (const auto& interval : intervals) {
            os << tree_property;
            interval.dump(os);
          }
        }

        /// Get the equation property that is provided by the user.
        /// The method has been created to serve as an API.
        ///
        /// @return equation property
        const typename Equation::Property& getEquation() const
        {
          return (*graph)[equation].getProperty();
        }

        const_iterator begin() const
        {
          return intervals.begin();
        }

        const_iterator end() const
        {
          return intervals.end();
        }

        void addCyclicDependency(const std::list<DFSStep<Graph, EquationDescriptor, Access>>& steps)
        {
          addListIt(steps.begin(), steps.end());
        }

      private:
        template<typename It>
        void addListIt(It step, It end)
        {
          if (step == end) {
            // The whole cycle has been traversed
            return;
          }

          auto next = std::next(step);

          if (next == end) {
            IndexSet newIntervals(step->getEquationIndexes());

            for (const auto& interval : intervals) {
              newIntervals -= interval.getRange();
            }

            for (const auto& range : llvm::make_range(newIntervals.rangesBegin(), newIntervals.rangesEnd())) {
              intervals.push_back(Interval(range, llvm::None));
            }
          } else {
            Container newIntervals;
            IndexSet range = step->getEquationIndexes();

            for (const auto& interval: intervals) {
              if (!range.overlaps(interval.getRange())) {
                newIntervals.push_back(interval);
                continue;
              }

              IndexSet restrictedRanges(interval.getRange());
              restrictedRanges -= range;

              for (const auto& restrictedRange: llvm::make_range(restrictedRanges.rangesBegin(), restrictedRanges.rangesEnd())) {
                newIntervals.emplace_back(restrictedRange, interval.getDestinations());
              }

              auto intersectingRanges = range.intersect(interval.getRange());

              for (const MultidimensionalRange& intersectingRange : llvm::make_range(intersectingRanges.rangesBegin(), intersectingRanges.rangesEnd())) {
                range -= intersectingRange;

                llvm::ArrayRef<Dependency> dependencies = interval.getDestinations();
                std::vector<Dependency> newDependencies(dependencies.begin(), dependencies.end());

                auto& newDependency = newDependencies.emplace_back(
                    step->getRead(),
                    std::make_unique<FilteredEquation>(*graph, next->getEquation()));

                //for (const auto& nextEquationRange : next->getEquationIndexes()) {
                //  newDependency.getNode().intervals.push_back(Interval(nextEquationRange, llvm::None));
                //}

                newDependency.getNode().addListIt(next, end);

                /*
                auto dependency = llvm::find_if(newDependencies, [&](const Dependency& dependency) {
                  return dependency.getNode().equation == step->getEquation();
                });

                if (dependency == newDependencies.end()) {
                  auto& newDependency = newDependencies.emplace_back(step->getRead(), std::make_unique<FilteredEquation>(*graph, next->getEquation()));
                  newDependency.getNode().addListIt(next, end);
                } else {
                  dependency->getNode().addListIt(next, end);
                }
                 */

                Interval newInterval(intersectingRange, newDependencies);
                newIntervals.push_back(std::move(newInterval));
              }
            }

            for (const auto& subRange: llvm::make_range(range.rangesBegin(), range.rangesEnd())) {
              std::vector<Dependency> dependencies;

              auto& dependency = dependencies.emplace_back(
                  step->getRead(),
                  std::make_unique<FilteredEquation>(*graph, next->getEquation()));

              dependency.getNode().addListIt(next, end);
              newIntervals.emplace_back(subRange, dependencies);
            }

            intervals = std::move(newIntervals);
          }
        }

      private:
        const Graph* graph;
        EquationDescriptor equation;
        Container intervals;
    };
  }
}

namespace marco::modeling
{
  template<typename VariableProperty, typename EquationProperty>
  class CyclesFinder
  {
    public:
      using DependencyGraph = ArrayVariablesDependencyGraph<VariableProperty, EquationProperty>;

      using Variable = typename DependencyGraph::Variable;
      using Equation = typename DependencyGraph::Equation;

      using EquationDescriptor = typename DependencyGraph::EquationDescriptor;
      using AccessProperty = typename DependencyGraph::AccessProperty;
      using Access = typename DependencyGraph::Access;

      using WritesMap = typename DependencyGraph::WritesMap;

      using DFSStep = internal::scc::DFSStep<DependencyGraph, EquationDescriptor, Access>;
      using FilteredEquation = internal::scc::FilteredEquation<DependencyGraph, EquationDescriptor, Equation, Access>;
      using Cycle = FilteredEquation;

      CyclesFinder(bool includeSecondaryCycles = true)
        : includeSecondaryCycles(includeSecondaryCycles)
      {
      }

      void addEquations(llvm::ArrayRef<EquationProperty> equations)
      {
        vectorDependencyGraph.addEquations(equations);
      }

      std::vector<Cycle> getEquationsCycles() const
      {
        llvm::ThreadPool threadPool;
        unsigned int numOfThreads = threadPool.getThreadCount();

        std::vector<Cycle> result;
        std::mutex resultMutex;

        auto SCCs = vectorDependencyGraph.getSCCs();
        size_t numOfSCCs = SCCs.size();
        std::atomic_size_t currentSCC = 0;

        auto processFn = [&]() {
          size_t i = currentSCC++;

          while (i < numOfSCCs) {
            const auto& scc = SCCs[i];
            auto writes = vectorDependencyGraph.getWritesMap(scc.begin(), scc.end());

            if (scc.hasCycle()) {
              for (const EquationDescriptor& equationDescriptor : scc) {
                auto cycles = getEquationCyclicDependencies(writes, equationDescriptor);
                FilteredEquation dependencies(vectorDependencyGraph, equationDescriptor);

                for (const auto& cycle : cycles) {
                  dependencies.addCyclicDependency(cycle);
                }

                std::lock_guard<std::mutex> lockGuard(resultMutex);
                result.push_back(std::move(dependencies));
              }
            }

            i = currentSCC++;
          }
        };

        for (unsigned int i = 0; i < numOfThreads; ++i) {
          threadPool.async(processFn);
        }

        threadPool.wait();
        return result;
      }

    private:
      std::vector<std::list<DFSStep>> getEquationCyclicDependencies(
          const WritesMap& writes,
          EquationDescriptor equation) const
      {
        std::vector<std::list<DFSStep>> cyclicPaths;
        std::stack<std::list<DFSStep>> stack;

        // The first equation starts with the full range, as it has no predecessors
        IndexSet indexes(vectorDependencyGraph[equation].getIterationRanges());

        std::list<DFSStep> emptyPath;

        for (auto& extendedPath : appendReads(emptyPath, equation, indexes)) {
          stack.push(std::move(extendedPath));
        }

        while (!stack.empty()) {
          auto& path = stack.top();

          std::vector<std::list<DFSStep>> extendedPaths;

          const auto& equationIndexes = path.back().getEquationIndexes();
          const auto& read = path.back().getRead();
          const auto& accessFunction = read.getAccessFunction();
          auto readIndexes = accessFunction.map(equationIndexes);

          // Get the equations writing into the read variable
          auto writeInfos = writes.equal_range(read.getVariable());

          for (const auto& [variableId, writeInfo] : llvm::make_range(writeInfos.first, writeInfos.second)) {
            const auto& writtenIndexes = writeInfo.getWrittenVariableIndexes();

            // If the ranges do not overlap, then there is no loop involving the writing equation
            if (!readIndexes.overlaps(writtenIndexes)) {
              continue;
            }

            auto intersection = readIndexes.intersect(writtenIndexes);
            EquationDescriptor writingEquation = writeInfo.getEquation();
            IndexSet writingEquationIndexes(vectorDependencyGraph[writingEquation].getIterationRanges());

            const AccessFunction& writingEquationAccessFunction = vectorDependencyGraph[writingEquation].getWrite().getAccessFunction();
            auto usedWritingEquationIndexes = writingEquationAccessFunction.inverseMap(intersection, writingEquationIndexes);

            if (detectLoop(cyclicPaths, path, writingEquation, usedWritingEquationIndexes)) {
              // Loop detected. It may either be a loop regarding the first variable or not. In any case, we should
              // stop visiting the tree, which would be infinite.
              continue;
            }

            for (auto& extendedPath : appendReads(path, writingEquation, usedWritingEquationIndexes))
              extendedPaths.push_back(std::move(extendedPath));
          }

          stack.pop();

          for (auto& extendedPath : extendedPaths) {
            stack.push(std::move(extendedPath));
          }
        }

        return cyclicPaths;
      }

      std::vector<std::list<DFSStep>> appendReads(
          const std::list<DFSStep>& path,
          EquationDescriptor equation,
          const IndexSet& equationRange) const
      {
        std::vector<std::list<DFSStep>> result;

        for (const Access& read : vectorDependencyGraph[equation].getReads()) {
          std::list<DFSStep> extendedPath = path;
          extendedPath.emplace_back(vectorDependencyGraph, equation, equationRange, read);
          result.push_back(std::move(extendedPath));
        }

        return result;
      }

      /// Detect whether adding a new equation with a given range would lead to a loop.
      /// The path to be check is intentionally passed by copy, as its flow is restricted depending on the
      /// equation to be added and such modification must not interfere with other paths.
      ///
      /// @param cyclicPaths      cyclic paths results list
      /// @param path             candidate path
      /// @param graph            graph to which the equations belong to
      /// @param equation         equation that should be added to the path
      /// @param equationIndexes  indexes of the equation to be added
      /// @return true if the candidate equation would create a loop when added to the path; false otherwise
      bool detectLoop(
          std::vector<std::list<DFSStep>>& cyclicPaths,
          std::list<DFSStep> path,
          EquationDescriptor equation,
          const IndexSet& equationIndexes) const
      {
        if (!path.empty()) {
          // Restrict the flow (starting from the end)

          auto previousWriteAccessFunction = vectorDependencyGraph[equation].getWrite().getAccessFunction();
          auto previouslyWrittenIndexes = previousWriteAccessFunction.map(equationIndexes);

          for (auto it = path.rbegin(); it != path.rend(); ++it) {
            const auto& readAccessFunction = it->getRead().getAccessFunction();
            it->setEquationIndexes(readAccessFunction.inverseMap(previouslyWrittenIndexes, it->getEquationIndexes()));

            previousWriteAccessFunction = vectorDependencyGraph[it->getEquation()].getWrite().getAccessFunction();
            previouslyWrittenIndexes = previousWriteAccessFunction.map(it->getEquationIndexes());
          }

          // Search along the restricted path if the candidate equation has already been visited with the same indexes
          auto step = llvm::find_if(path, [&](const DFSStep& step) {
            return step.getEquation() == equation && step.getEquationIndexes().contains(equationIndexes);
          });

          if (step == path.begin()) {
            // We have found a loop involving the variable defined by the first equation. This is the kind of loops
            // we are interested to find, so add it to the results.

            cyclicPaths.push_back(std::move(path));
            return true;
          }

          // We have not found a loop for the variable of interest (that is, the one defined by the first equation),
          // but yet we can encounter loops among other equations. Thus, we need to identify them and stop traversing
          // the (infinite) tree.

          if (step != path.end()) {
            if (includeSecondaryCycles) {
              cyclicPaths.push_back(std::move(path));
            }

            return true;
          }
        }

        return false;
      }

    private:
      DependencyGraph vectorDependencyGraph;
      bool includeSecondaryCycles;
  };
}

#endif // MARCO_MODELING_CYCLES_H
