#ifndef MARCO_MODELING_SCC_H
#define MARCO_MODELING_SCC_H

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SCCIterator.h"
#include "marco/Modeling/AccessFunction.h"
#include "marco/Modeling/Dependency.h"
#include "marco/Modeling/Dumpable.h"
#include "marco/Modeling/Graph.h"
#include "marco/Modeling/IndexSet.h"
#include "marco/Modeling/MultidimensionalRange.h"
#include "marco/Utils/TreeOStream.h"
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
              swap(first.node, second.node);
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
        // TODO convert to iterative and improve documentation
        template<typename It>
        void addListIt(It step, It end)
        {
          if (step == end) {
            return;
          }

          if (auto next = std::next(step); next != end) {
            Container newIntervals;
            IndexSet range = step->getEquationIndexes();

            for (const auto& interval: intervals) {
              if (!range.overlaps(interval.getRange())) {
                newIntervals.push_back(interval);
                continue;
              }

              IndexSet restrictedRanges(interval.getRange());
              restrictedRanges -= range;

              for (const auto& restrictedRange: restrictedRanges) {
                newIntervals.emplace_back(restrictedRange, interval.getDestinations());
              }

              for (const MultidimensionalRange& intersectingRange : range.intersect(interval.getRange())) {
                range -= intersectingRange;

                llvm::ArrayRef<Dependency> dependencies = interval.getDestinations();
                std::vector<Dependency> newDependencies(dependencies.begin(), dependencies.end());

                auto& newDependency = newDependencies.emplace_back(
                    step->getRead(),
                    std::make_unique<FilteredEquation>(*graph, next->getEquation()));

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

            for (const auto& subRange: range) {
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
      using DependencyGraph = internal::VVarDependencyGraph<VariableProperty, EquationProperty>;

      using Variable = typename DependencyGraph::Variable;
      using Equation = typename DependencyGraph::Equation;

      using EquationDescriptor = typename DependencyGraph::EquationDescriptor;
      using AccessProperty = typename DependencyGraph::AccessProperty;
      using Access = typename DependencyGraph::Access;

      using WritesMap = typename DependencyGraph::WritesMap;

      using DFSStep = internal::scc::DFSStep<DependencyGraph, EquationDescriptor, Access>;
      using FilteredEquation = internal::scc::FilteredEquation<DependencyGraph, EquationDescriptor, Equation, Access>;
      using Cycle = FilteredEquation;

      CyclesFinder(llvm::ArrayRef<EquationProperty> equations) : vectorDependencyGraph(equations)
      {
      }

      std::vector<Cycle> getEquationsCycles() const
      {
        std::vector<Cycle> result;
        auto SCCs = vectorDependencyGraph.getSCCs();

        for (const auto& scc: SCCs) {
          auto writes = vectorDependencyGraph.getWritesMap(scc.begin(), scc.end());

          if (!scc.hasCycle()) {
            continue;
          }

          for (const auto& equationDescriptor : scc) {
            auto cycles = getEquationCyclicDependencies(writes, equationDescriptor);
            FilteredEquation dependencies(vectorDependencyGraph, equationDescriptor);

            for (const auto& cycle : cycles) {
              dependencies.addCyclicDependency(cycle);
            }

            result.push_back(std::move(dependencies));
          }
        }

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

            auto usedWritingEquationIndexes = inverseAccessIndexes(
                writingEquationIndexes,
                vectorDependencyGraph[writingEquation].getWrite().getAccessFunction(),
                intersection);

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
          // Restrict the flow (starting from the end).

          auto previousWriteAccessFunction = vectorDependencyGraph[equation].getWrite().getAccessFunction();
          auto previouslyWrittenIndexes = previousWriteAccessFunction.map(equationIndexes);

          for (auto it = path.rbegin(); it != path.rend(); ++it) {
            const auto& readAccessFunction = it->getRead().getAccessFunction();
            it->setEquationIndexes(inverseAccessIndexes(it->getEquationIndexes(), readAccessFunction, previouslyWrittenIndexes));

            previousWriteAccessFunction = vectorDependencyGraph[it->getEquation()].getWrite().getAccessFunction();
            previouslyWrittenIndexes = previousWriteAccessFunction.map(it->getEquationIndexes());
          }

          // Search along the restricted path if the candidate equation has already been visited with the same indexes
          auto step = llvm::find_if(path, [&](const DFSStep& step) {
            return step.getEquation() == equation && step.getEquationIndexes() == equationIndexes;
          });

          if (step == path.begin()) {
            // We have found a loop involving the variable defined by the first equation. This is the kind of loops
            // we are interested to find, so add it to the results.

            cyclicPaths.push_back(std::move(path));
            return true;
          }

          // Also search in the already identified cycles

          //auto cycle = llvm::find_if(cyclicPaths, [&](const std::list<DFSStep>& cycle) {

          //});

          // We have not found a loop for the variable of interest (that is, the one defined by the first equation),
          // but yet we can encounter loops among other equations. Thus, we need to identify them and stop traversing
          // the (infinite) tree.

          if (step != path.end()) {
            return true;
          }
        }

        return false;
      }

      /// Apply the inverse of an access function to a set of indices.
      /// If the access function is not invertible, then the inverse indexes are determined
      /// starting from a parent set.
      ///
      /// @param parentIndexes   parent index set
      /// @param accessFunction  access function to be inverted and applied (if possible)
      /// @param accessIndexes   indexes to be inverted
      /// @return indexes mapping to accessIndexes when accessFunction is applied to them
      IndexSet inverseAccessIndexes(
          const IndexSet& parentIndexes,
          const AccessFunction& accessFunction,
          const IndexSet& accessIndexes) const
      {
        if (accessFunction.isInvertible()) {
          auto mapped = accessFunction.inverseMap(accessIndexes);
          assert(accessFunction.map(mapped).contains(accessIndexes));
          return mapped;
        }

        // If the access function is not invertible, then not all the iteration variables are
        // used. This loss of information don't allow to reconstruct the equation ranges that
        // leads to the dependency loop. Thus, we need to iterate on all the original equation
        // points and determine which of them lead to a loop. This is highly expensive but also
        // inevitable, and confined only to very few cases within real scenarios.

        IndexSet result;

        for (const auto& range: parentIndexes) {
          for (const auto& point: range) {
            if (accessIndexes.contains(accessFunction.map(point))) {
              result += point;
            }
          }
        }

        return result;
      }

    private:
      DependencyGraph vectorDependencyGraph;
  };
}

#endif // MARCO_MODELING_SCC_H
