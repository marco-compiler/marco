#ifndef MARCO_MODELING_SCHEDULING_H
#define MARCO_MODELING_SCHEDULING_H

#include "marco/Modeling/AccessFunctionRotoTranslation.h"
#include "marco/Modeling/ArrayVariablesDependencyGraph.h"
#include "marco/Modeling/ScalarVariablesDependencyGraph.h"
#include "marco/Modeling/SCCsDependencyGraph.h"
#include <numeric>

namespace marco::modeling
{
  namespace internal::scheduling
  {
    template<typename VariableProperty, typename EquationProperty>
    class EquationView
    {
      public:
        using EquationTraits = typename ::marco::modeling::dependency
          ::EquationTraits<EquationProperty>;

        using Id = typename EquationTraits::Id;
        using VariableType = typename EquationTraits::VariableType;
        using AccessProperty = typename EquationTraits::AccessProperty;

        using Access = ::marco::modeling::dependency::Access<
            VariableType, AccessProperty>;

        EquationView(EquationProperty property, IndexSet indices)
            : realProperty(std::move(property)),
              indices(std::move(indices))
        {
        }

        EquationProperty& operator*()
        {
          return realProperty;
        }

        const EquationProperty& operator*() const
        {
          return realProperty;
        }

        Id getId() const
        {
          return EquationTraits::getId(&realProperty);
        }

        size_t getNumOfIterationVars() const
        {
          return EquationTraits::getNumOfIterationVars(&realProperty);
        }

        const IndexSet& getIterationRanges() const
        {
          return indices;
        }

        Access getWrite() const
        {
          return EquationTraits::getWrite(&realProperty);
        }

        std::vector<Access> getReads() const
        {
          return EquationTraits::getReads(&realProperty);
        }

      private:
        EquationProperty realProperty;
        IndexSet indices;
    };

    enum class DirectionPossibility
    {
      Any,
      Forward,
      Backward,
      Scalar
    };
  }

  namespace dependency
  {
    template<typename VP, typename EP>
    struct EquationTraits<internal::scheduling::EquationView<VP, EP>>
    {
      using EquationType = internal::scheduling::EquationView<VP, EP>;
      using Id = typename EquationTraits<EP>::Id;

      static Id getId(const EquationType* equation)
      {
        return (*equation).getId();
      }

      static size_t getNumOfIterationVars(const EquationType* equation)
      {
        return (*equation).getNumOfIterationVars();
      }

      static IndexSet getIterationRanges(const EquationType* equation)
      {
        return (*equation).getIterationRanges();
      }

      using VariableType = typename EquationTraits<EP>::VariableType;
      using AccessProperty = typename EquationTraits<EP>::AccessProperty;

      static Access<VariableType, AccessProperty> getWrite(
          const EquationType* equation)
      {
        return (*equation).getWrite();
      }

      static std::vector<Access<VariableType, AccessProperty>> getReads(
          const EquationType* equation)
      {
        return (*equation).getReads();
      }
    };
  }

  namespace scheduling
  {
    /// The direction to be used by the equations loop to update its iteration
    /// variable.
    enum class Direction
    {
      Forward,
      Backward,
      Unknown
    };

    template<typename EquationProperty>
    class ScheduledEquation
    {
      public:
        ScheduledEquation(
            EquationProperty property,
            IndexSet indices,
            llvm::ArrayRef<Direction> directions)
            : property(std::move(property)),
              indices(std::move(indices)),
              directions(directions.begin(), directions.end())
        {
        }

        const EquationProperty& getEquation() const
        {
          return property;
        }

        const IndexSet& getIndexes() const
        {
          return indices;
        }

        llvm::ArrayRef<Direction> getIterationDirections() const
        {
          return directions;
        }

      private:
        EquationProperty property;
        IndexSet indices;
        llvm::SmallVector<Direction> directions;
    };

    template<typename ElementType>
    class ScheduledSCC
    {
      private:
        using Container = std::vector<ElementType>;

      public:
        using const_iterator = typename Container::const_iterator;

        ScheduledSCC(llvm::ArrayRef<ElementType> equations, bool cycle)
          : equations(equations.begin(), equations.end()), cycle(cycle)
        {
          assert(!this->equations.empty());
        }

        const ElementType& operator[](size_t index) const
        {
          assert(index < equations.size());
          return equations[index];
        }

        bool hasCycle() const
        {
          return cycle;
        }

        const_iterator begin() const
        {
          return equations.begin();
        }

        const_iterator end() const
        {
          return equations.end();
        }

      private:
        Container equations;
        bool cycle;
    };
  }

  /// The scheduler allows to sort the equations in such a way that each scalar
  /// variable is determined before being accessed.
  /// The scheduling algorithm assumes that all the algebraic loops have
  /// already been resolved and the only possible kind of loop is the one given
  /// by an equation depending on itself (for example, x[i] = f(x[i - 1]), with
  /// i belonging to a range wider than one).
  template<typename VariableProperty, typename EquationProperty>
  class Scheduler
  {
    private:
      using EquationTraits =
          typename dependency::EquationTraits<EquationProperty>;

      using EquationView = internal::scheduling::EquationView<
          VariableProperty, EquationProperty>;

      using VectorDependencyGraph =
          ArrayVariablesDependencyGraph<VariableProperty, EquationView>;

      using Equation = typename VectorDependencyGraph::Equation;

      using SCC = typename VectorDependencyGraph::SCC;
      using SCCsGraph = SCCsDependencyGraph<SCC>;
      using SCCDescriptor = typename SCCsGraph::SCCDescriptor;
      using IndependentSCCs = std::vector<SCCDescriptor>;

      using ScalarDependencyGraph =
          ScalarVariablesDependencyGraph<VariableProperty, EquationView>;

      using ScheduledEquation =
          scheduling::ScheduledEquation<EquationProperty>;

      using ScheduledSCC =
          scheduling::ScheduledSCC<ScheduledEquation>;

      using ScheduledSCCsGroup = llvm::SmallVector<ScheduledSCC>;
      using Schedule = llvm::SmallVector<ScheduledSCCsGroup>;
      using DirectionPossibility = internal::scheduling::DirectionPossibility;

    private:
      static const int64_t kUnlimitedGroupElements = -1;
      mlir::MLIRContext* context;

    public:
      explicit Scheduler(mlir::MLIRContext* context)
        : context(context)
      {
      }

      mlir::MLIRContext* getContext() const
      {
        assert(context != nullptr);
        return context;
      }

      Schedule schedule(
          llvm::ArrayRef<EquationProperty> equations,
          int64_t maxGroupElements = kUnlimitedGroupElements) const
      {
        Schedule result;

        llvm::SmallVector<EquationView> equationViews;
        splitEquationIndices(equations, equationViews);

        VectorDependencyGraph vectorDependencyGraph(getContext());
        vectorDependencyGraph.addEquations(equationViews);

        SCCsGraph SCCsDependencyGraph;
        auto SCCs = vectorDependencyGraph.getSCCs();
        SCCsDependencyGraph.addSCCs(SCCs);

        auto scheduledSCCGroups =
            sortSCCs(SCCsDependencyGraph, maxGroupElements);

        for (const IndependentSCCs& sccGroup : scheduledSCCGroups) {
          auto& scheduledGroup = result.emplace_back();

          for (SCCDescriptor sccDescriptor : sccGroup) {
            const SCC& scc = SCCsDependencyGraph[sccDescriptor];

            if (scc.size() == 1) {
              llvm::SmallVector<DirectionPossibility, 3> directionPossibilities;
              getSchedulingDirections(scc, directionPossibilities);

              bool isSchedulableAsRange = llvm::all_of(
                  directionPossibilities,
                  [](DirectionPossibility direction) {
                    return direction == DirectionPossibility::Any ||
                        direction == DirectionPossibility::Forward ||
                        direction == DirectionPossibility::Backward;
                  });

              if (isSchedulableAsRange) {
                const auto& equation = scc[0];

                llvm::SmallVector<scheduling::Direction> directions;

                for (DirectionPossibility direction : directionPossibilities) {
                  if (direction == DirectionPossibility::Any ||
                      direction == DirectionPossibility::Forward) {
                    directions.push_back(scheduling::Direction::Forward);
                  } else if (direction == DirectionPossibility::Backward) {
                    directions.push_back(scheduling::Direction::Backward);
                  } else {
                    directions.push_back(scheduling::Direction::Unknown);
                  }
                }

                ScheduledEquation scheduledEquation(
                    *equation.getProperty(),
                    equation.getIterationRanges(),
                    directions);

                scheduledGroup.emplace_back(
                    std::move(scheduledEquation), false);

                continue;
              } else {
                // Mixed accesses detected. Scheduling is possible only on the
                // scalar equations.
                std::vector<EquationView> scalarEquationViews;

                for (const auto& equationDescriptor : scc) {
                  scalarEquationViews.push_back(
                      scc.getGraph()[equationDescriptor].getProperty());
                }

                ScalarDependencyGraph scalarDependencyGraph(getContext());
                scalarDependencyGraph.addEquations(scalarEquationViews);

                for (const auto& equationDescriptor :
                     scalarDependencyGraph.reversePostOrder()) {
                  const auto& scalarEquation =
                      scalarDependencyGraph[equationDescriptor];

                  llvm::SmallVector<scheduling::Direction> directions(
                      scalarEquation.getProperty().getNumOfIterationVars(),
                      scheduling::Direction::Forward);

                  ScheduledEquation scheduledEquation(
                      *scalarEquation.getProperty(),
                      IndexSet(MultidimensionalRange(scalarEquation.getIndex())),
                      directions);

                  scheduledGroup.emplace_back(
                      std::move(scheduledEquation), false);
                }
              }
            } else {
              // A strongly connected component can be scheduled with respect to
              // other SCCs, but the equations composing it are cyclic and thus
              // can't be scheduled.
              std::vector<ScheduledEquation> SCC;

              for (const auto& equationDescriptor : scc) {
                const auto& equation = scc[equationDescriptor];

                SCC.push_back(ScheduledEquation(
                    *equation.getProperty(),
                    equation.getIterationRanges(),
                    scheduling::Direction::Unknown));
              }

              scheduledGroup.emplace_back(std::move(SCC), true);
            }
          }
        }

        return result;
      }

    private:
      std::vector<IndependentSCCs> sortSCCs(
          const SCCsGraph& dependencyGraph,
          int64_t maxGroupElements) const
      {
        // Compute the in-degree of each node.
        llvm::DenseMap<SCCDescriptor, size_t> inDegrees;

        for (SCCDescriptor scc : llvm::make_range(
                 dependencyGraph.SCCsBegin(), dependencyGraph.SCCsEnd())) {
          inDegrees[scc] = 0;
        }

        for (SCCDescriptor node : llvm::make_range(
                 dependencyGraph.SCCsBegin(), dependencyGraph.SCCsEnd())) {
          for (SCCDescriptor child : llvm::make_range(
                   dependencyGraph.dependentSCCsBegin(node),
                   dependencyGraph.dependentSCCsEnd(node))) {
            if (node == child) {
              // Ignore self-loops.
              continue;
            }

            inDegrees[child]++;
          }
        }

        // Compute the sets of independent SCCs.
        std::vector<IndependentSCCs> result;
        std::set<SCCDescriptor> nodes;
        std::set<SCCDescriptor> newNodes;

        for (SCCDescriptor scc : llvm::make_range(
                 dependencyGraph.SCCsBegin(), dependencyGraph.SCCsEnd())) {
          if (inDegrees[scc] == 0) {
            nodes.insert(scc);
          }
        }

        while (!nodes.empty()) {
          std::vector<SCCDescriptor> independentSCCs;

          for (SCCDescriptor node : nodes) {
            if (inDegrees[node] == 0 &&
                (maxGroupElements == kUnlimitedGroupElements ||
                 independentSCCs.size() < maxGroupElements)) {
              independentSCCs.push_back(node);

              for (SCCDescriptor child : llvm::make_range(
                       dependencyGraph.dependentSCCsBegin(node),
                       dependencyGraph.dependentSCCsEnd(node))) {
                if (node == child) {
                  // Ignore self-loops.
                  continue;
                }

                assert(inDegrees[child] > 0);
                inDegrees[child]--;
                newNodes.insert(child);

                // Avoid visiting again the node at the next iteration.
                newNodes.erase(node);
              }
            } else {
              newNodes.insert(node);
            }
          }

          assert(!independentSCCs.empty());
          result.push_back(std::move(independentSCCs));

          nodes = std::move(newNodes);
          newNodes.clear();
        }

        return result;
      }

      /// Split the equation indices so that the accesses to the written
      /// variable either are the same of the written indices, or do not
      /// overlap at all.
      void splitEquationIndices(
        llvm::ArrayRef<EquationProperty> equations,
        llvm::SmallVectorImpl<EquationView>& equationViews) const
      {
        for (const EquationProperty& equation : equations) {
          IndexSet equationIndices =
              EquationTraits::getIterationRanges(&equation);

          auto writeAccess = EquationTraits::getWrite(&equation);

          const AccessFunction& writeAccessFunction =
              writeAccess.getAccessFunction();

          auto writtenVariable = writeAccess.getVariable();
          IndexSet writtenIndices = writeAccessFunction.map(equationIndices);

          // The written indices for which there is no overlap with other
          // accesses.
          llvm::SmallVector<IndexSet, 10> partitions;
          partitions.push_back(equationIndices);

          for (const auto& readAccess : EquationTraits::getReads(&equation)) {
            const AccessFunction& readAccessFunction =
                readAccess.getAccessFunction();

            auto readVariable = readAccess.getVariable();

            if (readVariable != writtenVariable) {
              continue;
            }

            IndexSet readIndices = readAccessFunction.map(equationIndices);

            // Restrict the read indices to the written ones.
            readIndices = readIndices.intersect(writtenIndices);

            if (readIndices.empty()) {
              // There is no overlap, so there's no need to split the
              // indices of the equation.
              continue;
            }

            // The indices of the equation that lead to a write to the
            // variables accessed by the read operation.
            IndexSet writingEquationIndices =
                writeAccessFunction.inverseMap(readIndices, equationIndices);

            // Determine the new partitions.
            llvm::SmallVector<IndexSet> newPartitions;

            for (IndexSet& partition : partitions) {
              IndexSet intersection =
                  partition.intersect(writingEquationIndices);

              if (intersection.empty()) {
                newPartitions.push_back(std::move(partition));
              } else {
                IndexSet diff = partition - intersection;

                if (!diff.empty()) {
                  newPartitions.push_back(std::move(diff));
                }

                newPartitions.push_back(std::move(intersection));
              }
            }

            partitions = std::move(newPartitions);
          }

          // Check that the split indices do represent exactly the initial
          // ones.
          assert(std::accumulate(
                     partitions.begin(), partitions.end(),
                     IndexSet()) == equationIndices);

          // Create the views.
          for (IndexSet& partition : partitions) {
            equationViews.push_back(
                EquationView(equation, std::move(partition)));
          }
        }
      }

      /// Given a SSC containing only one equation that may depend on itself,
      /// determine the access direction with respect to the variable that is
      /// written by the equation.
      void getSchedulingDirections(
          const SCC& scc,
          llvm::SmallVectorImpl<DirectionPossibility>& directions) const
      {
        if (!scc.hasCycle()) {
          // If there is no cycle, then the iteration direction of the equation
          // is irrelevant. We prefer the forward direction for simplicity.

          directions.append(
              scc[0].getNumOfIterationVars(),
              DirectionPossibility::Forward);

          return;
        }

        // If all the dependencies have the same direction, then we can set
        // the induction variable to increase or decrease accordingly.

        const Equation& equation = scc[0];
        auto equationRange = equation.getIterationRanges();
        const auto& write = equation.getWrite();
        const auto& writtenVariable = write.getVariable();
        const AccessFunction& writeAccessFunction = write.getAccessFunction();
        IndexSet writtenIndices(writeAccessFunction.map(equationRange));

        for (const auto& read : equation.getReads()) {
          // The access is considered only if it reads the same variable it is
          // being defined by the equation and the ranges overlap.
          const auto& readVariable = read.getVariable();

          if (writtenVariable != readVariable) {
            continue;
          }

          const AccessFunction& readAccessFunction = read.getAccessFunction();
          IndexSet readIndices(readAccessFunction.map(equationRange));

          if (!readIndices.overlaps(writtenIndices)) {
            continue;
          }

          llvm::SmallVector<DirectionPossibility, 3> accessDirections;
          auto inverseWriteAccess = writeAccessFunction.inverse();

          if (inverseWriteAccess) {
            auto relativeAccess =
                inverseWriteAccess->combine(readAccessFunction);

            getAccessFunctionDirections(*relativeAccess, accessDirections);
          } else {
            accessDirections.append(
                scc[0].getNumOfIterationVars(),
                DirectionPossibility::Scalar);
          }

          if (directions.empty()) {
            directions = std::move(accessDirections);
          } else {
            mergeDirectionPossibilities(directions, accessDirections);
          }

          // If all the inductions have been scalarized, then the situation
          // can't get better anymore.
          if (llvm::all_of(directions, [](DirectionPossibility direction) {
                return direction == DirectionPossibility::Scalar;
              })) {
            return;
          }
        }
      }

      /// Get the access direction of an access function.
      /// For example, an access consisting in [i0 + 1][i1 + 2] has a forward
      /// direction, meaning that it requires variables that will be defined
      /// later in the loop execution. A [i0 - 1][i1 -2] access function has a
      /// backward direction and a [i0 + 1][i1 - 2] has a mixed one.
      /// The indices of the above induction variables refer to the order in
      /// which the induction variables have been defined, meaning that i0 is
      /// the outer-most induction, i1 the second outer-most one, etc.
      void getAccessFunctionDirections(
          const AccessFunction& accessFunction,
          llvm::SmallVectorImpl<DirectionPossibility>& directions) const
      {
        if (auto rotoTranslation =
                accessFunction.dyn_cast<AccessFunctionRotoTranslation>()) {
          return getAccessFunctionDirections(*rotoTranslation, directions);
        }

        directions.append(
            accessFunction.getNumOfDims(), DirectionPossibility::Scalar);
      }

      void getAccessFunctionDirections(
          const AccessFunctionRotoTranslation& accessFunction,
          llvm::SmallVectorImpl<DirectionPossibility>& directions) const
      {
        if (accessFunction.isIdentityLike()) {
          // Examine the offset of the individual dimension access.
          for (size_t i = 0, e = accessFunction.getNumOfResults(); i < e; ++i) {
            auto offset = accessFunction.getOffset(i);

            if (offset == 0) {
              directions.push_back(DirectionPossibility::Any);
            } else if (offset > 0) {
              directions.push_back(DirectionPossibility::Backward);
            } else {
              directions.push_back(DirectionPossibility::Forward);
            }
          }
        } else {
          // If the iteration indices are out of order, then some accesses will
          // refer to future written variables and others to past written ones.

          directions.append(
              accessFunction.getNumOfDims(), DirectionPossibility::Scalar);
        }
      }

      void mergeDirectionPossibilities(
          llvm::SmallVectorImpl<DirectionPossibility>& result,
          llvm::ArrayRef<DirectionPossibility> newDirections) const
      {
        assert(result.size() == newDirections.size());

        for (size_t i = 0, e = result.size(); i < e; ++i) {
          result[i] = mergeDirectionPossibilities(result[i], newDirections[i]);
        }
      }

      DirectionPossibility mergeDirectionPossibilities(
          DirectionPossibility lhs, DirectionPossibility rhs) const
      {
        if (lhs == DirectionPossibility::Scalar ||
            rhs == DirectionPossibility::Scalar) {
          return DirectionPossibility::Scalar;
        }

        if (lhs == DirectionPossibility::Any) {
          return rhs;
        }

        if (rhs == DirectionPossibility::Any) {
          return lhs;
        }

        if (lhs != rhs) {
          return DirectionPossibility::Scalar;
        }

        assert(lhs == rhs);
        return lhs;
      }
  };
}

#endif // MARCO_MODELING_SCHEDULING_H
