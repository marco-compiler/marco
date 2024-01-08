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
            : property(std::move(property)),
              indices(std::move(indices))
        {
        }

        EquationProperty& operator*()
        {
          return property;
        }

        const EquationProperty& operator*() const
        {
          return property;
        }

        Id getId() const
        {
          // Forward the request to the traits class of the property.
          return EquationTraits::getId(&property);
        }

        size_t getNumOfIterationVars() const
        {
          // Forward the request to the traits class of the property.
          return EquationTraits::getNumOfIterationVars(&property);
        }

        const IndexSet& getIterationRanges() const
        {
          return indices;
        }

        Access getWrite() const
        {
          // Forward the request to the traits class of the property.
          return EquationTraits::getWrite(&property);
        }

        std::vector<Access> getReads() const
        {
          // Forward the request to the traits class of the property.
          return EquationTraits::getReads(&property);
        }

      private:
        EquationProperty property;
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

        ScheduledSCC(llvm::ArrayRef<ElementType> equations)
          : equations(equations.begin(), equations.end())
        {
          assert(!this->equations.empty());
        }

        const ElementType& operator[](size_t index) const
        {
          assert(index < equations.size());
          return equations[index];
        }

        size_t size() const
        {
          return equations.size();
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
    };
  }

  /// The scheduler allows to sort the equations in such a way that each scalar
  /// variable is determined before being accessed.
  /// The scheduling algorithm assumes that all the algebraic loops have
  /// already been resolved and the only possible kind of loop is the one given
  /// by an equation depending on itself (for example, x[i] = f(x[i - 1]), with
  /// i belonging to a range wider than one).
  template<typename VariableProperty, typename SCCProperty>
  class Scheduler
  {
    private:
      using SCCsGraph = SCCsDependencyGraph<SCCProperty>;
      using SCCTraits = typename SCCsGraph::SCCTraits;
      using EquationDescriptor = typename SCCTraits::ElementRef;
      using EquationTraits = dependency::EquationTraits<EquationDescriptor>;
      using SCCDescriptor = typename SCCsGraph::SCCDescriptor;
      using IndependentSCCs = std::vector<SCCDescriptor>;

      using EquationView = internal::scheduling::EquationView<
          VariableProperty, EquationDescriptor>;

      using ScalarDependencyGraph =
          ScalarVariablesDependencyGraph<VariableProperty, EquationView>;

      using ScheduledEquation =
          scheduling::ScheduledEquation<EquationDescriptor>;

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

      [[nodiscard]] mlir::MLIRContext* getContext() const
      {
        assert(context != nullptr);
        return context;
      }

      Schedule schedule(
          llvm::ArrayRef<SCCProperty> SCCs,
          int64_t maxGroupElements = kUnlimitedGroupElements) const
      {
        Schedule result;

        SCCsGraph SCCsDependencyGraph;
        SCCsDependencyGraph.addSCCs(SCCs);

        auto scheduledSCCGroups =
            sortSCCs(SCCsDependencyGraph, maxGroupElements);

        for (const IndependentSCCs& sccGroup : scheduledSCCGroups) {
          auto& scheduledGroup = result.emplace_back();

          for (SCCDescriptor sccDescriptor : sccGroup) {
            const SCCProperty& scc = SCCsDependencyGraph[sccDescriptor];
            auto sccElements = SCCTraits::getElements(&scc);

            if (sccElements.size() == 1) {
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
                const auto& equation = sccElements[0];

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
                    equation,
                    EquationTraits::getIterationRanges(&equation),
                    directions);

                scheduledGroup.emplace_back(std::move(scheduledEquation));
                continue;
              } else {
                // Mixed accesses detected. Scheduling is possible only on the
                // scalar equations.
                std::vector<EquationView> scalarEquationViews;

                for (const auto& equation : sccElements) {
                  scalarEquationViews.push_back(EquationView(
                      equation,
                      EquationTraits::getIterationRanges(&equation)));
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

                  scheduledGroup.emplace_back(std::move(scheduledEquation));
                }
              }
            } else {
              // A strongly connected component with more than one equation can
              // be scheduled with respect to other SCCs, but the equations
              // composing it are cyclic and thus can't be ordered.
              std::vector<ScheduledEquation> SCC;

              for (const auto& equation : sccElements) {
                SCC.push_back(ScheduledEquation(
                    equation,
                    EquationTraits::getIterationRanges(&equation),
                    scheduling::Direction::Unknown));
              }

              scheduledGroup.emplace_back(std::move(SCC));
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

      /// Given a SSC containing only one equation that may depend on itself,
      /// determine the access direction with respect to the variable that is
      /// written by the equation.
      void getSchedulingDirections(
          const SCCProperty& scc,
          llvm::SmallVectorImpl<DirectionPossibility>& directions) const
      {
        auto sccElements = SCCTraits::getElements(&scc);
        assert(sccElements.size() == 1);
        const auto& equation = sccElements[0];
        auto writeAccess = EquationTraits::getWrite(&equation);
        auto writtenVariable = writeAccess.getVariable();
        auto readAccesses = EquationTraits::getReads(&equation);

        bool hasSelfLoop =
            llvm::any_of(readAccesses, [&](const auto& readAccess) {
              return writtenVariable == readAccess.getVariable();
            });

        size_t rank = EquationTraits::getNumOfIterationVars(&equation);

        if (!hasSelfLoop) {
          // If there is no cycle, then the iteration direction of the equation
          // is irrelevant. We prefer the forward direction for simplicity.
          directions.append(rank, DirectionPossibility::Any);
          return;
        }

        // If all the dependencies have the same direction, then we can set
        // the induction variable to increase or decrease accordingly.
        IndexSet equationIndices =
            EquationTraits::getIterationRanges(&equation);

        const AccessFunction& writeAccessFunction =
            writeAccess.getAccessFunction();

        IndexSet writtenIndices(writeAccessFunction.map(equationIndices));

        for (const auto& read : readAccesses) {
          // The access is considered only if it reads the same variable it is
          // being defined by the equation and the ranges overlap.
          const auto& readVariable = read.getVariable();

          if (writtenVariable != readVariable) {
            continue;
          }

          const AccessFunction& readAccessFunction = read.getAccessFunction();
          IndexSet readIndices(readAccessFunction.map(equationIndices));

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
                EquationTraits::getNumOfIterationVars(&equation),
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

        for (size_t i = directions.size(); i < rank; ++i) {
          directions.push_back(DirectionPossibility::Any);
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
