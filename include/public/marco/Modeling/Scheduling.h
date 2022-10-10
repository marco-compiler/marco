#ifndef MARCO_MODELING_SCHEDULING_H
#define MARCO_MODELING_SCHEDULING_H

#include "marco/Modeling/Dependency.h"

namespace marco::modeling
{
  namespace scheduling
  {
    /// The direction to be used by the equations loop to update its iteration variable.
    enum class Direction
    {
      None,     // [i]
      Forward,  // [i + n] with n > 0
      Backward, // [i + n] with n < 0
      Constant, // [n] with n constant value
      Mixed,    // mix of the previous cases
      Unknown
    };
  }

  namespace internal::scheduling
  {
    template<typename EquationProperty>
    class ScheduledEquation
    {
      public:
        using Direction = ::marco::modeling::scheduling::Direction;

        ScheduledEquation(EquationProperty property, IndexSet indices, Direction direction)
            : property(std::move(property)), indices(std::move(indices)), direction(direction)
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

        Direction getIterationDirection() const
        {
          return direction;
        }

      private:
        EquationProperty property;
        IndexSet indices;
        Direction direction;
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

        bool hasCycle() const
        {
          return equations.size() != 1;
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

    template<typename ScheduledSCC>
    class Schedule
    {
      private:
        using Container = std::vector<ScheduledSCC>;

      public:
        using const_iterator = typename Container::const_iterator;

        Schedule(llvm::ArrayRef<ScheduledSCC> scheduledSCCs)
          : scheduledSCCs(scheduledSCCs.begin(), scheduledSCCs.end())
        {
        }

        /// Get whether any of the scheduled SCCs has a cycle among its equations.
        bool hasCycles() const
        {
          return llvm::none_of(scheduledSCCs, [](const auto& scheduledSCC) {
            return scheduledSCC.hasCycle();
          });
        }

        /// Get the number of SCCs.
        size_t size() const
        {
          return scheduledSCCs.size();
        }

        const ScheduledSCC& operator[](size_t index) const
        {
          assert(index < scheduledSCCs.size());
          return scheduledSCCs[index];
        }

        /// }
        /// @name Iterators
        /// {

        const_iterator begin() const
        {
          return scheduledSCCs.begin();
        }

        const_iterator end() const
        {
          return scheduledSCCs.end();
        }

        /// }

      private:
        Container scheduledSCCs;
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
      using VectorDependencyGraph = internal::VVarDependencyGraph<VariableProperty, EquationProperty>;
      using Equation = typename VectorDependencyGraph::Equation;
      using SCC = typename VectorDependencyGraph::SCC;
      using SCCDependencyGraph = internal::SCCDependencyGraph<SCC>;
      using ScalarDependencyGraph = internal::SVarDependencyGraph<VariableProperty, EquationProperty>;
      using ScheduledEquation = internal::scheduling::ScheduledEquation<EquationProperty>;
      using ScheduledSCC = internal::scheduling::ScheduledSCC<ScheduledEquation>;
      using Schedule = internal::scheduling::Schedule<ScheduledSCC>;

    public:
      Schedule schedule(llvm::ArrayRef<EquationProperty> equations) const
      {
        std::vector<ScheduledSCC> result;

        VectorDependencyGraph vectorDependencyGraph;
        vectorDependencyGraph.addEquations(equations);

        SCCDependencyGraph sccDependencyGraph;
        sccDependencyGraph.addSCCs(vectorDependencyGraph.getSCCs());

        auto scheduledSCCs = sccDependencyGraph.postOrder();

        for (const auto& sccDescriptor : scheduledSCCs) {
          const SCC& scc = sccDependencyGraph[sccDescriptor];

          if (scc.size() == 1) {
            auto accessesDirection = getAccessesDirection(scc);

            if (accessesDirection == scheduling::Direction::Forward) {
              const auto& equation = scc[0];

              ScheduledEquation scheduledEquation(
                  equation.getProperty(),
                  equation.getIterationRanges(),
                  scheduling::Direction::Backward);

              result.emplace_back(std::move(scheduledEquation));
              continue;
            }

            if (accessesDirection == scheduling::Direction::Backward) {
              const auto& equation = scc[0];

              ScheduledEquation scheduledEquation(
                  equation.getProperty(),
                  equation.getIterationRanges(),
                  scheduling::Direction::Forward);

              result.emplace_back(std::move(scheduledEquation));
              continue;
            }

            // Mixed accesses detected. Scheduling is possible only on the
            // scalar equations.
            std::vector<EquationProperty> equationProperties;

            for (const auto& equationDescriptor : scc) {
              equationProperties.push_back(scc.getGraph()[equationDescriptor].getProperty());
            }

            ScalarDependencyGraph scalarDependencyGraph;
            scalarDependencyGraph.addEquations(equationProperties);

            for (const auto& equationDescriptor : scalarDependencyGraph.postOrder()) {
              const auto& scalarEquation = scalarDependencyGraph[equationDescriptor];

              ScheduledEquation scheduledEquation(
                  scalarEquation.getProperty(),
                  IndexSet(MultidimensionalRange(scalarEquation.getIndex())),
                  scheduling::Direction::Forward);

              result.emplace_back(std::move(scheduledEquation));
            }
          } else {
            // A strong connected component can be scheduled with respect to
            // other SCCs, but the equations composing it are cyclic and thus
            // can't be scheduled.
            std::vector<ScheduledEquation> SCC;

            for (const auto& equationDescriptor : scc) {
              const auto& equation = scc[equationDescriptor];

              SCC.push_back(ScheduledEquation(
                  equation.getProperty(),
                  equation.getIterationRanges(),
                  scheduling::Direction::Unknown));
            }

            result.emplace_back(std::move(SCC));
          }
        }

        return Schedule(std::move(result));
      }

    private:
      /// Given a SSC containing only one equation that may depend on itself,
      /// determine the access direction with respect to the variable that is
      /// written by the equation.
      ///
      /// @param scc  SCC to be examined (consisting of only one equation with a loop on itself)
      /// @return access direction
      scheduling::Direction getAccessesDirection(const SCC& scc) const
      {
        if (!scc.hasCycle()) {
          // If there is no cycle, then the iteration variable of the equation is irrelevant.
          // We prefer the forward direction for simplicity, so we need to return a backward
          // dependency (which will be converted into a forward iteration).
          return scheduling::Direction::Backward;
        }

        // If all the dependencies have the same direction, then we can set
        // the induction variable to increase or decrease accordingly.

        const Equation& equation = scc[0];
        auto equationRange = equation.getIterationRanges();
        const auto& write = equation.getWrite();
        const auto& writtenVariable = write.getVariable();
        const AccessFunction& writeAccessFunction = write.getAccessFunction();
        IndexSet writtenIndices(writeAccessFunction.map(equationRange));

        auto direction = scheduling::Direction::Unknown;

        for (const auto& read : equation.getReads()) {
          // The access is considered only if it reads the same variable it is being defined by the
          // equation and the ranges overlap.
          const auto& readVariable = read.getVariable();

          if (writtenVariable != readVariable) {
            continue;
          }

          const AccessFunction& readAccessFunction = read.getAccessFunction();
          IndexSet readIndices(readAccessFunction.map(equationRange));

          if (!readIndices.overlaps(writtenIndices)) {
            continue;
          }

          // Determine the access direction of the access
          if (!writeAccessFunction.isInvertible()) {
            return scheduling::Direction::Unknown;
          }

          auto relativeAccess = writeAccessFunction.inverse().combine(readAccessFunction);
          auto accessDirection = getAccessFunctionDirection(relativeAccess);

          assert(accessDirection != scheduling::Direction::None &&
                  "Algebraic loop detected. Maybe the equation has not been made explicit with respect to the matched variable?");

          if (direction == scheduling::Direction::Unknown) {
            direction = accessDirection;
          } else if (direction != accessDirection) {
            return scheduling::Direction::Mixed;
          }
        }

        return direction;
      }

      /// Get the access direction of an access function.
      /// For example, an access consisting in [i0 + 1][i1 + 2] has a forward
      /// direction, meaning that it requires variables that will be defined
      /// later in the loop execution. A [i0 - 1][i1 -2] access function has a
      /// backward direction and a [i0 + 1][i1 - 2] has a mixed one.
      /// The indices of the above induction variables refer to the order in
      /// which the induction variables have been defined, meaning that i0 is
      /// the outer-most induction, i1 the second outer-most one, etc.
      ///
      /// @param accessFunction access function to be analyzed
      /// @return access direction
      scheduling::Direction getAccessFunctionDirection(const AccessFunction& accessFunction) const
      {
        auto direction = scheduling::Direction::Unknown;
        assert(accessFunction.size() != 0);

        for (const auto& dimensionAccess : llvm::enumerate(accessFunction)) {
          auto dimensionDirection = scheduling::Direction::None;

          if (dimensionAccess.value().isConstantAccess()) {
            dimensionDirection = scheduling::Direction::Constant;
          } else {
            if (dimensionAccess.value().getInductionVariableIndex() != dimensionAccess.index()) {
              // If the iteration indices are out of order, then some accesses will refer to future
              // written variables and others to past written ones.
              return scheduling::Direction::Mixed;
            }

            // Examine the offset of the single dimension access
            auto offset = dimensionAccess.value().getOffset();

            if (offset == 0) {
              dimensionDirection = scheduling::Direction::None;
            } else if (offset > 0) {
              dimensionDirection = scheduling::Direction::Forward;
            } else if (offset < 0) {
              dimensionDirection = scheduling::Direction::Backward;
            }
          }

          if (direction == scheduling::Direction::Unknown) {
            direction = dimensionDirection;
          } else if (direction != dimensionDirection) {
            return scheduling::Direction::Mixed;
          }
        }

        assert(direction != scheduling::Direction::Unknown);
        return direction;
      }
  };
}

#endif // MARCO_MODELING_SCHEDULING_H
