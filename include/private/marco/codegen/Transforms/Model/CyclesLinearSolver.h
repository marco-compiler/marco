#ifndef MARCO_CODEGEN_TRANSFORMS_MODEL_CYCLESLINEARSOLVER_H
#define MARCO_CODEGEN_TRANSFORMS_MODEL_CYCLESLINEARSOLVER_H

#include "marco/Codegen/Transforms/Model/Matching.h"
#include "marco/Modeling/AccessFunction.h"
#include "mlir/IR/Builders.h"

namespace marco::codegen
{
  template<typename Cycle>
  class CyclesLinearSolver
  {
    private:
      struct UnsolvedCycle
      {
        UnsolvedCycle(Cycle cycle, const Equation* blockingEquation)
            : cycle(std::move(cycle)), blockingEquation(std::move(blockingEquation))
        {
        }

        Cycle cycle;
        const Equation* blockingEquation;
      };

    public:
      CyclesLinearSolver(mlir::OpBuilder& builder) : builder(builder)
      {
      }

      Equations<MatchedEquation> getSolvedEquations() const
      {
        return solvedEquations;
      }

      Equations<MatchedEquation> getUnsolvedEquations() const
      {
        Equations<MatchedEquation> result;

        for (const auto& cycle : unsolvedCycles) {
          const auto& equation = cycle.getEquation();

          result.add(std::make_unique<MatchedEquation>(
              equation->clone(), equation->getIterationRanges(), equation->getWrite().getPath()));
        }

        return result;
      }

      void solve(const Cycle& cycle)
      {
        auto res = solve(cycle, solvedEquations, unsolvedCycles);

        if (mlir::succeeded(res)) {
          // Try to solve previously unsolved cycles
          retryUnsolvedCycles(cycle.getEquation());
        }
      }

    private:
      mlir::LogicalResult solve(
          const Cycle& cycle,
          Equations<MatchedEquation>& currentIterationSolvedEquations,
          std::vector<UnsolvedCycle>& currentIterationUnsolvedCycles)
      {
        std::vector<std::unique_ptr<MatchedEquation>> currentCycleSolvedEquations;
        const Equation* blockingEquation = nullptr;

        auto result = processIntervals(
            currentCycleSolvedEquations, *cycle.getEquation(), cycle.begin(), cycle.end(), blockingEquation);

        if (mlir::succeeded(result)) {
          for (auto& equation : currentCycleSolvedEquations) {
            currentIterationSolvedEquations.add(std::move(equation));
          }
        } else {
          assert(blockingEquation != nullptr);
          currentIterationUnsolvedCycles.emplace_back(cycle, blockingEquation);
          return mlir::failure();
        }

        return mlir::success();
      }

      void retryUnsolvedCycles(const Equation* solvedEquation)
      {
        std::vector<const Equation*> solved;
        std::vector<UnsolvedCycle> currentIterationUnsolvedCycles;
        auto it = unsolvedCycles.begin();

        while (it != unsolvedCycles.end()) {
          if (it->blockingEquation == solvedEquation) {
            auto res = solve(it->cycle, solvedEquations, currentIterationUnsolvedCycles);

            if (mlir::succeeded(res)) {
              assert(currentIterationUnsolvedCycles.empty());
              solved.push_back(it->cycle.getEquation());
            }

            // Remove the current cycle. If it was yet not solvable, then it is already
            // added again by the 'solve' method to the list of unsolved cycles.
            it = unsolvedCycles.erase(it);
          } else {
            ++it;
          }
        }

        for (auto& unsolvedCycle : currentIterationUnsolvedCycles) {
          unsolvedCycles.push_back(std::move(unsolvedCycle));
        }

        for (const auto& equation : solved) {
          retryUnsolvedCycles(equation);
        }
      }

      /// Replace an access to a variable with the expression given by another equation.
      /// A copy of the equation to be used as replacement is first made explicit. If it
      /// is not possible, the clone is erased and a failure is reported back to the caller.
      ///
      /// @param destination      the equation containing the access to be replaced
      /// @param accessFunction   access function used to access the variable
      /// @param accessPath       path leading to the access to be replaced
      /// @param source           equation which once made explicit will provide the expression to be plugged in
      /// @return whether the substitution was successful
      mlir::LogicalResult replaceAccessWithEquation(
          Equation& destination,
          const modeling::AccessFunction& accessFunction,
          const EquationPath& accessPath,
          const MatchedEquation& source)
      {
        // Clone the equation IR, in order to leave the original equation untouched.
        // Its matched path may in fact be needed elsewhere, and making the original
        // equation explicit would invalidate it.
        auto sourceClone = source.cloneAndExplicitate(builder);

        if (sourceClone == nullptr) {
          return mlir::failure();
        }

        TemporaryEquationGuard guard(*sourceClone);
        return sourceClone->replaceInto(builder, destination, accessFunction, accessPath);
      }

      /// Process all the indexes of an equation for which a cycle has been detected.
      template<typename IntervalIt>
      mlir::LogicalResult processIntervals(
          std::vector<std::unique_ptr<MatchedEquation>>& results,
          MatchedEquation& equation,
          IntervalIt intervalBegin,
          IntervalIt intervalEnd,
          const Equation*& blockingEquation)
      {
        for (auto interval = intervalBegin; interval != intervalEnd; ++interval) {
          // Each interval may have multiple accesses leading to cycles. Process each one of them.
          auto dependencies = interval->getDestinations();

          auto result = processDependencies(
              results, equation, dependencies.begin(), dependencies.end(), blockingEquation);

          if (mlir::failed(result)) {
            return result;
          }
        }

        return mlir::success();
      }

      /// Substitute all the read accesses that lead to cycles with the expressions given
      /// by the respective writing equations.
      template<typename DependencyIt>
      mlir::LogicalResult processDependencies(
          std::vector<std::unique_ptr<MatchedEquation>>& results,
          Equation& destination,
          DependencyIt dependencyBegin,
          DependencyIt dependencyEnd,
          const Equation*& blockingEquation)
      {
        for (auto dependency = dependencyBegin; dependency != dependencyEnd; ++dependency) {
          // The access to be replaced
          const auto& access = dependency->getAccess();
          const auto& accessFunction = access.getAccessFunction();
          const auto& accessPath = access.getProperty();

          // The equation that writes into the variable read by the previous access
          const auto& filteredWritingEquation = dependency->getNode();

          auto intervalBegin = filteredWritingEquation.begin();
          auto intervalEnd = filteredWritingEquation.end();

          if (intervalBegin != intervalEnd) {
            std::vector<std::unique_ptr<MatchedEquation>> children;

            // First process the chained dependencies
            if (auto res = processIntervals(
                  children,
                  *filteredWritingEquation.getEquation(),
                  intervalBegin, intervalEnd,
                  blockingEquation); mlir::failed(res)) {
              return res;
            }

            // Then put them into the destination equation. Note that it must be cloned
            // for each different child, as each one of them provides a different
            // expression to replace the read access.
            for (const auto& child : children) {
              auto clonedDestination = Equation::build(destination.cloneIR(), destination.getVariables());
              auto res = replaceAccessWithEquation(*clonedDestination, accessFunction, accessPath, *child);

              if (mlir::failed(res)) {
                blockingEquation = child.get();
                clonedDestination->eraseIR();
                return res;
              }

              // Delete the temporary equation that has been cloned, made explicit and
              // substituted to the read access.
              child->eraseIR();

              // Add the solved equation. In doing so, reuse the original indexes and write access path.
              results.push_back(std::make_unique<MatchedEquation>(
                  std::move(clonedDestination), destination.getIterationRanges(), accessPath));
            }
          } else {
            // The replacement equation has no further cycles, so we can just replace the access
            auto clonedDestination = Equation::build(destination.cloneIR(), destination.getVariables());

            auto res = replaceAccessWithEquation(
                *clonedDestination, accessFunction, accessPath, *filteredWritingEquation.getEquation());

            if (mlir::failed(res)) {
              blockingEquation = filteredWritingEquation.getEquation();
              clonedDestination->eraseIR();
              return res;
            }

            // Add the solved equation
            results.push_back(std::make_unique<MatchedEquation>(
                std::move(clonedDestination), destination.getIterationRanges(), accessPath));
          }
        }

        return mlir::success();
      }

    private:
      mlir::OpBuilder& builder;

      // The equations which had cycles but that have been solved.
      Equations<MatchedEquation> solvedEquations;

      // The cycles that can't be solved by substitution.
      std::vector<UnsolvedCycle> unsolvedCycles;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODEL_CYCLESLINEARSOLVER_H
