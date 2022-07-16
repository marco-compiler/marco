#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_CYCLESLINEARSOLVER_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_CYCLESLINEARSOLVER_H

#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Modeling/AccessFunction.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "CyclesLinearSolver"

namespace marco::codegen
{
  template<typename Cycle>
  class CyclesSubstitutionSolver
  {
    private:
      enum class Result {
          PARTIALLY_SOLVED,
          FULLY_SOLVED,
          ALREADY_SOLVED,
          UNSOLVED
      };

    private:
      /// An equation which originally presented cycles but now, for some indices, does not anymore.
      struct SolvedEquation
      {
        SolvedEquation(const Equation* equation, llvm::ArrayRef<modeling::IndexSet> solvedIndices)
          : equation(std::move(equation))
        {
          for (const auto& indices : solvedIndices) {
            this->solvedIndices += indices;
          }
        }

        // The equation which originally had cycles.
        // The pointer refers to the original equation, which presented (and still presents) cycles.
        const Equation* equation;

        // The indices of the equations for which the cycles have been solved
        modeling::IndexSet solvedIndices;
      };

      struct UnsolvedCycle
      {
        UnsolvedCycle(Cycle cycle, const Equation* blockingEquation)
            : cycle(std::move(cycle)), blockingEquation(std::move(blockingEquation))
        {
        }

        Cycle cycle;

        // The equation which could not be made explicit
        const Equation* blockingEquation;
      };

    public:
      CyclesSubstitutionSolver(mlir::OpBuilder& builder) : builder(builder)
      {
      }

      Equations<MatchedEquation> getSolution() const
      {
        return newEquations_;
      }

      bool hasUnsolvedCycles() const
      {
        return !unsolvedCycles_.empty();
      }

      Equations<MatchedEquation> getUnsolvedEquations() const
      {
        Equations<MatchedEquation> result;

        for (const auto& unsolvedCycle : unsolvedCycles_) {
          const auto& equation = unsolvedCycle.cycle.getEquation();

          modeling::IndexSet indices(equation->getIterationRanges());
          indices -= getSolvedEquationIndices(equation);

          for (const auto& range : llvm::make_range(indices.rangesBegin(), indices.rangesEnd())) {
            result.add(std::make_unique<MatchedEquation>(
                equation->clone(), modeling::IndexSet(range), equation->getWrite().getPath()));
          }
        }

        return result;
      }

      void solve(const Cycle& cycle)
      {
        auto result = solve(cycle, newEquations_, solvedEquations_, unsolvedCycles_);

        if (result == Result::FULLY_SOLVED || result == Result::PARTIALLY_SOLVED || result == Result::ALREADY_SOLVED) {
          // Try to solve previously unsolved cycles
          retryUnsolvedCycles(cycle.getEquation());
        }
      }

    private:
      Result solve(
          const Cycle& cycle,
          Equations<MatchedEquation>& currentIterationNewEquations,
          std::vector<SolvedEquation>& currentIterationSolvedEquations,
          std::vector<UnsolvedCycle>& currentIterationUnsolvedCycles)
      {
        LLVM_DEBUG({
          llvm::dbgs() << debugSeparator() << "\n";
          llvm::dbgs() << "Processing cycle starting with:\n\n";
          printEquation(llvm::dbgs(), *cycle.getEquation());
          llvm::dbgs() << debugSeparator() << "\n\n";
        });

        std::vector<std::unique_ptr<MatchedEquation>> newEquationsVector;
        const Equation* blockingEquation = nullptr;

        auto result = processIntervals(
            newEquationsVector,
            currentIterationSolvedEquations,
            cycle.getEquation(),
            cycle.begin(), cycle.end(),
            blockingEquation);

        LLVM_DEBUG({
          if (!newEquationsVector.empty()) {
            llvm::dbgs() << debugSeparator() << "\n";
            llvm::dbgs() << "Solutions:";

            for (auto& equation : newEquationsVector) {
              llvm::dbgs() << "\n";
              printEquation(llvm::dbgs(), *equation);
            }

            llvm::dbgs() << debugSeparator() << "\n\n";
          }
        });

        for (auto& equation : newEquationsVector) {
          currentIterationNewEquations.add(std::move(equation));
        }

        if (result != Result::FULLY_SOLVED) {
          assert(blockingEquation != nullptr);
          currentIterationUnsolvedCycles.emplace_back(cycle, blockingEquation);
        }

        return result;
      }

      void retryUnsolvedCycles(const Equation* solvedEquation)
      {
        std::vector<SolvedEquation> solvedRoots;
        std::vector<UnsolvedCycle> currentIterationUnsolvedCycles;
        auto it = unsolvedCycles_.begin();

        LLVM_DEBUG({
          if (it != unsolvedCycles_.end()) {
            llvm::dbgs() << "Retrying the unsolved cycles\n";
          }
        });

        while (it != unsolvedCycles_.end()) {
          if (it->blockingEquation == solvedEquation) {
            LLVM_DEBUG(llvm::dbgs() << "\n");
            [[maybe_unused]] auto result = solve(it->cycle, newEquations_, solvedRoots, currentIterationUnsolvedCycles);

            for (const auto& solved : solvedRoots) {
              solvedEquations_.push_back(solved);
            }

            [[maybe_unused]] bool success = result == Result::FULLY_SOLVED || result == Result::PARTIALLY_SOLVED || result == Result::ALREADY_SOLVED;
            assert(!success || currentIterationUnsolvedCycles.empty());

            // Remove the current cycle. If it was yet not solvable, then it is already
            // added again by the 'solve' method to the list of unsolved cycles.
            it = unsolvedCycles_.erase(it);
          } else {
            ++it;
          }
        }

        for (auto& unsolvedCycle : currentIterationUnsolvedCycles) {
          unsolvedCycles_.push_back(std::move(unsolvedCycle));
        }

        for (const auto& solvedEq : solvedRoots) {
          retryUnsolvedCycles(solvedEq.equation);
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
        auto sourceClone = source.cloneIRAndExplicitate(builder);

        if (sourceClone == nullptr) {
          return mlir::failure();
        }

        TemporaryEquationGuard guard(*sourceClone);

        return sourceClone->replaceInto(
            builder, sourceClone->getIterationRanges(), destination, accessFunction, accessPath);
      }

      /// Process all the indexes of an equation for which a cycle has been detected.
      template<typename IntervalIt>
      Result processIntervals(
          std::vector<std::unique_ptr<MatchedEquation>>& newEquations,
          std::vector<SolvedEquation>& solvedEquations,
          MatchedEquation* equation,
          IntervalIt intervalBegin,
          IntervalIt intervalEnd,
          const Equation*& blockingEquation)
      {
        assert(intervalBegin != intervalEnd);

        size_t processedIntervals = 0;
        size_t solvedIntervals = 0;
        size_t unsolvedIntervals = 0;

        for (auto interval = intervalBegin; interval != intervalEnd; ++interval) {
          LLVM_DEBUG({
            llvm::dbgs() << "Processing interval ";
            std::cerr << interval->getRange();
            llvm::dbgs() << " of equation:\n";
            equation->dumpIR(llvm::dbgs());
            llvm::dbgs() << "\n\n";
          });

          ++processedIntervals;

          if (hasSolvedEquation(equation, modeling::IndexSet(interval->getRange()))) {
            // If the equation to be inserted has already been solved, then the cycle
            // doesn't exist anymore.
            continue;
          }

          // Each interval may have multiple accesses leading to cycles. Process each one of them.
          auto dependencies = interval->getDestinations();

          // Temporarily store the new equations into a local container, so that we can determine
          // the new equations together with their indices.
          std::vector<std::unique_ptr<MatchedEquation>> newEquationsCurrentInterval;

          auto res = processDependencies(
              newEquationsCurrentInterval, *equation, interval->getRange(),
              dependencies.begin(), dependencies.end(),
              blockingEquation);

          if (mlir::succeeded(res)) {
            ++solvedIntervals;

            // Store the indexes for which the equation does not present cycles anymore
            for (const auto& newEquation : newEquationsCurrentInterval) {
              addSolvedEquation(solvedEquations, equation, modeling::IndexSet(newEquation->getIterationRanges()));
            }
          } else {
            ++unsolvedIntervals;
          }

          // Move the new equations into the result container
          for (auto& newEquation : newEquationsCurrentInterval) {
            newEquations.push_back(std::move(newEquation));
          }
        }

        if (solvedIntervals == processedIntervals) {
          assert(unsolvedIntervals == 0);
          return Result::FULLY_SOLVED;
        }

        if (unsolvedIntervals != 0) {
          if (solvedIntervals != 0) {
            return Result::PARTIALLY_SOLVED;
          }

          return Result::UNSOLVED;
        }

        return Result::ALREADY_SOLVED;
      }

      /// Substitute all the read accesses that lead to cycles with the expressions given
      /// by the respective writing equations.
      template<typename DependencyIt>
      mlir::LogicalResult processDependencies(
          std::vector<std::unique_ptr<MatchedEquation>>& results,
          MatchedEquation& destination,
          modeling::MultidimensionalRange indexes,
          DependencyIt dependencyBegin,
          DependencyIt dependencyEnd,
          const Equation*& blockingEquation)
      {
        std::vector<std::unique_ptr<MatchedEquation>> processed;

        processed.push_back(std::make_unique<MatchedEquation>(
            Equation::build(destination.cloneIR(), destination.getVariables()),
            modeling::IndexSet(indexes), destination.getWrite().getPath()));

        for (auto dependency = dependencyBegin; dependency != dependencyEnd; ++dependency) {
          // The access to be replaced
          const auto& access = dependency->getAccess();
          const modeling::AccessFunction& accessFunction = access.getAccessFunction();
          const EquationPath& accessPath = access.getProperty();

          [[maybe_unused]] auto dependencyId = nextDependencyId++;

          LLVM_DEBUG({
            llvm::dbgs() << "Processing dependency with ID " << dependencyId << ":\n";
            destination.getValueAtPath(accessPath).print(llvm::dbgs());
            llvm::dbgs() << "\n\n";
          });

          // The equation that writes into the variable read by the previous access
          const auto& filteredWritingEquation = dependency->getNode();

          auto intervalBegin = filteredWritingEquation.begin();
          auto intervalEnd = filteredWritingEquation.end();

          std::vector<std::unique_ptr<MatchedEquation>> children;
          std::vector<SolvedEquation> solved;

          // Process the chained dependencies
          auto result = processIntervals(
              children, solved, filteredWritingEquation.getEquation(), intervalBegin, intervalEnd, blockingEquation);

          if (result != Result::FULLY_SOLVED && result != Result::ALREADY_SOLVED) {
            // In order to solve a dependency, all the indices of the writing equation must be solved
            blockingEquation = filteredWritingEquation.getEquation();
            return mlir::failure();
          }

          // Utility function to erase the cloned equations in case of failure.
          auto eraseFn = [](std::vector<std::unique_ptr<MatchedEquation>>& equations) {
            for (auto& equation : equations) {
              equation->eraseIR();
            }

            equations.clear();
          };

          if (!children.empty()) {
            // Replace the access with the writing equation
            std::vector<std::unique_ptr<MatchedEquation>> newProcessed;

            for (const auto& child : children) {
              for (const auto& equation : processed) {
                auto cloned = Equation::build(equation->cloneIR(), equation->getVariables());
                auto replacementResult = replaceAccessWithEquation(*cloned, accessFunction, accessPath, *child);

                if (mlir::failed(replacementResult)) {
                  blockingEquation = filteredWritingEquation.getEquation();
                  cloned->eraseIR();
                  eraseFn(children);
                  eraseFn(newProcessed);
                  eraseFn(processed);
                  return replacementResult;
                }

                // Add the equation with the replaced access
                auto readAccessIndices = accessFunction.inverseMap(
                    modeling::IndexSet(child->getWrite().getAccessFunction().map(child->getIterationRanges())),
                    modeling::IndexSet(equation->getIterationRanges()));

                auto newEquationIndices = readAccessIndices.intersect(equation->getIterationRanges());

                for (const auto& newEquationRange : llvm::make_range(newEquationIndices.rangesBegin(), newEquationIndices.rangesEnd())) {
                  newProcessed.push_back(std::make_unique<MatchedEquation>(
                      Equation::build(cloned->cloneIR(), cloned->getVariables()),
                      modeling::IndexSet(newEquationRange), destination.getWrite().getPath()));
                }
              }
            }

            eraseFn(children);
            eraseFn(processed);

            for (auto& equation : newProcessed) {
              processed.push_back(std::move(equation));
            }
          }

          LLVM_DEBUG({
             llvm::dbgs() << "Dependency " << dependencyId << " has been processed. Results:\n\n";

             for (const auto& equation : processed) {
               equation->dumpIR(llvm::dbgs());
               llvm::dbgs() << "\n\n";
             }
          });
        }

        for (auto& equation : processed) {
          results.push_back(std::move(equation));
        }

        return mlir::success();
      }

      void addSolvedEquation(
          std::vector<SolvedEquation>& solvedEquations,
          Equation* const equation,
          modeling::IndexSet indices)
      {
        auto it = llvm::find_if(solvedEquations, [&](SolvedEquation& solvedEquation) {
          return solvedEquation.equation == equation;
        });

        if (it != solvedEquations.end()) {
          modeling::IndexSet& solvedIndices = it->solvedIndices;
          solvedIndices += indices;
        } else {
          solvedEquations.push_back(SolvedEquation(equation, indices));
        }
      }

      bool hasSolvedEquation(Equation* const equation, modeling::IndexSet indices) const
      {
        auto it = llvm::find_if(solvedEquations_, [&](const SolvedEquation& solvedEquation) {
          return solvedEquation.equation == equation && solvedEquation.solvedIndices.contains(indices);
        });

        return it != solvedEquations_.end();
      }

      modeling::IndexSet getSolvedEquationIndices(Equation* const equation) const
      {
        auto it = llvm::find_if(solvedEquations_, [&](const SolvedEquation& solvedEquation) {
          return solvedEquation.equation == equation;
        });

        modeling::IndexSet indices;

        if (it != solvedEquations_.end()) {
          indices += it->solvedIndices;
        }

        return indices;
      }

      llvm::StringLiteral debugSeparator() const
      {
        return "##############################################################################";
      }

      void printEquation(llvm::raw_ostream& os, const Equation& equation) const
      {
        os << "Indices: ";
        std::cerr << equation.getIterationRanges() << "\n\n";
        equation.dumpIR(os);
        os << "\n";
      }

    private:
      mlir::OpBuilder& builder;

      // The equations which originally had cycles but have been partially or fully solved.
      std::vector<SolvedEquation> solvedEquations_;

      // The newly created equations which has no cycles anymore.
      Equations<MatchedEquation> newEquations_;

      // The cycles that can't be solved by substitution.
      std::vector<UnsolvedCycle> unsolvedCycles_;

      // Internal counter for debugging purpose
      size_t nextDependencyId = 0;
  };
}

#undef DEBUG_TYPE

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_CYCLESLINEARSOLVER_H
