#include "marco/Codegen/Transforms/ModelSolving/Cycles.h"
#include "marco/Codegen/Transforms/ModelSolving/CyclesSubstitutionSolver.h"
#include "marco/Modeling/Cycles.h"

using namespace ::marco::codegen;
using namespace ::marco::modeling;

#define DEBUG_TYPE "CyclesSolving"

static bool solveBySubstitution(Model<MatchedEquation>& model, mlir::OpBuilder& builder, bool secondaryCycles)
{
  bool allCyclesSolved;

  // The list of equations among which the cycles have to be searched
  std::vector<MatchedEquation*> toBeProcessed;

  // The first iteration will use all the equations of the model
  for (const auto& equation : model.getEquations()) {
    toBeProcessed.push_back(equation.get());
  }

  Equations<MatchedEquation> solution;
  std::vector<std::unique_ptr<MatchedEquation>> newEquations;
  std::vector<std::unique_ptr<MatchedEquation>> unsolvedEquations;

  do {
    // Get all the cycles within the system of equations
    CyclesFinder<Variable*, MatchedEquation*> cyclesFinder(toBeProcessed, secondaryCycles);
    auto cycles = cyclesFinder.getEquationsCycles();

    // Solve the cycles one by one
    CyclesSubstitutionSolver<decltype(cyclesFinder)::Cycle> solver(builder);

    for (const auto& cycle : cycles) {
      IndexSet indexesWithoutCycles(cycle.getEquation()->getIterationRanges());

      for (const auto& interval : cycle) {
        indexesWithoutCycles -= interval.getRange();
      }

      solver.solve(cycle);

      // Add the indices that do not present any loop
      for (const auto& range : indexesWithoutCycles) {
        auto clonedEquation = Equation::build(
            cycle.getEquation()->getOperation(),
            cycle.getEquation()->getVariables());

        solution.add(std::make_unique<MatchedEquation>(
            std::move(clonedEquation), range, cycle.getEquation()->getWrite().getPath()));
      }
    }

    // Add the equations which had no cycle for any index.
    // To do this, map the equations with cycles for a faster lookup.
    std::set<const MatchedEquation*> equationsWithCycles;

    for (const auto& cycle : cycles) {
      equationsWithCycles.insert(cycle.getEquation());
    }

    for (auto& equation : toBeProcessed) {
      if (equationsWithCycles.find(equation) == equationsWithCycles.end()) {
        solution.add(std::make_unique<MatchedEquation>(
            equation->clone(), equation->getIterationRanges(), equation->getWrite().getPath()));
      }
    }

    // Create the list of equations to be processed in the next iteration
    toBeProcessed.clear();
    newEquations.clear();
    unsolvedEquations.clear();

    if (auto currentSolution = solver.getSolution(); currentSolution.size() != 0) {
      for (auto& equation : currentSolution) {
        auto& movedEquation = newEquations.emplace_back(std::move(equation));
        toBeProcessed.push_back(movedEquation.get());
      }
    }

    for (auto& equation : solver.getUnsolvedEquations()) {
      auto& movedEquation = unsolvedEquations.emplace_back(std::move(equation));
      toBeProcessed.push_back(movedEquation.get());
    }

    allCyclesSolved = !solver.hasUnsolvedCycles();
  } while (!newEquations.empty());

  // Set the new equations of the model
  model.setEquations(solution);

  return allCyclesSolved;
}

namespace marco::codegen
{
  mlir::LogicalResult solveCycles(
      Model<MatchedEquation>& model, mlir::OpBuilder& builder)
  {
    // Try an aggressive method first
    LLVM_DEBUG({
       llvm::dbgs() << "Solving cycles by substitution, with secondary cycles.\n";
    });

    if (solveBySubstitution(model, builder, true)) {
      return mlir::success();
    }

    // Retry by limiting the cycles identification to the primary ones
    LLVM_DEBUG({
      llvm::dbgs() << "Solving cycles by substitution, without secondary cycles.\n";
    });

    if (solveBySubstitution(model, builder, false)) {
      return mlir::success();
    }

    return mlir::failure();
  }
}
