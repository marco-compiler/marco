#include "marco/Codegen/Transforms/ModelSolving/Cycles.h"
#include "marco/Codegen/Transforms/ModelSolving/CyclesSubstitutionSolver.h"
#include "marco/Codegen/Transforms/ModelSolving/CyclesSymbolicSolver.h"
#include "marco/Modeling/Cycles.h"

using namespace ::marco::codegen;
using namespace ::marco::modeling;

#define DEBUG_TYPE "CyclesSolving"

static bool solveBySubstitution(Model<MatchedEquation>& model, mlir::OpBuilder& builder, bool secondaryCycles)
{
  bool allCyclesSolved;

  // The list of equations among which the cycles have to be searched
  llvm::SmallVector<MatchedEquation*> toBeProcessed;

  // The first iteration will use all the equations of the model
  for (const auto& equation : model.getEquations()) {
    toBeProcessed.push_back(equation.get());
  }

  Equations<MatchedEquation> solution;
  llvm::SmallVector<std::unique_ptr<MatchedEquation>> newEquations;
  llvm::SmallVector<std::unique_ptr<MatchedEquation>> unsolvedEquations;

  do {
    // Get all the cycles within the system of equations
    CyclesFinder<Variable*, MatchedEquation*> cyclesFinder(
        model.getOperation().getContext(), secondaryCycles);

    cyclesFinder.addEquations(toBeProcessed);
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
      for (const auto& range : llvm::make_range(indexesWithoutCycles.rangesBegin(), indexesWithoutCycles.rangesEnd())) {
        auto clonedEquation = Equation::build(
            cycle.getEquation()->getOperation(),
            cycle.getEquation()->getVariables());

        solution.add(std::make_unique<MatchedEquation>(
            std::move(clonedEquation), IndexSet(range), cycle.getEquation()->getWrite().getPath()));
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

  for (auto& unsolvedEquation : unsolvedEquations) {
    solution.add(std::move(unsolvedEquation));
  }

  // Set the new equations of the model
  model.setEquations(solution);

  return allCyclesSolved;
}

void getLoopEquationSet(const CyclesFinder<Variable*, MatchedEquation*>::Cycle& cycle, std::vector<MatchedEquationSubscription>& equations) {
  // Get the unique equations of the loop

  for (const auto& it : cycle) {
//    std::cerr << std::endl << cycle.getEquation() << std::endl;
//    cycle.getEquation()->dumpIR();
//    std::cerr << std::endl;
    auto range = marco::modeling::IndexSet(it.getRange());
    equations.emplace_back(cycle.getEquation(), range);
//    std::cerr << "Range: " << range << std::endl;


    auto destinations = it.getDestinations();
    for (const auto& dest : destinations) {
      getLoopEquationSet(dest.getNode(), equations);
    }
  }
}

static bool solveWithSymbolicSolver(Model<MatchedEquation>& model, mlir::OpBuilder& builder, bool secondaryCycles)
{
  bool allCyclesSolved;

  // The list of equations among which the cycles have to be searched
  llvm::SmallVector<MatchedEquation*> toBeProcessed;

  // The first iteration will use all the equations of the model
  for (const auto& equation : model.getEquations()) {
    toBeProcessed.push_back(equation.get());
  }

  Equations<MatchedEquation> solution;
  llvm::SmallVector<std::unique_ptr<MatchedEquation>> newEquations;
  llvm::SmallVector<std::unique_ptr<MatchedEquation>> unsolvedEquations;

  do {
    // Get all the cycles within the system of equations
    CyclesFinder<Variable*, MatchedEquation*> cyclesFinder(
        model.getOperation().getContext(), secondaryCycles);

    cyclesFinder.addEquations(toBeProcessed);
    auto cycles = cyclesFinder.getEquationsCycles();

    // Solve the cycles one by one
    CyclesSymbolicSolver solver(builder);

//    std::cerr << "Number of cycles: " << cycles.size() << std::endl;
    for (const auto& cycle : cycles) {
      IndexSet indexesWithoutCycles(cycle.getEquation()->getIterationRanges());

      for (const auto& interval : cycle) {
        indexesWithoutCycles -= interval.getRange();
      }

//      cycle.dump();

      // Get the unique equations of the loop
      std::vector<MatchedEquationSubscription> equationSet;
      getLoopEquationSet(cycle, equationSet);


//      std::cerr << "Equation set size: " << equationSet.size() << std::endl;

      solver.solve(equationSet);

      // Add the indices that do not present any loop
      for (const auto& range : llvm::make_range(indexesWithoutCycles.rangesBegin(), indexesWithoutCycles.rangesEnd())) {
        auto clonedEquation = Equation::build(
            cycle.getEquation()->getOperation(),
            cycle.getEquation()->getVariables());

        solution.add(std::make_unique<MatchedEquation>(
            std::move(clonedEquation), IndexSet(range), cycle.getEquation()->getWrite().getPath()));
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
//      std::cerr << "Current solution size: " << currentSolution.size() << std::endl;
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

  for (auto& unsolvedEquation : unsolvedEquations) {
    solution.add(std::move(unsolvedEquation));
  }

  // Set the new equations of the model
  model.setEquations(solution);

//  std::cerr << "Solution set" << std::endl;

  return allCyclesSolved;
}

namespace marco::codegen
{
  mlir::LogicalResult solveCycles(
      Model<MatchedEquation>& model, mlir::OpBuilder& builder)
  {
//    // Try an aggressive method first
//    LLVM_DEBUG({
//       llvm::dbgs() << "Solving cycles by substitution, with secondary cycles.\n";
//    });
//
//    if (solveBySubstitution(model, builder, true)) {
//      return mlir::success();
//    }
//
//    // Retry by limiting the cycles identification to the primary ones
//    LLVM_DEBUG({
//      llvm::dbgs() << "Solving cycles by substitution, without secondary cycles.\n";
//    });
//
//    if (solveBySubstitution(model, builder, false)) {
//      return mlir::success();
//    }

    // Use the symbolic solver
    LLVM_DEBUG({
      llvm::dbgs() << "Solving cycles with the symbolic solver\n";
    });

    if (solveWithSymbolicSolver(model, builder, true)) {
      return mlir::success();
    }

    return mlir::failure();
  }
}
