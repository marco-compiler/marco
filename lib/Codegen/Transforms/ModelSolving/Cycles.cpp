#include "marco/Codegen/Transforms/ModelSolving/Cycles.h"
#include "marco/Codegen/Transforms/ModelSolving/CyclesSubstitutionSolver.h"
#include "marco/Codegen/Transforms/ModelSolving/CyclesCramerSolver.h"
#include "marco/Modeling/Cycles.h"

using namespace ::marco::codegen;
using namespace ::marco::modeling;

#define DEBUG_TYPE "CyclesSolving"

static void createScalarClones(Equations<MatchedEquation> clones, mlir::OpBuilder builder, Equations<MatchedEquation> equations) {
  // Get the number of scalar equations in the system
  for (const auto& equation : equations) {
    auto loc = equation->getOperation()->getLoc();
    for (const auto& range : equation->getIterationRanges()) {
      auto clone = Equation::build(equation->cloneIR(), equation->getVariables());
      // Ensure that this clone gets deleted when we are done
      TemporaryEquationGuard equationGuard(*clone);

      // If the equation has explicit loops, we need to perform additional
      // computations
      // TODO create methods in Equation.h for this to avoid duplicating code
      std::stack<ForEquationOp> loops;
      auto parent = clone->getOperation()->getParentOfType<ForEquationOp>();

      while (parent != nullptr) {
        loops.push(parent);
        parent = parent->getParentOfType<ForEquationOp>();
      }

      builder.setInsertionPointToStart(clone->getOperation().bodyBlock());
      while (!loops.empty()) {
        auto loop = loops.top();
        auto constant = builder.create<ConstantOp>(loc, builder.getIndexAttr(range[loops.size() - 1]));
        loop.induction().replaceAllUsesWith(constant);

        clone->getOperation()->moveBefore(loop.getOperation());
        loops.pop();
        loop.erase();
      }

      // Rebuild the clone so that it is now of the correct type, scalar
      auto scalarClone = Equation::build(clone->cloneIR(), equation->getVariables());
      clones.add(std::make_unique<MatchedEquation>(std::move(scalarClone), IndexSet(Point(0)), equation->getWrite().getPath()));
    }
  }
}

static void populateFlatMap(std::map<size_t, std::unique_ptr<MatchedEquation>>& flatMap, std::vector<MatchedEquation> equations) {
  auto point = Point(0);
  for (const auto& equation : equations) {
    auto flatAccess = equation.getFlatAccessIndex(point);
    assert(flatMap.count(flatAccess) == 0);
    flatMap.emplace(flatAccess, std::make_unique<MatchedEquation>(
                                    Equation::build(equation.cloneIR(), equation.getVariables()), IndexSet(Point(0)), equation.getWrite().getPath()));
  }
}

static void populateFlatMap(std::map<size_t, std::unique_ptr<MatchedEquation>>& flatMap, Equations<MatchedEquation> equations) {
  auto point = Point(0);
  for (const auto& equation : equations) {
    auto flatAccess = equation->getFlatAccessIndex(point);
    assert(flatMap.count(flatAccess) == 0);
    flatMap.emplace(flatAccess, std::make_unique<MatchedEquation>(
                                    Equation::build(equation->cloneIR(), equation->getVariables()), IndexSet(Point(0)), equation->getWrite().getPath()));
  }
}

static bool replaceSolvedIntoUnsolved(
    std::map<size_t, std::unique_ptr<MatchedEquation>>& unsolvedMap, mlir::OpBuilder builder, std::map<size_t, std::unique_ptr<MatchedEquation>>& solvedMap) {
  // Replace the newly found equations (if any) in the unsolved equations.
  // This assumes that the equations are in scalar form.
  bool replaced = false;
  for (const auto& [index, readingEquation] : unsolvedMap) {

    auto clone = Equation::build(readingEquation->cloneIR(), readingEquation->getVariables());
    TemporaryEquationGuard equationGuard(*clone);

    bool replacedRound = false;
    for (const auto& readingAccess : readingEquation->getReads()) {
      auto readingIndex = readingEquation->getFlatAccessIndex(readingAccess, Point(0));

      if (solvedMap.count(readingIndex) != 0) {
        if (mlir::failed(solvedMap.at(readingIndex)->replaceInto(
                builder, solvedMap.at(readingIndex)->getIterationRanges(), *clone,
                readingAccess.getAccessFunction(), readingAccess.getPath()))) {
          std::cerr << "REPLACEMENT ERROR SHOULDN'T HAPPEN\n";
          return false;
        } else {
          replacedRound = true;
        }
      }
    }

    if (replacedRound) {
      auto matchedClone = std::make_unique<MatchedEquation>(
          Equation::build(clone->cloneIR(), clone->getVariables()),
          readingEquation->getIterationRanges(),readingEquation->getWrite().getPath());

      unsolvedMap[index] = std::move(matchedClone);
    }

    replaced = replacedRound || replaced;
  }
  return replaced;
}

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
    CyclesFinder<Variable*, MatchedEquation*> cyclesFinder(secondaryCycles);
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

    auto currentUnsolvedEquations = solver.getUnsolvedEquations();
    auto currentSolution = solver.getSolution();

    // Create the list of equations to be processed in the next iteration
    toBeProcessed.clear();
    newEquations.clear();
    unsolvedEquations.clear();

    for (auto& equation : currentSolution) {
      auto& movedEquation = newEquations.emplace_back(std::move(equation));
      toBeProcessed.push_back(movedEquation.get());
    }

    for (auto& equation : currentUnsolvedEquations) {
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

static bool solveWithCramer(
    Model<MatchedEquation>& model,
    mlir::OpBuilder& builder,
    bool secondaryCycles)
{
  bool allCyclesSolved;

  Equations<MatchedEquation> clones;
  createScalarClones(clones, builder, model.getEquations());

  // The list of equations among which the cycles have to be searched
  llvm::SmallVector<MatchedEquation*> toBeProcessed;

  // The first iteration will use all the equations of the model
  size_t systemSize = clones.size();

  for (const auto& equation : clones) {
    toBeProcessed.push_back(equation.get());
  }


  Equations<MatchedEquation> solution;
  std::map<size_t, std::unique_ptr<MatchedEquation>> solutionMap;
  std::map<size_t, std::unique_ptr<MatchedEquation>> unsolvedMap;

  bool isReplaced;
  bool isSolved;
  do {
    // Get all the cycles within the system of equations
    CyclesFinder<Variable*, MatchedEquation*> cyclesFinder(secondaryCycles);
    cyclesFinder.addEquations(toBeProcessed);
    auto cycles = cyclesFinder.getEquationsCycles();

    // Get all sets of cycles
    std::vector<std::pair<decltype(cyclesFinder)::Cycle, std::set<MatchedEquation*>>> cycleGroups;
    std::set<std::set<MatchedEquation*>> cycleGroupSet;
    for (const auto& cycle : cycles) {
      // Extract the system of equations
      std::set<MatchedEquation*> inputSet;
      inputSet.insert(cycle.getEquation());
      for (const auto& it : cycle) {
        auto destinations = it.getDestinations();
        for (const auto& dest : destinations) {
          inputSet.insert(dest.getNode().getEquation());
        }
      }
      auto [it, inserted] = cycleGroupSet.insert(inputSet);
      if (inserted) {
        cycleGroups.emplace_back(cycle, inputSet);
      }
    }

    // Solve the cycles one by one

    isReplaced = false;
    isSolved = false;

    for (const auto& [cycle, set] : cycleGroups) {
      std::vector<MatchedEquation> input;
      for (const auto eq : set) {
        input.push_back(*eq);
      }

      std::map<size_t, std::unique_ptr<MatchedEquation>> outputMap;
      populateFlatMap(outputMap, input);

      // Solve the system of equations
      CramerSolver solver(builder, systemSize);
      isSolved = solver.solve(outputMap) || isSolved;

      // Collect the solved equations
      auto currentSolution = solver.getSolution();
      populateFlatMap(solutionMap, currentSolution);
      for (const auto& [index, equation] : solutionMap) {
        unsolvedMap.erase(index);
      }

      // Collect the unsolved equations
      populateFlatMap(unsolvedMap, solver.getUnsolvedEquations());

      // Substitute the solved equations into the unsolved ones
      isReplaced = replaceSolvedIntoUnsolved(unsolvedMap, builder, solutionMap) || isReplaced;
    }

    // Create the list of equations to be processed in the next iteration
    toBeProcessed.clear();

    for (auto& [index, equation] : unsolvedMap) {
      toBeProcessed.push_back(equation.get());
    }

    allCyclesSolved = unsolvedMap.empty();
  } while (isSolved || isReplaced);

  // Try to solve the full system if solving by subsystems didn't work
  if(allCyclesSolved) {
    for (auto& [index, equation] : solutionMap) {
      solution.add(std::move(equation));
    }
  } /*else {
    std::cerr << "SOLVING THE FULL SYSTEM\n";
    CramerSolver solver(builder, systemSize);

    std::vector<MatchedEquation> input;
    for (const auto& eq : clones) {
      input.push_back(*eq);
    }
    std::map<size_t, std::unique_ptr<MatchedEquation>> outputMap;
    populateFlatMap(outputMap, input);

    allCyclesSolved = solver.solve(outputMap);

    unsolvedEquations.clear();
    solution = Equations<MatchedEquation>();
    for (auto& equation : solver.getUnsolvedEquations()) {
      unsolvedEquations.emplace_back(std::move(equation));
    }

    for (auto& equation : solver.getSolution()) {
      solution.add(std::move(equation));
    }
  }*/

  for (auto& [index, equation] : unsolvedMap) {
    solution.add(std::move(equation));
  }

  // Set the new equations of the model
  model.setEquations(solution);

  return allCyclesSolved;
}

namespace marco::codegen
{
  mlir::LogicalResult solveCycles(
      Model<MatchedEquation>& model, mlir::OpBuilder& builder)
  {
/*    // Try an aggressive method first
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
    }*/

    // Retry with Cramer
    LLVM_DEBUG({
      llvm::dbgs() << "Solving cycles with Cramer, with secondary cycles\n";
    });

    if (solveWithCramer(model, builder, true)) {
      return mlir::success();
    }

    return mlir::failure();
  }
}
