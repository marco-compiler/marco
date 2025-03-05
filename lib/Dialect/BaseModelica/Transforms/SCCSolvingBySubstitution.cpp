#define DEBUG_TYPE "scc-solving-by-substitution"

#include "marco/Dialect/BaseModelica/Transforms/SCCSolvingBySubstitution.h"
#include "marco/Dialect/BaseModelica/Analysis/VariableAccessAnalysis.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/Modeling/Bridge.h"
#include "marco/Modeling/DependencyGraph.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_SCCSOLVINGBYSUBSTITUTIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;
using namespace ::mlir::bmodelica::bridge;

namespace {
struct CyclicEquation {
  MatchedEquationInstanceOp equation;
  IndexSet equationIndices;
  VariableAccess writeAccess;
  IndexSet writtenVariableIndices;
  VariableAccess readAccess;
  IndexSet readVariableIndices;
};
} // namespace

using Cycle = llvm::SmallVector<CyclicEquation, 3>;

[[maybe_unused]] static void printCycle(llvm::raw_ostream &os,
                                        const Cycle &cycle) {
  for (const CyclicEquation &cyclicEquation : cycle) {
    os << cyclicEquation.writeAccess.getVariable() << " -> ";
  }

  os << cycle.back().readAccess.getVariable() << "\n";

  for (const CyclicEquation &cyclicEquation : cycle) {
    MatchedEquationInstanceOp equationOp = cyclicEquation.equation;
    os << "[writing " << cyclicEquation.writeAccess.getVariable() << " "
       << cyclicEquation.writtenVariableIndices << "] ";

    equationOp.printInline(llvm::dbgs());
    os << "\n";
  }
}

namespace {
class SCCSolvingBySubstitutionPass
    : public mlir::bmodelica::impl::SCCSolvingBySubstitutionPassBase<
          SCCSolvingBySubstitutionPass>,
      public VariableAccessAnalysis::AnalysisProvider {
public:
  using SCCSolvingBySubstitutionPassBase<
      SCCSolvingBySubstitutionPass>::SCCSolvingBySubstitutionPassBase;

  void runOnOperation() override;

  std::optional<std::reference_wrapper<VariableAccessAnalysis>>
  getCachedVariableAccessAnalysis(EquationTemplateOp op) override;

private:
  std::optional<std::reference_wrapper<VariableAccessAnalysis>>
  getVariableAccessAnalysis(EquationTemplateOp equationTemplate,
                            mlir::SymbolTableCollection &symbolTableCollection);

  mlir::LogicalResult processModelOp(ModelOp modelOp);

  mlir::LogicalResult
  getCycles(llvm::SmallVectorImpl<Cycle> &result,
            mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
            llvm::ArrayRef<MatchedEquationInstanceOp> equations);

  mlir::LogicalResult
  solveCycles(mlir::RewriterBase &rewriter,
              mlir::SymbolTableCollection &symbolTableCollection,
              ModelOp modelOp, llvm::ArrayRef<SCCOp> SCCs);

  mlir::LogicalResult
  solveCycle(mlir::RewriterBase &rewriter,
             mlir::SymbolTableCollection &symbolTableCollection,
             ModelOp modelOp, SCCOp scc);

  /// Detect the SCCs among a set of equations and create the SCC
  /// operations containing them.
  mlir::LogicalResult
  createSCCs(mlir::RewriterBase &rewriter,
             mlir::SymbolTableCollection &symbolTableCollection,
             ModelOp modelOp, SCCOp originalSCC,
             llvm::ArrayRef<MatchedEquationInstanceOp> equations);

  mlir::LogicalResult cleanModelOp(ModelOp modelOp);
};
} // namespace

void SCCSolvingBySubstitutionPass::runOnOperation() {
  llvm::SmallVector<ModelOp, 1> modelOps;

  walkClasses(getOperation(), [&](mlir::Operation *op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  auto runFn = [&](mlir::Operation *op) {
    auto modelOp = mlir::cast<ModelOp>(op);
    LLVM_DEBUG(llvm::dbgs() << "Input model:\n" << modelOp << "\n");

    if (mlir::failed(processModelOp(modelOp))) {
      return mlir::failure();
    }

    if (mlir::failed(cleanModelOp(modelOp))) {
      return mlir::failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Output model:\n" << modelOp << "\n");
    return mlir::success();
  };

  if (mlir::failed(
          mlir::failableParallelForEach(&getContext(), modelOps, runFn))) {
    return signalPassFailure();
  }
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
SCCSolvingBySubstitutionPass::getCachedVariableAccessAnalysis(
    EquationTemplateOp op) {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::Operation *parentOp = op->getParentOp();
  llvm::SmallVector<mlir::Operation *> parentOps;

  while (parentOp != moduleOp) {
    parentOps.push_back(parentOp);
    parentOp = parentOp->getParentOp();
  }

  mlir::AnalysisManager analysisManager = getAnalysisManager();

  for (mlir::Operation *currentParentOp : llvm::reverse(parentOps)) {
    analysisManager = analysisManager.nest(currentParentOp);
  }

  return analysisManager.getCachedChildAnalysis<VariableAccessAnalysis>(op);
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
SCCSolvingBySubstitutionPass::getVariableAccessAnalysis(
    EquationTemplateOp equationTemplate,
    mlir::SymbolTableCollection &symbolTableCollection) {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::Operation *parentOp = equationTemplate->getParentOp();
  llvm::SmallVector<mlir::Operation *> parentOps;

  while (parentOp != moduleOp) {
    parentOps.push_back(parentOp);
    parentOp = parentOp->getParentOp();
  }

  mlir::AnalysisManager analysisManager = getAnalysisManager();

  for (mlir::Operation *op : llvm::reverse(parentOps)) {
    analysisManager = analysisManager.nest(op);
  }

  if (auto analysis =
          analysisManager.getCachedChildAnalysis<VariableAccessAnalysis>(
              equationTemplate)) {
    return *analysis;
  }

  auto &analysis = analysisManager.getChildAnalysis<VariableAccessAnalysis>(
      equationTemplate);

  if (mlir::failed(analysis.initialize(symbolTableCollection))) {
    return std::nullopt;
  }

  return std::reference_wrapper(analysis);
}

mlir::LogicalResult
SCCSolvingBySubstitutionPass::processModelOp(ModelOp modelOp) {
  mlir::IRRewriter rewriter(&getContext());

  // Collect the equations.
  llvm::SmallVector<SCCOp> initialSCCs;
  llvm::SmallVector<SCCOp> mainSCCs;
  modelOp.collectInitialSCCs(initialSCCs);
  modelOp.collectMainSCCs(mainSCCs);

  // The symbol table collection to be used for caching.
  mlir::SymbolTableCollection symbolTableCollection;

  // Perform the solving process on the 'initial conditions' model.
  if (!initialSCCs.empty()) {
    if (mlir::failed(solveCycles(rewriter, symbolTableCollection, modelOp,
                                 initialSCCs))) {
      modelOp.emitError()
          << "Cycles solving failed for the 'initial conditions' model";

      return mlir::failure();
    }
  }

  // Perform the solving process on the 'main' model.
  if (!mainSCCs.empty()) {
    if (mlir::failed(
            solveCycles(rewriter, symbolTableCollection, modelOp, mainSCCs))) {
      modelOp.emitError() << "Cycles solving failed for the 'main' model";
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult SCCSolvingBySubstitutionPass::getCycles(
    llvm::SmallVectorImpl<Cycle> &result,
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    llvm::ArrayRef<MatchedEquationInstanceOp> equations) {
  LLVM_DEBUG({
    llvm::dbgs() << "Searching cycles among the following equations:\n";

    for (MatchedEquationInstanceOp equationOp : equations) {
      llvm::dbgs() << "[writing " << equationOp.getProperties().match.name
                   << " " << equationOp.getProperties().match.indices << "] ";

      equationOp.printInline(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
  });

  llvm::SmallVector<std::unique_ptr<VariableBridge>> variableBridges;
  llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge *> variablesMap;
  llvm::SmallVector<std::unique_ptr<MatchedEquationBridge>> equationBridges;
  llvm::SmallVector<MatchedEquationBridge *> equationPtrs;

  for (VariableOp variableOp : modelOp.getVariables()) {
    auto &bridge =
        variableBridges.emplace_back(VariableBridge::build(variableOp));

    auto symbolRefAttr = mlir::SymbolRefAttr::get(variableOp.getSymNameAttr());
    variablesMap[symbolRefAttr] = bridge.get();
  }

  for (MatchedEquationInstanceOp equation : equations) {
    auto variableAccessAnalysis = getVariableAccessAnalysis(
        equation.getTemplate(), symbolTableCollection);

    auto &bridge = equationBridges.emplace_back(MatchedEquationBridge::build(
        static_cast<int64_t>(equationBridges.size()), equation,
        symbolTableCollection, *variableAccessAnalysis, variablesMap));

    equationPtrs.push_back(bridge.get());
  }

  using DependencyGraph =
      marco::modeling::DependencyGraph<VariableBridge *,
                                       MatchedEquationBridge *>;

  DependencyGraph dependencyGraph(&getContext());
  dependencyGraph.addEquations(equationPtrs);

  auto cycles = dependencyGraph.getEquationsCycles();

  for (auto &cycle : cycles) {
    auto &resultCycle = result.emplace_back();

    for (auto &cyclicEquation : cycle) {
      resultCycle.emplace_back(
          CyclicEquation{dependencyGraph[cyclicEquation.equation]->op,
                         std::move(cyclicEquation.equationIndices),
                         std::move(cyclicEquation.writeAccess).getProperty(),
                         std::move(cyclicEquation.writtenVariableIndices),
                         std::move(cyclicEquation.readAccess).getProperty(),
                         std::move(cyclicEquation.readVariableIndices)});
    }
  }

  llvm::sort(result, [](const Cycle &first, const Cycle &second) {
    return first.size() > second.size();
  });

  LLVM_DEBUG(llvm::dbgs() << result.size() << " cycles found\n");
  return mlir::success();
}

static mlir::LogicalResult solveCycle(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, const Cycle &cycle,
    size_t index,
    llvm::SmallVectorImpl<MatchedEquationInstanceOp> &newEquations,
    llvm::DenseSet<MatchedEquationInstanceOp> &nonExplicitableEquations) {
  if (index + 1 == cycle.size()) {
    MatchedEquationInstanceOp equationOp = cycle[index].equation;
    rewriter.setInsertionPoint(equationOp);

    newEquations.push_back(mlir::cast<MatchedEquationInstanceOp>(
        rewriter.clone(*equationOp.getOperation())));

    return mlir::success();
  }

  llvm::SmallVector<MatchedEquationInstanceOp> writingEquations;

  auto removeWritingEquations = llvm::make_scope_exit([&]() {
    for (MatchedEquationInstanceOp writingEquation : writingEquations) {
      rewriter.eraseOp(writingEquation);
    }
  });

  if (mlir::failed(solveCycle(rewriter, symbolTableCollection, cycle, index + 1,
                              writingEquations, nonExplicitableEquations))) {
    return mlir::failure();
  }

  const CyclicEquation &readingEquation = cycle[index];
  MatchedEquationInstanceOp readingEquationOp = readingEquation.equation;

  LLVM_DEBUG(llvm::dbgs() << "Cycle index: " << index << "\n");

  LLVM_DEBUG({
    llvm::dbgs() << "Reading equation:\n";
    llvm::dbgs() << "[reading " << readingEquation.readAccess.getVariable()
                 << " " << readingEquation.readVariableIndices << "] ";

    readingEquationOp.printInline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  const AccessFunction &readAccessFunction =
      readingEquation.readAccess.getAccessFunction();

  for (MatchedEquationInstanceOp writingEquationOp : writingEquations) {
    LLVM_DEBUG({
      llvm::dbgs() << "Writing equation:\n";

      llvm::dbgs() << "[writing "
                   << writingEquationOp.getProperties().match.name << " "
                   << writingEquationOp.getProperties().match.indices << "] ";

      writingEquationOp.printInline(llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    MatchedEquationInstanceOp explicitWritingEquationOp =
        writingEquationOp.cloneAndExplicitate(rewriter, symbolTableCollection);

    if (!explicitWritingEquationOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "The writing equation can't be made explicit\n");
      nonExplicitableEquations.insert(writingEquationOp);
      return mlir::failure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Explicit writing equation:\n";
      explicitWritingEquationOp.printInline(llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    auto removeExplicitEquation = llvm::make_scope_exit(
        [&]() { rewriter.eraseOp(explicitWritingEquationOp); });

    auto explicitWriteAccess = explicitWritingEquationOp.getAccessAtPath(
        symbolTableCollection, EquationPath(EquationPath::LEFT, 0));

    if (!explicitWriteAccess) {
      return mlir::failure();
    }

    const AccessFunction &writeAccessFunction =
        explicitWriteAccess->getAccessFunction();

    IndexSet writingEquationIndices =
        explicitWritingEquationOp.getIterationSpace();

    IndexSet writtenVariableIndices =
        writeAccessFunction.map(writingEquationIndices);

    IndexSet readingEquationIndices;

    if (writtenVariableIndices.empty()) {
      readingEquationIndices = readingEquation.equationIndices;
    } else {
      readingEquationIndices = readAccessFunction.inverseMap(
          writtenVariableIndices, readingEquation.equationIndices);

      readingEquationIndices =
          readingEquationIndices.intersect(readingEquation.equationIndices);
    }

    std::optional<std::reference_wrapper<const IndexSet>>
        optionalReadingEquationIndices = std::nullopt;

    if (!readingEquationIndices.empty()) {
      optionalReadingEquationIndices =
          std::reference_wrapper(readingEquationIndices);
    }

    if (mlir::failed(readingEquationOp.cloneWithReplacedAccess(
            rewriter, symbolTableCollection, optionalReadingEquationIndices,
            readingEquation.readAccess, explicitWritingEquationOp.getTemplate(),
            *explicitWriteAccess, newEquations))) {
      return mlir::failure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "New equations:\n";

    for (MatchedEquationInstanceOp equationOp : newEquations) {
      llvm::dbgs() << "[writing " << equationOp.getProperties().match.name
                   << " " << equationOp.getProperties().match.indices << "] ";

      equationOp.printInline(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
  });

  return mlir::success();
}

static mlir::LogicalResult solveCycle(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, const Cycle &cycle,
    llvm::SmallVectorImpl<MatchedEquationInstanceOp> &newEquations,
    llvm::DenseSet<MatchedEquationInstanceOp> &nonExplicitableEquations) {
  LLVM_DEBUG({
    llvm::dbgs() << "Solving cycle composed by the following equations:\n";

    for (const CyclicEquation &cyclicEquation : cycle) {
      llvm::dbgs() << cyclicEquation.writeAccess.getVariable() << " -> ";
    }

    llvm::dbgs() << cycle.back().readAccess.getVariable() << "\n";

    for (const CyclicEquation &cyclicEquation : cycle) {
      MatchedEquationInstanceOp equationOp = cyclicEquation.equation;
      llvm::dbgs() << "[writing " << cyclicEquation.writeAccess.getVariable()
                   << " " << cyclicEquation.writtenVariableIndices << "] ";
      equationOp.printInline(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
  });

  return ::solveCycle(rewriter, symbolTableCollection, cycle, 0, newEquations,
                      nonExplicitableEquations);
}

static bool isContainedInBiggerCycle(llvm::ArrayRef<Cycle> cycles,
                                     size_t cycleIndex) {
  const Cycle &cycle = cycles[cycleIndex];
  llvm::DenseSet<MatchedEquationInstanceOp> involvedEquations;

  for (size_t i = 1, e = cycle.size(); i < e; ++i) {
    involvedEquations.insert(cycle[i].equation);
  }

  // The cycles are sorted by decreasing size, so we can just iterate up to the
  // currently analyzed cycle.
  for (size_t otherCycleIndex = 0; otherCycleIndex < cycleIndex;
       ++otherCycleIndex) {
    const Cycle &otherCycle = cycles[otherCycleIndex];

    if (otherCycle.size() <= cycle.size()) {
      break;
    }

    llvm::DenseSet<MatchedEquationInstanceOp> otherInvolvedEquations;

    for (size_t i = 1, e = otherCycle.size(); i < e; ++i) {
      otherInvolvedEquations.insert(otherCycle[i].equation);
    }

    if (llvm::all_of(involvedEquations,
                     [&](MatchedEquationInstanceOp equation) {
                       return otherInvolvedEquations.contains(equation);
                     })) {
      if (involvedEquations != otherInvolvedEquations) {
        return true;
      }
    }
  }

  return false;
}

mlir::LogicalResult SCCSolvingBySubstitutionPass::solveCycles(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    llvm::ArrayRef<SCCOp> SCCs) {
  for (SCCOp scc : SCCs) {
    if (mlir::failed(
            solveCycle(rewriter, symbolTableCollection, modelOp, scc))) {
      return mlir::failure();
    }
  }

  return mlir::LogicalResult::success();
}

mlir::LogicalResult SCCSolvingBySubstitutionPass::solveCycle(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    SCCOp scc) {
  // The equations that initially compose the SCC.
  llvm::SmallVector<MatchedEquationInstanceOp> originalEquations;
  scc.collectEquations(originalEquations);

  if (static_cast<int64_t>(originalEquations.size()) > maxEquationsInSCC) {
    // The SCC has too many equations.
    return mlir::success();
  }

  // The equations to be considered during an iteration.
  // Initially, they are the equations within the SCC.
  llvm::SmallVector<MatchedEquationInstanceOp> currentEquations(
      originalEquations.begin(), originalEquations.end());

  // The equations to be erased after having solved the SCC.
  llvm::DenseSet<MatchedEquationInstanceOp> toBeErased;

  // The newly inserted equations.
  llvm::SmallVector<MatchedEquationInstanceOp> allNewEquations;

  // The set of equations that have deemed to be non-explicitable.
  llvm::DenseSet<MatchedEquationInstanceOp> nonExplicitableEquations;

  // Compute the cyclic dependencies within the SCC.
  llvm::SmallVector<Cycle, 3> cycles;

  if (mlir::failed(getCycles(cycles, symbolTableCollection, modelOp,
                             currentEquations))) {
    return mlir::failure();
  }

  int64_t currentIteration = 0;

  while (!cycles.empty() && currentIteration++ < maxIterations) {
    // Try to solve one cycle.
    for (const auto &cycle : llvm::enumerate(cycles)) {
      currentEquations.clear();

      if (isContainedInBiggerCycle(cycles, cycle.index())) {
        LLVM_DEBUG({
          llvm::dbgs() << "The following cycle is skipped for being part of a "
                          "bigger SCC\n";
          printCycle(llvm::dbgs(), cycle.value());
        });

        continue;
      }

      // Check if one of the equations can't be made explicit.
      for (size_t i = 1, e = cycle.value().size(); i < e; ++i) {
        if (nonExplicitableEquations.contains(cycle.value()[i].equation)) {
          LLVM_DEBUG({
            llvm::dbgs() << "The following cycle is skipped for having a "
                            "non-explicitable equation\n";
            printCycle(llvm::dbgs(), cycle.value());
          });

          continue;
        }
      }

      llvm::SmallVector<MatchedEquationInstanceOp> newEquations;

      if (mlir::succeeded(::solveCycle(rewriter, symbolTableCollection,
                                       cycle.value(), newEquations,
                                       nonExplicitableEquations))) {
        MatchedEquationInstanceOp firstEquation = cycle.value()[0].equation;
        IndexSet originalIterationSpace = firstEquation.getIterationSpace();

        if (!originalIterationSpace.empty()) {
          IndexSet remainingIndices = originalIterationSpace;

          for (MatchedEquationInstanceOp newEquation : newEquations) {
            remainingIndices -= newEquation.getIterationSpace();
          }

          rewriter.setInsertionPoint(firstEquation);

          llvm::SmallVector<VariableAccess> accesses;
          llvm::SmallVector<VariableAccess> writeAccesses;

          if (mlir::failed(
                  firstEquation.getAccesses(accesses, symbolTableCollection))) {
            return mlir::failure();
          }

          if (mlir::failed(firstEquation.getWriteAccesses(
                  writeAccesses, symbolTableCollection, accesses))) {
            return mlir::failure();
          }

          assert(!writeAccesses.empty());
          llvm::sort(writeAccesses, [](const VariableAccess &first,
                                       const VariableAccess &second) {
            if (first.getAccessFunction().isAffine() &&
                !second.getAccessFunction().isAffine()) {
              return true;
            }

            if (!first.getAccessFunction().isAffine() &&
                second.getAccessFunction().isAffine()) {
              return false;
            }

            return first < second;
          });

          if (!remainingIndices.empty()) {
            auto clonedOp = mlir::cast<MatchedEquationInstanceOp>(
                rewriter.clone(*firstEquation.getOperation()));

            clonedOp.getProperties().setIndices(remainingIndices);

            clonedOp.getProperties().match.indices =
                writeAccesses[0].getAccessFunction().map(
                    clonedOp.getProperties().indices);

            currentEquations.push_back(clonedOp);
          }
        }

        toBeErased.insert(firstEquation);

        for (MatchedEquationInstanceOp newEquation : newEquations) {
          allNewEquations.push_back(newEquation);
        }

        break;
      }
    }

    // Search for the remaining cycles.
    cycles.clear();

    for (MatchedEquationInstanceOp equationOp : originalEquations) {
      if (!toBeErased.contains(equationOp)) {
        currentEquations.push_back(equationOp);
      }
    }

    for (MatchedEquationInstanceOp equationOp : allNewEquations) {
      if (!toBeErased.contains(equationOp)) {
        currentEquations.push_back(equationOp);
      }
    }

    if (mlir::failed(getCycles(cycles, symbolTableCollection, modelOp,
                               currentEquations))) {
      return mlir::failure();
    }
  }

  if (cycles.empty()) {
    // Collect the remaining equations.
    llvm::SmallVector<MatchedEquationInstanceOp> resultEquations;

    for (MatchedEquationInstanceOp equation :
         scc.getOps<MatchedEquationInstanceOp>()) {
      if (!toBeErased.contains(equation)) {
        resultEquations.push_back(equation);
      }
    }

    // Compute the new SCCs and erase the original one.
    if (mlir::failed(createSCCs(rewriter, symbolTableCollection, modelOp, scc,
                                resultEquations))) {
      return mlir::failure();
    }

    // Erase the equations that have been discarded.
    for (MatchedEquationInstanceOp equation : toBeErased) {
      rewriter.eraseOp(equation);
    }

    rewriter.eraseOp(scc);
  } else {
    // Could not solve the cycle.
    // Cancel the modifications and keep the original SCC.
    for (MatchedEquationInstanceOp equationOp : allNewEquations) {
      equationOp.erase();
    }
  }

  return mlir::success();
}

mlir::LogicalResult SCCSolvingBySubstitutionPass::createSCCs(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    SCCOp originalSCC, llvm::ArrayRef<MatchedEquationInstanceOp> equations) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  llvm::SmallVector<std::unique_ptr<VariableBridge>> variableBridges;
  llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge *> variablesMap;
  llvm::SmallVector<std::unique_ptr<MatchedEquationBridge>> equationBridges;
  llvm::SmallVector<MatchedEquationBridge *> equationPtrs;

  for (VariableOp variableOp : modelOp.getVariables()) {
    auto &bridge =
        variableBridges.emplace_back(VariableBridge::build(variableOp));

    auto symbolRefAttr = mlir::SymbolRefAttr::get(variableOp.getSymNameAttr());
    variablesMap[symbolRefAttr] = bridge.get();
  }

  for (MatchedEquationInstanceOp equation : equations) {
    auto variableAccessAnalysis = getVariableAccessAnalysis(
        equation.getTemplate(), symbolTableCollection);

    auto &bridge = equationBridges.emplace_back(MatchedEquationBridge::build(
        static_cast<int64_t>(equationBridges.size()), equation,
        symbolTableCollection, *variableAccessAnalysis, variablesMap));

    equationPtrs.push_back(bridge.get());
  }

  using DependencyGraph =
      marco::modeling::DependencyGraph<VariableBridge *,
                                       MatchedEquationBridge *>;

  DependencyGraph dependencyGraph(&getContext());
  dependencyGraph.addEquations(equationPtrs);

  llvm::SmallVector<DependencyGraph::SCC> SCCs;
  dependencyGraph.getSCCs(SCCs);

  rewriter.setInsertionPointAfter(originalSCC);

  for (const DependencyGraph::SCC &scc : SCCs) {
    auto sccOp = rewriter.create<SCCOp>(modelOp.getLoc());
    mlir::OpBuilder::InsertionGuard sccGuard(rewriter);

    rewriter.setInsertionPointToStart(
        rewriter.createBlock(&sccOp.getBodyRegion()));

    for (const auto &sccElement : scc) {
      const auto &equation = dependencyGraph[*sccElement];
      const IndexSet &indices = sccElement.getIndices();

      size_t numOfInductions = equation->op.getInductionVariables().size();
      bool isScalarEquation = numOfInductions == 0;

      auto clonedOp = mlir::cast<MatchedEquationInstanceOp>(
          rewriter.clone(*equation->op.getOperation()));

      llvm::SmallVector<VariableAccess> accesses;
      llvm::SmallVector<VariableAccess> writeAccesses;

      if (mlir::failed(
              equation->op.getAccesses(accesses, symbolTableCollection))) {
        return mlir::failure();
      }

      if (mlir::failed(equation->op.getWriteAccesses(
              writeAccesses, symbolTableCollection, accesses))) {
        return mlir::failure();
      }

      llvm::sort(writeAccesses,
                 [](const VariableAccess &first, const VariableAccess &second) {
                   if (first.getAccessFunction().isAffine() &&
                       !second.getAccessFunction().isAffine()) {
                     return true;
                   }

                   if (!first.getAccessFunction().isAffine() &&
                       second.getAccessFunction().isAffine()) {
                     return false;
                   }

                   return first < second;
                 });

      if (isScalarEquation) {
        clonedOp.getProperties().match.indices =
            writeAccesses[0].getAccessFunction().map(IndexSet());

      } else {
        IndexSet slicedIndices = indices.takeFirstDimensions(numOfInductions);
        clonedOp.getProperties().setIndices(slicedIndices);

        clonedOp.getProperties().match.indices =
            writeAccesses[0].getAccessFunction().map(slicedIndices);
      }
    }
  }

  return mlir::success();
}

mlir::LogicalResult
SCCSolvingBySubstitutionPass::cleanModelOp(ModelOp modelOp) {
  mlir::RewritePatternSet patterns(&getContext());
  ModelOp::getCleaningPatterns(patterns, &getContext());
  return mlir::applyPatternsAndFoldGreedily(modelOp, std::move(patterns));
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createSCCSolvingBySubstitutionPass() {
  return std::make_unique<SCCSolvingBySubstitutionPass>();
}

std::unique_ptr<mlir::Pass> createSCCSolvingBySubstitutionPass(
    const SCCSolvingBySubstitutionPassOptions &options) {
  return std::make_unique<SCCSolvingBySubstitutionPass>(options);
}
} // namespace mlir::bmodelica
