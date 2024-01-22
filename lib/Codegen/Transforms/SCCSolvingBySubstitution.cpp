#include "marco/Codegen/Transforms/SCCSolvingBySubstitution.h"
#include "marco/Codegen/Analysis/DerivativesMap.h"
#include "marco/Codegen/Analysis/VariableAccessAnalysis.h"
#include "marco/Codegen/Transforms/Modeling/Bridge.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Modeling/DependencyGraph.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "scc-solving-by-substitution"

namespace mlir::modelica
{
#define GEN_PASS_DEF_SCCSOLVINGBYSUBSTITUTIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;
using namespace ::mlir::modelica::bridge;

namespace
{
  struct CyclicEquation
  {
    MatchedEquationInstanceOp equation;
    IndexSet equationIndices;
    VariableAccess writeAccess;
    IndexSet writtenVariableIndices;
    VariableAccess readAccess;
    IndexSet readVariableIndices;
  };
}

using Cycle = llvm::SmallVector<CyclicEquation, 3>;

namespace
{
  class SCCSolvingBySubstitutionPass
      : public mlir::modelica::impl::SCCSolvingBySubstitutionPassBase<
            SCCSolvingBySubstitutionPass>,
        public VariableAccessAnalysis::AnalysisProvider
  {
    public:
      using SCCSolvingBySubstitutionPassBase<SCCSolvingBySubstitutionPass>
          ::SCCSolvingBySubstitutionPassBase;

      void runOnOperation() override;

      std::optional<std::reference_wrapper<VariableAccessAnalysis>>
      getCachedVariableAccessAnalysis(EquationTemplateOp op) override;

    private:
      std::optional<std::reference_wrapper<VariableAccessAnalysis>>
      getVariableAccessAnalysis(
          MatchedEquationInstanceOp equation,
          mlir::SymbolTableCollection& symbolTableCollection);

      mlir::LogicalResult processModelOp(ModelOp modelOp);

      mlir::LogicalResult getCycles(
          llvm::SmallVectorImpl<Cycle>& result,
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          llvm::ArrayRef<MatchedEquationInstanceOp> equations);

      mlir::LogicalResult solveCycles(
          mlir::RewriterBase& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          llvm::ArrayRef<SCCOp> SCCs);

      mlir::LogicalResult solveCycle(
          mlir::RewriterBase& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          SCCOp scc);

      void createSCCs(
          mlir::RewriterBase& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          SCCOp originalSCC,
          llvm::ArrayRef<MatchedEquationInstanceOp> equations);

      mlir::LogicalResult cleanModelOp(ModelOp modelOp);
  };
}

void SCCSolvingBySubstitutionPass::runOnOperation()
{
  ModelOp modelOp = getOperation();
  LLVM_DEBUG(llvm::dbgs() << "Input model:\n" << modelOp << "\n");

  if (mlir::failed(processModelOp(modelOp))) {
    return signalPassFailure();
  }

  if (mlir::failed(cleanModelOp(modelOp))) {
    return signalPassFailure();
  }

  LLVM_DEBUG(llvm::dbgs() << "Output model:\n" << modelOp << "\n");

  // Determine the analyses to be preserved.
  markAnalysesPreserved<DerivativesMap>();
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
SCCSolvingBySubstitutionPass::getCachedVariableAccessAnalysis(EquationTemplateOp op)
{
  return getCachedChildAnalysis<VariableAccessAnalysis>(op);
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
SCCSolvingBySubstitutionPass::getVariableAccessAnalysis(
    MatchedEquationInstanceOp equation,
    mlir::SymbolTableCollection& symbolTableCollection)
{
  if (auto analysis = getCachedChildAnalysis<VariableAccessAnalysis>(
          equation.getTemplate())) {
    return *analysis;
  }

  auto& analysis = getChildAnalysis<VariableAccessAnalysis>(
      equation.getTemplate());

  if (mlir::failed(analysis.initialize(symbolTableCollection))) {
    return std::nullopt;
  }

  return std::reference_wrapper(analysis);
}

mlir::LogicalResult SCCSolvingBySubstitutionPass::processModelOp(
    ModelOp modelOp)
{
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
    if (mlir::failed(solveCycles(
            rewriter, symbolTableCollection, modelOp, initialSCCs))) {
      modelOp.emitError()
            << "Cycles solving failed for the 'initial conditions' model";

      return mlir::failure();
    }
  }

  // Perform the solving process on the 'main' model.
  if (!mainSCCs.empty()) {
    if (mlir::failed(solveCycles(
            rewriter, symbolTableCollection, modelOp, mainSCCs))) {
      modelOp.emitError() << "Cycles solving failed for the 'main' model";
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult SCCSolvingBySubstitutionPass::getCycles(
    llvm::SmallVectorImpl<Cycle>& result,
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    llvm::ArrayRef<MatchedEquationInstanceOp> equations)
{
  LLVM_DEBUG({
    llvm::dbgs() << "Searching cycles among the following equations:\n";

    for (MatchedEquationInstanceOp equationOp : equations) {
      auto matchedAccess = equationOp.getMatchedAccess(symbolTableCollection);
      llvm::dbgs() << "[writing ";

      if (!matchedAccess) {
        llvm::dbgs() << "<unknown>";
      } else {
        llvm::dbgs() << matchedAccess->getVariable();
      }

      llvm::dbgs() << "] ";
      equationOp.printInline(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
  });

  llvm::SmallVector<std::unique_ptr<VariableBridge>> variableBridges;
  llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge*> variablesMap;
  llvm::SmallVector<std::unique_ptr<MatchedEquationBridge>> equationBridges;
  llvm::SmallVector<MatchedEquationBridge*> equationPtrs;

  for (VariableOp variableOp : modelOp.getVariables()) {
    auto& bridge = variableBridges.emplace_back(
        VariableBridge::build(variableOp));

    auto symbolRefAttr = mlir::SymbolRefAttr::get(variableOp.getSymNameAttr());
    variablesMap[symbolRefAttr] = bridge.get();
  }

  for (MatchedEquationInstanceOp equation : equations) {
    auto variableAccessAnalysis =
        getVariableAccessAnalysis(equation, symbolTableCollection);

    auto& bridge = equationBridges.emplace_back(
        MatchedEquationBridge::build(
            equation, symbolTableCollection, *variableAccessAnalysis,
            variablesMap));

    equationPtrs.push_back(bridge.get());
  }

  using DependencyGraph = marco::modeling::DependencyGraph<
      VariableBridge*, MatchedEquationBridge*>;

  DependencyGraph dependencyGraph(&getContext());
  dependencyGraph.addEquations(equationPtrs);

  auto cycles = dependencyGraph.getEquationsCycles();

  for (auto& cycle : cycles) {
    auto& resultCycle = result.emplace_back();

    for (auto& cyclicEquation : cycle) {
      resultCycle.emplace_back(CyclicEquation{
          dependencyGraph[cyclicEquation.equation]->op,
          std::move(cyclicEquation.equationIndices),
          std::move(cyclicEquation.writeAccess).getProperty(),
          std::move(cyclicEquation.writtenVariableIndices),
          std::move(cyclicEquation.readAccess).getProperty(),
          std::move(cyclicEquation.readVariableIndices)
      });
    }
  }

  LLVM_DEBUG(llvm::dbgs() << result.size() << " cycles found\n");
  return mlir::success();
}

static mlir::LogicalResult solveCycle(
    mlir::RewriterBase& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    const Cycle& cycle,
    size_t index,
    llvm::SmallVectorImpl<MatchedEquationInstanceOp>& newEquations)
{
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

  if (mlir::failed(solveCycle(
          rewriter, symbolTableCollection, cycle, index + 1,
          writingEquations))) {
    return mlir::failure();
  }

  const CyclicEquation& readingEquation = cycle[index];
  MatchedEquationInstanceOp readingEquationOp = readingEquation.equation;

  LLVM_DEBUG(llvm::dbgs() << "Cycle index: " << index << "\n");

  LLVM_DEBUG({
      llvm::dbgs() << "Reading equation:\n";
      readingEquationOp.printInline(llvm::dbgs());

      llvm::dbgs() << "\n"
                   << "Read variable: "
                   << readingEquation.readAccess.getVariable()
                   << "\n";
  });

  const AccessFunction& readAccessFunction =
      readingEquation.readAccess.getAccessFunction();

  for (MatchedEquationInstanceOp writingEquationOp : writingEquations) {
    LLVM_DEBUG({
        llvm::dbgs() << "Writing equation:\n";
        writingEquationOp.printInline(llvm::dbgs());
        llvm::dbgs() << "\n";
    });

    MatchedEquationInstanceOp explicitWritingEquationOp =
        writingEquationOp.cloneAndExplicitate(rewriter, symbolTableCollection);

    if (!explicitWritingEquationOp) {
      LLVM_DEBUG(llvm::dbgs() << "The writing equation can't be made explicit\n");
      return mlir::failure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Explicit writing equation:\n";
      explicitWritingEquationOp.printInline(llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    auto removeExplicitEquation = llvm::make_scope_exit([&]() {
      rewriter.eraseOp(explicitWritingEquationOp);
    });

    auto explicitWriteAccess =
        explicitWritingEquationOp.getMatchedAccess(symbolTableCollection);

    if (!explicitWriteAccess) {
      return mlir::failure();
    }

    const AccessFunction& writeAccessFunction =
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

      readingEquationIndices = readingEquationIndices.intersect(
          readingEquation.equationIndices);
    }

    std::optional<std::reference_wrapper<const IndexSet>>
    optionalReadingEquationIndices = std::nullopt;

    if (!readingEquationIndices.empty()) {
      optionalReadingEquationIndices =
          std::reference_wrapper(readingEquationIndices);
    }

    if (mlir::failed(readingEquationOp.cloneWithReplacedAccess(
            rewriter,
            optionalReadingEquationIndices,
            readingEquation.readAccess,
            explicitWritingEquationOp.getTemplate(),
            *explicitWriteAccess,
            newEquations))) {
      return mlir::failure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "New equations:\n";

    for (MatchedEquationInstanceOp equationOp : newEquations) {
      equationOp.printInline(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
  });

  return mlir::success();
}

static mlir::LogicalResult solveCycle(
    mlir::RewriterBase& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    const Cycle& cycle,
    llvm::SmallVectorImpl<MatchedEquationInstanceOp>& newEquations)
{
  LLVM_DEBUG({
    llvm::dbgs() << "Solving cycle composed by the following equations:\n";

    for (const CyclicEquation& cyclicEquation : cycle) {
      llvm::dbgs() << cyclicEquation.writeAccess.getVariable() << " -> ";
    }

    llvm::dbgs() << cycle.back().readAccess.getVariable() << "\n";

    for (const CyclicEquation& cyclicEquation : cycle) {
      MatchedEquationInstanceOp equationOp = cyclicEquation.equation;
      llvm::dbgs() << "[writing " << cyclicEquation.writeAccess.getVariable() << "] ";
      equationOp.printInline(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
  });

  return ::solveCycle(rewriter, symbolTableCollection, cycle, 0, newEquations);
}

static bool isContainedInBiggerCycle(
    const Cycle& cycle, llvm::ArrayRef<Cycle> cycles)
{
  llvm::DenseSet<MatchedEquationInstanceOp> involvedEquations;

  for (size_t i = 1, e = cycle.size(); i < e; ++i) {
    involvedEquations.insert(cycle[i].equation);
  }

  for (const Cycle& otherCycle : cycles) {
    if (otherCycle.size() <= cycle.size()) {
      continue;
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
    mlir::RewriterBase& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    llvm::ArrayRef<SCCOp> SCCs)
{
  for (SCCOp scc : SCCs) {
    if (mlir::failed(solveCycle(
            rewriter, symbolTableCollection, modelOp, scc))) {
      return mlir::failure();
    }
  }

  return mlir::LogicalResult::success();
}

mlir::LogicalResult SCCSolvingBySubstitutionPass::solveCycle(
    mlir::RewriterBase& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    SCCOp scc)
{
  llvm::SmallVector<MatchedEquationInstanceOp> originalEquations;
  scc.collectEquations(originalEquations);

  llvm::SmallVector<MatchedEquationInstanceOp> currentEquations(
      originalEquations.begin(), originalEquations.end());

  llvm::DenseSet<MatchedEquationInstanceOp> toBeErased;
  llvm::SmallVector<MatchedEquationInstanceOp> allNewEquations;

  auto createSCCsFn = llvm::make_scope_exit([&]() {
    for (MatchedEquationInstanceOp equation : toBeErased) {
      rewriter.eraseOp(equation);
    }

    llvm::SmallVector<MatchedEquationInstanceOp> resultEquations;

    for (MatchedEquationInstanceOp equation :
         scc.getOps<MatchedEquationInstanceOp>()) {
      resultEquations.push_back(equation);
    }

    createSCCs(rewriter, symbolTableCollection, modelOp, scc, resultEquations);
    rewriter.eraseOp(scc);
  });

  llvm::SmallVector<Cycle, 3> cycles;

  if (mlir::failed(getCycles(
          cycles, symbolTableCollection, modelOp, currentEquations))) {
    return mlir::failure();
  }

  if (cycles.empty()) {
    return mlir::success();
  }

  bool atLeastOneChanged;
  int64_t currentIteration = 0;

  while (!cycles.empty() && currentIteration++ < maxIterations) {
    // Collect all the equation indices leading to cycles.
    llvm::DenseMap<MatchedEquationInstanceOp, IndexSet> cyclicIndices;

    for (const Cycle& cycle : cycles) {
      MatchedEquationInstanceOp equationOp = cycle[0].equation;
      IndexSet indices = equationOp.getIterationSpace();
      cyclicIndices[equationOp] += indices;
    }

    // Try to solve one cycle.
    atLeastOneChanged = false;

    for (const Cycle& cycle : cycles) {
      currentEquations.clear();

      if (isContainedInBiggerCycle(cycle, cycles)) {
        LLVM_DEBUG({
          llvm::dbgs() << "The following cycle is skipped for being part of a bigger SCC\n";

          for (const CyclicEquation& cyclicEquation : cycle) {
            llvm::dbgs() << cyclicEquation.writeAccess.getVariable() << " -> ";
          }

          llvm::dbgs() << cycle.back().readAccess.getVariable() << "\n";

          for (const CyclicEquation& cyclicEquation : cycle) {
            MatchedEquationInstanceOp equationOp = cyclicEquation.equation;
            llvm::dbgs() << "[writing "
                         << cyclicEquation.writeAccess.getVariable() << "] ";
            equationOp.printInline(llvm::dbgs());
            llvm::dbgs() << "\n";
          }
        });

        continue;
      }

      llvm::SmallVector<MatchedEquationInstanceOp> newEquations;

      if (mlir::succeeded(::solveCycle(
              rewriter, symbolTableCollection, cycle, newEquations))) {
        MatchedEquationInstanceOp firstEquation = cycle[0].equation;
        IndexSet originalIterationSpace = firstEquation.getIterationSpace();

        if (!originalIterationSpace.empty()) {
          IndexSet remainingIndices = originalIterationSpace;

          for (MatchedEquationInstanceOp newEquation : newEquations) {
            remainingIndices -= newEquation.getIterationSpace();
          }

          rewriter.setInsertionPoint(firstEquation);

          for (const MultidimensionalRange& range : llvm::make_range(
                   remainingIndices.rangesBegin(),
                   remainingIndices.rangesEnd())) {
            auto clonedOp = mlir::cast<MatchedEquationInstanceOp>(
                rewriter.clone(*firstEquation.getOperation()));

            if (auto indices = firstEquation.getIndices()) {
              MultidimensionalRange explicitRange =
                  range.takeFirstDimensions(indices->getValue().rank());

              clonedOp.setIndicesAttr(
                  MultidimensionalRangeAttr::get(
                      rewriter.getContext(), std::move(explicitRange)));
            }

            currentEquations.push_back(clonedOp);
          }
        }

        toBeErased.insert(firstEquation);

        for (MatchedEquationInstanceOp newEquation : newEquations) {
          allNewEquations.push_back(newEquation);
        }

        atLeastOneChanged = true;
        break;
      }
    }

    if (!atLeastOneChanged) {
      // The IR can't be modified more.
      return mlir::LogicalResult::success();
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

    if (mlir::failed(getCycles(
            cycles, symbolTableCollection, modelOp, currentEquations))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

void SCCSolvingBySubstitutionPass::createSCCs(
    mlir::RewriterBase& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    SCCOp originalSCC,
    llvm::ArrayRef<MatchedEquationInstanceOp> equations)
{
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  llvm::SmallVector<std::unique_ptr<VariableBridge>> variableBridges;
  llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge*> variablesMap;
  llvm::SmallVector<std::unique_ptr<MatchedEquationBridge>> equationBridges;
  llvm::SmallVector<MatchedEquationBridge*> equationPtrs;

  for (VariableOp variableOp : modelOp.getVariables()) {
    auto& bridge = variableBridges.emplace_back(
        VariableBridge::build(variableOp));

    auto symbolRefAttr = mlir::SymbolRefAttr::get(variableOp.getSymNameAttr());
    variablesMap[symbolRefAttr] = bridge.get();
  }

  for (MatchedEquationInstanceOp equation : equations) {
    auto variableAccessAnalysis =
        getVariableAccessAnalysis(equation, symbolTableCollection);

    auto& bridge = equationBridges.emplace_back(
        MatchedEquationBridge::build(
            equation, symbolTableCollection, *variableAccessAnalysis,
            variablesMap));

    equationPtrs.push_back(bridge.get());
  }

  using DependencyGraph = marco::modeling::DependencyGraph<
      VariableBridge*, MatchedEquationBridge*>;

  DependencyGraph dependencyGraph(&getContext());
  dependencyGraph.addEquations(equationPtrs);

  llvm::SmallVector<DependencyGraph::SCC> SCCs;
  dependencyGraph.getSCCs(SCCs);

  rewriter.setInsertionPointAfter(originalSCC);

  for (const DependencyGraph::SCC& scc : SCCs) {
    auto sccOp = rewriter.create<SCCOp>(modelOp.getLoc());
    mlir::OpBuilder::InsertionGuard sccGuard(rewriter);

    rewriter.setInsertionPointToStart(
        rewriter.createBlock(&sccOp.getBodyRegion()));

    for (const auto& sccElement : scc) {
      const auto& equation = dependencyGraph[*sccElement];
      const IndexSet& indices = sccElement.getIndices();

      size_t numOfInductions = equation->op.getInductionVariables().size();
      bool isScalarEquation = numOfInductions == 0;

      for (const MultidimensionalRange& matchedEquationRange :
           llvm::make_range(indices.rangesBegin(), indices.rangesEnd())) {
        auto clonedOp = mlir::cast<MatchedEquationInstanceOp>(
            rewriter.clone(*equation->op.getOperation()));

        if (!isScalarEquation) {
          MultidimensionalRange explicitRange =
              matchedEquationRange.takeFirstDimensions(numOfInductions);

          clonedOp.setIndicesAttr(
              MultidimensionalRangeAttr::get(&getContext(), explicitRange));
        }
      }
    }
  }
}

mlir::LogicalResult SCCSolvingBySubstitutionPass::cleanModelOp(ModelOp modelOp)
{
  mlir::RewritePatternSet patterns(&getContext());
  ModelOp::getCleaningPatterns(patterns, &getContext());
  return mlir::applyPatternsAndFoldGreedily(modelOp, std::move(patterns));
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createSCCSolvingBySubstitutionPass()
  {
    return std::make_unique<SCCSolvingBySubstitutionPass>();
  }

  std::unique_ptr<mlir::Pass> createSCCSolvingBySubstitutionPass(
      const SCCSolvingBySubstitutionPassOptions& options)
  {
    return std::make_unique<SCCSolvingBySubstitutionPass>(options);
  }
}
