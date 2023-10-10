#include "marco/Codegen/Transforms/CyclesSolving.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Analysis/DerivativesMap.h"
#include "marco/Codegen/Analysis/VariableAccessAnalysis.h"
#include "marco/Modeling/Cycles.h"
#include "marco/Codegen/Transforms/ModelSolving/CyclesSymbolicSolver.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "cycles-solving"

namespace mlir::modelica
{
#define GEN_PASS_DEF_CYCLESSOLVINGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

/// Convert the dimensions of a variable into an IndexSet.
/// Scalar variables are masked as 1-D arrays with just one element.
static IndexSet getVariableIndices(
    ModelOp root,
    mlir::SymbolRefAttr variable,
    mlir::SymbolTableCollection& symbolTable)
{
  auto variableOp = symbolTable.lookupSymbolIn<VariableOp>(
      root.getOperation(), variable.getRootReference());

  IndexSet indices = variableOp.getIndices();

  if (indices.empty()) {
    // Scalar variable.
    indices += MultidimensionalRange(Range(0, 1));
  }

  return indices;
}

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
  class CyclesSolvingPass
      : public mlir::modelica::impl::CyclesSolvingPassBase<CyclesSolvingPass>,
        public VariableAccessAnalysis::AnalysisProvider
  {
    public:
      using CyclesSolvingPassBase::CyclesSolvingPassBase;

      void runOnOperation() override;

      std::optional<std::reference_wrapper<VariableAccessAnalysis>>
      getCachedVariableAccessAnalysis(EquationTemplateOp op) override;

    private:
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
          llvm::ArrayRef<MatchedEquationInstanceOp> equations);

      mlir::LogicalResult solveCyclesSymbolic(
          mlir::RewriterBase& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          llvm::ArrayRef<MatchedEquationInstanceOp> equations);
  };
}

void CyclesSolvingPass::runOnOperation()
{
  ModelOp modelOp = getOperation();
  std::cerr << "BEFORE: " << std::endl;
  modelOp.dump();

  if (mlir::failed(processModelOp(modelOp))) {
      return signalPassFailure();
  }

  // Determine the analyses to be preserved.
  markAnalysesPreserved<DerivativesMap>();

  llvm::DenseSet<EquationTemplateOp> templateOps;

  for (auto equationOp : modelOp.getOps<MatchedEquationInstanceOp>()) {
    templateOps.insert(equationOp.getTemplate());
  }

  for (EquationTemplateOp templateOp : templateOps) {
    if (auto analysis = getCachedVariableAccessAnalysis(templateOp)) {
      analysis->get().preserve();
    }
  }

  std::cerr << "AFTER: " << std::endl;
  modelOp.dump();
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
CyclesSolvingPass::getCachedVariableAccessAnalysis(EquationTemplateOp op)
{
  return getCachedChildAnalysis<VariableAccessAnalysis>(op);
}

namespace
{
  struct VariableBridge
  {
    VariableBridge(mlir::SymbolRefAttr name, IndexSet indices)
        : name(name),
          indices(std::move(indices))
    {
    }

    mlir::SymbolRefAttr name;
    IndexSet indices;
  };

  struct EquationBridge
  {
    EquationBridge(
        MatchedEquationInstanceOp op,
        mlir::SymbolTableCollection& symbolTable,
        llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge*>& variablesMap)
        : op(op),
          symbolTable(&symbolTable),
          variablesMap(&variablesMap)
    {
    }

    MatchedEquationInstanceOp op;
    mlir::SymbolTableCollection* symbolTable;
    llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge*>* variablesMap;
  };
}

namespace marco::modeling::dependency
{
  template<>
  struct VariableTraits<::VariableBridge*>
  {
    using Variable = ::VariableBridge*;
    using Id = ::VariableBridge*;

    static Id getId(const Variable* variable)
    {
      return *variable;
    }

    static size_t getRank(const Variable* variable)
    {
      size_t rank = (*variable)->indices.rank();

      if (rank == 0) {
        return 1;
      }

      return rank;
    }

    static IndexSet getIndices(const Variable* variable)
    {
      const IndexSet& result = (*variable)->indices;

      if (result.empty()) {
        return {Point(0)};
      }

      return result;
    }
  };

  template<>
  struct EquationTraits<::EquationBridge*>
  {
    using Equation = ::EquationBridge*;
    using Id = mlir::Operation*;

    static Id getId(const Equation* equation)
    {
      return (*equation)->op.getOperation();
    }

    static size_t getNumOfIterationVars(const Equation* equation)
    {
      uint64_t numOfExplicitInductions = static_cast<uint64_t>(
          (*equation)->op.getInductionVariables().size());

      uint64_t numOfImplicitInductions =
          (*equation)->op.getNumOfImplicitInductionVariables();

      uint64_t result = numOfExplicitInductions + numOfImplicitInductions;

      if (result == 0) {
        // Scalar equation.
        return 1;
      }

      return static_cast<size_t>(result);
    }

    static IndexSet getIterationRanges(const Equation* equation)
    {
      IndexSet iterationSpace = (*equation)->op.getIterationSpace();

      if (iterationSpace.empty()) {
        // Scalar equation.
        iterationSpace += MultidimensionalRange(Range(0, 1));
      }

      return iterationSpace;
    }

    using VariableType = ::VariableBridge*;
    using AccessProperty = VariableAccess;

    static std::vector<Access<VariableType, AccessProperty>>
    getAccesses(const Equation* equation)
    {
      std::vector<Access<VariableType, AccessProperty>> result;

      llvm::SmallVector<VariableAccess> accesses;

      if (mlir::failed((*equation)->op.getAccesses(
              accesses, *(*equation)->symbolTable))) {
        return result;
      }

      for (VariableAccess& access : accesses) {
        auto accessFunction = getAccessFunction(
            (*equation)->op.getContext(), access);

        auto variableIt =
            (*(*equation)->variablesMap).find(access.getVariable());

        if (variableIt != (*(*equation)->variablesMap).end()) {
          result.emplace_back(
              variableIt->getSecond(),
              std::move(accessFunction),
              access);
        }
      }

      return result;
    }

    static Access<VariableType, AccessProperty> getWrite(
        const Equation* equation)
    {
      auto matchPath = (*equation)->op.getPath();

      auto write = (*equation)->op.getAccessAtPath(
          *(*equation)->symbolTable, matchPath.getValue());

      assert(write.has_value() && "Can't get the write access");

      auto accessFunction = getAccessFunction(
          (*equation)->op.getContext(), *write);

      return Access(
          (*(*equation)->variablesMap)[write->getVariable()],
          std::move(accessFunction),
          *write);
    }

    static std::vector<Access<VariableType, AccessProperty>> getReads(
        const Equation* equation)
    {
      IndexSet equationIndices = getIterationRanges(equation);

      llvm::SmallVector<VariableAccess> accesses;

      if (mlir::failed((*equation)->op.getAccesses(
              accesses, *(*equation)->symbolTable))) {
        llvm_unreachable("Can't compute the accesses");
        return {};
      }

      llvm::SmallVector<VariableAccess> readAccesses;

      if (mlir::failed((*equation)->op.getReadAccesses(
              readAccesses,
              *(*equation)->symbolTable,
              equationIndices,
              accesses))) {
        llvm_unreachable("Can't compute read accesses");
        return {};
      }

      std::vector<Access<VariableType, AccessProperty>> reads;

      for (const VariableAccess& readAccess : readAccesses) {
        auto variableIt =
            (*(*equation)->variablesMap).find(readAccess.getVariable());

        reads.emplace_back(
            variableIt->getSecond(),
            getAccessFunction((*equation)->op.getContext(), readAccess),
            readAccess);
      }

      return reads;
    }

    static std::unique_ptr<AccessFunction> getAccessFunction(
        mlir::MLIRContext* context,
        const VariableAccess& access)
    {
      const AccessFunction& accessFunction = access.getAccessFunction();

      if (accessFunction.getNumOfResults() == 0) {
        // Access to scalar variable.
        return AccessFunction::build(mlir::AffineMap::get(
            accessFunction.getNumOfDims(), 0,
            mlir::getAffineConstantExpr(0, context)));
      }

      return accessFunction.clone();
    }
  };
}

mlir::LogicalResult CyclesSolvingPass::processModelOp(ModelOp modelOp)
{
  mlir::IRRewriter rewriter(&getContext());

  // Collect the equations.
  llvm::SmallVector<MatchedEquationInstanceOp> initialEquations;
  llvm::SmallVector<MatchedEquationInstanceOp> equations;
  modelOp.collectEquations(initialEquations, equations);

  // The symbol table collection to be used for caching.
  mlir::SymbolTableCollection symbolTableCollection;

  // Perform the solving process on the 'initial conditions' model.
  if (!initialEquations.empty()) {
    if (mlir::failed(solveCyclesSymbolic(
            rewriter, symbolTableCollection, modelOp, initialEquations))) {
      if (!allowUnsolvedCycles) {
        modelOp.emitError()
              << "Cycles solving failed for the 'initial conditions' model";

        return mlir::failure();
      }
    }
  }

  // Perform the solving process on the 'main' model.
  if (!equations.empty()) {
    if (mlir::failed(solveCyclesSymbolic(
            rewriter, symbolTableCollection, modelOp, equations))) {
      if (!allowUnsolvedCycles) {
        modelOp.emitError() << "Cycles solving failed for the 'main' model";
        return mlir::failure();
      }
    }
  }

  return mlir::success();
}

mlir::LogicalResult CyclesSolvingPass::getCycles(
    llvm::SmallVectorImpl<Cycle>& result,
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    llvm::ArrayRef<MatchedEquationInstanceOp> equations)
{
  LLVM_DEBUG({
    llvm::dbgs() << "Searching cycles among the following equations:\n";

    for (MatchedEquationInstanceOp equationOp : equations) {
      llvm::dbgs() << equationOp.getTemplate() << "\n" << equationOp << "\n";
    }
  });

  llvm::SmallVector<std::unique_ptr<VariableBridge>> variableBridges;
  llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge*> variablesMap;
  llvm::SmallVector<std::unique_ptr<EquationBridge>> equationBridges;
  llvm::SmallVector<EquationBridge*> equationPtrs;

  for (VariableOp variableOp : modelOp.getVariables()) {
    auto symbolRefAttr = mlir::SymbolRefAttr::get(variableOp.getSymNameAttr());

    auto& bridge = variableBridges.emplace_back(
        std::make_unique<VariableBridge>(
            symbolRefAttr,
            getVariableIndices(
                modelOp, symbolRefAttr, symbolTableCollection)));

    variablesMap[symbolRefAttr] = bridge.get();
  }

  for (MatchedEquationInstanceOp equation : equations) {
    auto& bridge = equationBridges.emplace_back(
        std::make_unique<EquationBridge>(
            equation, symbolTableCollection, variablesMap));

    equationPtrs.push_back(bridge.get());
  }

  using CyclesFinder =
      marco::modeling::CyclesFinder<VariableBridge*, EquationBridge*>;

  CyclesFinder cyclesFinder(&getContext());
  cyclesFinder.addEquations(equationPtrs);

  auto cycles = cyclesFinder.getEquationsCycles();

  for (auto& cycle : cycles) {
    auto& resultCycle = result.emplace_back();

    for (auto& cyclicEquation : cycle) {
      resultCycle.emplace_back(CyclicEquation{
          cyclicEquation.equation->op,
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

  LLVM_DEBUG(llvm::dbgs() << "Reading equation:\n"
                          << readingEquationOp.getTemplate() << "\n"
                          << readingEquationOp << "\n");

  const AccessFunction& readAccessFunction =
      readingEquation.readAccess.getAccessFunction();

  for (MatchedEquationInstanceOp writingEquationOp : writingEquations) {
    LLVM_DEBUG(llvm::dbgs() << "Writing equation:\n"
                            << writingEquationOp.getTemplate() << "\n"
                            << writingEquationOp << "\n");

    MatchedEquationInstanceOp explicitWritingEquationOp =
        writingEquationOp.cloneAndExplicitate(rewriter, symbolTableCollection);

    if (!explicitWritingEquationOp) {
      return mlir::failure();
    }

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

    for (MatchedEquationInstanceOp equation : newEquations) {
      llvm::dbgs() << equation.getTemplate() << "\n" << equation << "\n";
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
      MatchedEquationInstanceOp equationOp = cyclicEquation.equation;
      llvm::dbgs() << equationOp.getTemplate() << "\n" << equationOp << "\n";
    }
  });

  return ::solveCycle(rewriter, symbolTableCollection, cycle, 0, newEquations);
}

mlir::LogicalResult CyclesSolvingPass::solveCycles(
    mlir::RewriterBase& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    llvm::ArrayRef<MatchedEquationInstanceOp> equations)
{
  llvm::SmallVector<MatchedEquationInstanceOp> currentEquations(
      equations.begin(), equations.end());

  llvm::DenseSet<MatchedEquationInstanceOp> toBeErased;
  llvm::SmallVector<MatchedEquationInstanceOp> allNewEquations;

  auto eraseReplacedOnExit = llvm::make_scope_exit([&]() {
    for (MatchedEquationInstanceOp equationOp : toBeErased) {
      rewriter.eraseOp(equationOp);
    }
  });

  llvm::SmallVector<Cycle, 3> cycles;

  if (mlir::failed(getCycles(
          cycles, symbolTableCollection, modelOp, currentEquations))) {
    return mlir::failure();
  }

  bool atLeastOneChanged;

  while (!cycles.empty()) {
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

            if (auto explicitIndices = firstEquation.getIndices()) {
              MultidimensionalRange explicitRange =
                  range.takeFirstDimensions(
                      explicitIndices->getValue().rank());

              clonedOp.setIndicesAttr(
                  MultidimensionalRangeAttr::get(
                      rewriter.getContext(), std::move(explicitRange)));
            }

            if (auto implicitIndices = firstEquation.getImplicitIndices()) {
              MultidimensionalRange implicitRange =
                  range.takeLastDimensions(
                      implicitIndices->getValue().rank());

              clonedOp.setImplicitIndicesAttr(
                  MultidimensionalRangeAttr::get(
                      rewriter.getContext(), std::move(implicitRange)));
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

    // The IR can't be modified more.
    if (!atLeastOneChanged) {
      return mlir::LogicalResult::success(allowUnsolvedCycles);
    }

    // Search for the remaining cycles.
    cycles.clear();

    for (MatchedEquationInstanceOp equationOp : equations) {
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

void getLoopEquationSet(
    const Cycle& cycle,
    std::vector<marco::codegen::MatchedEquationSubscription>& equations,
    marco::codegen::CyclesSymbolicSolver solver) {

  for (const auto& it : cycle) {
    IndexSet range = it.equationIndices;

    if (!solver.hasSolvedEquation(&it.equation, range)) {
      auto test = marco::codegen::MatchedEquationSubscription(it.equation, range);
      equations.emplace_back(it.equation, range);
    }
  }
}

static mlir::LogicalResult solveCycleSymbolic(
    mlir::RewriterBase& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    const Cycle& cycle,
    llvm::SmallVectorImpl<MatchedEquationInstanceOp>& newEquations)
{
  LLVM_DEBUG({
    llvm::dbgs() << "Solving cycle composed by the following equations:\n";

    for (const CyclicEquation& cyclicEquation : cycle) {
      MatchedEquationInstanceOp equationOp = cyclicEquation.equation;
      llvm::dbgs() << equationOp.getTemplate() << "\n" << equationOp << "\n";
    }
  });

  auto solver = marco::codegen::CyclesSymbolicSolver(rewriter);
  std::vector<marco::codegen::MatchedEquationSubscription> equations;
  getLoopEquationSet(cycle, equations, solver);

  if (!equations.empty()) {
    solver.solve(equations);
  }

  return mlir::success();
}

mlir::LogicalResult CyclesSolvingPass::solveCyclesSymbolic(
    mlir::RewriterBase& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    llvm::ArrayRef<MatchedEquationInstanceOp> equations)
{
  llvm::SmallVector<MatchedEquationInstanceOp> currentEquations(
      equations.begin(), equations.end());

  llvm::DenseSet<MatchedEquationInstanceOp> toBeErased;
  llvm::SmallVector<MatchedEquationInstanceOp> allNewEquations;

  auto eraseReplacedOnExit = llvm::make_scope_exit([&]() {
    for (MatchedEquationInstanceOp equationOp : toBeErased) {
      rewriter.eraseOp(equationOp);
    }
  });

  llvm::SmallVector<Cycle, 3> cycles;

  if (mlir::failed(getCycles(
          cycles, symbolTableCollection, modelOp, currentEquations))) {
    return mlir::failure();
  }

  bool atLeastOneChanged;

  while (!cycles.empty()) {
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
      llvm::SmallVector<MatchedEquationInstanceOp> newEquations;

      if (mlir::succeeded(::solveCycleSymbolic(
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

            if (auto explicitIndices = firstEquation.getIndices()) {
              MultidimensionalRange explicitRange =
                  range.takeFirstDimensions(
                      explicitIndices->getValue().rank());

              clonedOp.setIndicesAttr(
                  MultidimensionalRangeAttr::get(
                      rewriter.getContext(), std::move(explicitRange)));
            }

            if (auto implicitIndices = firstEquation.getImplicitIndices()) {
              MultidimensionalRange implicitRange =
                  range.takeLastDimensions(
                      implicitIndices->getValue().rank());

              clonedOp.setImplicitIndicesAttr(
                  MultidimensionalRangeAttr::get(
                      rewriter.getContext(), std::move(implicitRange)));
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

    // The IR can't be modified more.
    if (!atLeastOneChanged) {
      return mlir::LogicalResult::success(allowUnsolvedCycles);
    }

    // Search for the remaining cycles.
    cycles.clear();

    for (MatchedEquationInstanceOp equationOp : equations) {
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

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createCyclesSolvingPass()
  {
    return std::make_unique<CyclesSolvingPass>();
  }

  std::unique_ptr<mlir::Pass> createCyclesSolvingPass(
      const CyclesSolvingPassOptions& options)
  {
    return std::make_unique<CyclesSolvingPass>(options);
  }
}
