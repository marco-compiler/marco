#include "marco/Codegen/Transforms/Scheduling.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Analysis/DerivativesMap.h"
#include "marco/Codegen/Analysis/VariableAccessAnalysis.h"
#include "marco/Modeling/Scheduling.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_SCHEDULINGPASS
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
  class SchedulingPass
      : public mlir::modelica::impl::SchedulingPassBase<SchedulingPass>,
        public VariableAccessAnalysis::AnalysisProvider
  {
    public:
      using SchedulingPassBase::SchedulingPassBase;

      void runOnOperation() override;

      std::optional<std::reference_wrapper<VariableAccessAnalysis>>
      getCachedVariableAccessAnalysis(EquationTemplateOp op) override;

    private:
      mlir::LogicalResult processModelOp(ModelOp modelOp);

      std::optional<std::reference_wrapper<VariableAccessAnalysis>>
      getVariableAccessAnalysis(
          MatchedEquationInstanceOp equation,
          mlir::SymbolTableCollection& symbolTableCollection);

      mlir::LogicalResult schedule(
          mlir::IRRewriter& rewriter,
          ModelOp modelOp,
          llvm::ArrayRef<MatchedEquationInstanceOp> equations,
          bool initialEquations,
          mlir::SymbolTableCollection& symbolTable);
  };
}

void SchedulingPass::runOnOperation()
{
  ModelOp modelOp = getOperation();

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
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
SchedulingPass::getCachedVariableAccessAnalysis(EquationTemplateOp op)
{
  return getCachedChildAnalysis<VariableAccessAnalysis>(op);
}

mlir::LogicalResult SchedulingPass::processModelOp(ModelOp modelOp)
{
  VariableAccessAnalysis::IRListener variableAccessListener(*this);
  mlir::IRRewriter rewriter(&getContext(), &variableAccessListener);

  // Collect the equations.
  llvm::SmallVector<MatchedEquationInstanceOp> initialEquations;
  llvm::SmallVector<MatchedEquationInstanceOp> equations;
  modelOp.collectEquations(initialEquations, equations);

  // The symbol table collection to be used for caching.
  mlir::SymbolTableCollection symbolTableCollection;

  // Perform the scheduling on the 'initial conditions' model.
  if (processICModel && !initialEquations.empty()) {
    if (mlir::failed(schedule(
            rewriter, modelOp, initialEquations, true,
            symbolTableCollection))) {
      modelOp.emitError()
          << "Scheduling failed for the 'initial conditions' model";

      return mlir::failure();
    }
  }

  // Perform the scheduling on the 'main' model.
  if (processMainModel && !equations.empty()) {
    if (mlir::failed(schedule(
            rewriter, modelOp, equations, false, symbolTableCollection))) {
      modelOp.emitError() << "Scheduling failed for the 'main' model";
      return mlir::failure();
    }
  }

  return mlir::success();
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
SchedulingPass::getVariableAccessAnalysis(
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

namespace
{
  struct VariableBridge
  {
    VariableBridge(mlir::SymbolRefAttr name, IndexSet indices)
        : name(name),
          indices(std::move(indices))
    {
    }

    // Forbid copies to avoid dangling pointers by design.
    VariableBridge(const VariableBridge& other) = delete;
    VariableBridge(VariableBridge&& other) = delete;
    VariableBridge& operator=(const VariableBridge& other) = delete;
    VariableBridge& operator==(const VariableBridge& other) = delete;

    mlir::SymbolRefAttr name;
    IndexSet indices;
  };

  struct EquationBridge
  {
    EquationBridge(
        MatchedEquationInstanceOp op,
        mlir::SymbolTableCollection& symbolTable,
        VariableAccessAnalysis& accessAnalysis,
        llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge*>& variablesMap)
        : op(op),
          symbolTable(&symbolTable),
          accessAnalysis(&accessAnalysis),
          variablesMap(&variablesMap)
    {
    }

    // Forbid copies to avoid dangling pointers by design.
    EquationBridge(const EquationBridge& other) = delete;
    EquationBridge(EquationBridge&& other) = delete;
    EquationBridge& operator=(const EquationBridge& other) = delete;
    EquationBridge& operator==(const EquationBridge& other) = delete;

    MatchedEquationInstanceOp op;
    mlir::SymbolTableCollection* symbolTable;
    VariableAccessAnalysis* accessAnalysis;
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
    using AccessProperty = EquationPath;

    static std::vector<Access<VariableType, AccessProperty>>
    getAccesses(const Equation* equation)
    {
      std::vector<Access<VariableType, AccessProperty>> accesses;

      auto cachedAccesses = (*equation)->accessAnalysis->getAccesses(
          (*equation)->op, *(*equation)->symbolTable);

      if (!cachedAccesses) {
        llvm_unreachable("Can't compute read accesses");
        return {};
      }

      for (auto& access : *cachedAccesses) {
        auto accessFunction = getAccessFunction(
            (*equation)->op.getContext(), access);

        auto variableIt =
            (*(*equation)->variablesMap).find(access.getVariable());

        if (variableIt != (*(*equation)->variablesMap).end()) {
          accesses.emplace_back(
              variableIt->getSecond(),
              std::move(accessFunction),
              access.getPath());
        }
      }

      return accesses;
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
          write->getPath());
    }

    static std::vector<Access<VariableType, AccessProperty>> getReads(
        const Equation* equation)
    {
      IndexSet equationIndices = getIterationRanges(equation);

      auto accesses = (*equation)->accessAnalysis->getAccesses(
          (*equation)->op, *(*equation)->symbolTable);

      if (!accesses) {
        llvm_unreachable("Can't compute read accesses");
        return {};
      }

      llvm::SmallVector<VariableAccess> readAccesses;

      if (mlir::failed((*equation)->op.getReadAccesses(
              readAccesses,
              *(*equation)->symbolTable,
              equationIndices,
              *accesses))) {
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
            readAccess.getPath());
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

mlir::LogicalResult SchedulingPass::schedule(
    mlir::IRRewriter& rewriter,
    ModelOp modelOp,
    llvm::ArrayRef<MatchedEquationInstanceOp> equations,
    bool initialEquations,
    mlir::SymbolTableCollection& symbolTable)
{
  using Scheduler =
      ::marco::modeling::Scheduler<::VariableBridge*, ::EquationBridge*>;

  llvm::SmallVector<std::unique_ptr<VariableBridge>> variableBridges;
  llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge*> variablesMap;
  llvm::SmallVector<std::unique_ptr<EquationBridge>> equationBridges;
  llvm::SmallVector<EquationBridge*> equationPtrs;

  // Create the scheduler. We use the pointers to the real nodes in order to
  // speed up the copies.
  Scheduler scheduler(&getContext());

  for (VariableOp variableOp : modelOp.getVariables()) {
    auto symbolRefAttr = mlir::SymbolRefAttr::get(variableOp.getSymNameAttr());

    auto& bridge = variableBridges.emplace_back(
        std::make_unique<VariableBridge>(
            symbolRefAttr, getVariableIndices(
                               modelOp, symbolRefAttr, symbolTable)));

    variablesMap[symbolRefAttr] = bridge.get();
  }

  for (MatchedEquationInstanceOp equation : equations) {
    auto accessAnalysis = getVariableAccessAnalysis(equation, symbolTable);

    if (!accessAnalysis) {
      return mlir::failure();
    }

    auto& bridge = equationBridges.emplace_back(
        std::make_unique<EquationBridge>(
            equation, symbolTable, *accessAnalysis, variablesMap));

    equationPtrs.push_back(bridge.get());
  }

  // Compute the schedule.
  auto scheduledBlocks = scheduler.schedule(equationPtrs);

  // Keep track of the old instances to be erased.
  llvm::DenseSet<MatchedEquationInstanceOp> toBeErased;

  for (const auto& scheduledBlock : llvm::enumerate(scheduledBlocks)) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    bool hasCycle = scheduledBlock.value().hasCycle();

    if (hasCycle) {
      // If the SCC contains a cycle, then all the equations must be declared
      // inside it.
      rewriter.setInsertionPointToEnd(modelOp.getBody());

      auto sccOp = rewriter.create<SCCOp>(
          modelOp.getLoc(), initialEquations, true);

      mlir::Block* sccBody = rewriter.createBlock(&sccOp.getBodyRegion());
      rewriter.setInsertionPointToStart(sccBody);
    }

    for (const auto& scheduledEquation : scheduledBlock.value()) {
      MatchedEquationInstanceOp matchedEquation =
          scheduledEquation.getEquation()->op;

      size_t numOfExplicitInductions =
          matchedEquation.getInductionVariables().size();

      size_t numOfImplicitInductions =
          matchedEquation.getNumOfImplicitInductionVariables();

      bool isScalarEquation = numOfExplicitInductions == 0;

      // Determine the iteration directions.
      llvm::SmallVector<mlir::Attribute, 3> iterationDirections;

      for (marco::modeling::scheduling::Direction direction :
           scheduledEquation.getIterationDirections()) {
        iterationDirections.push_back(
            EquationScheduleDirectionAttr::get(&getContext(), direction));
      }

      // Create an equation for each range of scheduled indices.
      const IndexSet& scheduledIndices = scheduledEquation.getIndexes();

      for (const MultidimensionalRange& scheduledRange :
           llvm::make_range(scheduledIndices.rangesBegin(),
                            scheduledIndices.rangesEnd())) {
        if (!hasCycle) {
          // If the SCC doesn't have a cycle, then each equation has to be
          // declared in a dedicated SCC operation.
          rewriter.setInsertionPointToEnd(modelOp.getBody());

          auto sccOp = rewriter.create<SCCOp>(
              modelOp.getLoc(), initialEquations, false);

          mlir::Block* sccBody = rewriter.createBlock(&sccOp.getBodyRegion());
          rewriter.setInsertionPointToStart(sccBody);
        }

        // Create the operation for the scheduled equation.
        auto scheduledEquationOp =
            rewriter.create<ScheduledEquationInstanceOp>(
                matchedEquation.getLoc(),
                matchedEquation.getTemplate(),
                matchedEquation.getPath(),
                rewriter.getArrayAttr(
                    llvm::ArrayRef(iterationDirections).take_front(
                        numOfExplicitInductions + numOfImplicitInductions)));

        if (!isScalarEquation) {
          MultidimensionalRange explicitRange =
              scheduledRange.takeFirstDimensions(numOfExplicitInductions);

          scheduledEquationOp.setIndicesAttr(
              MultidimensionalRangeAttr::get(&getContext(), explicitRange));
        }

        if (numOfImplicitInductions > 0) {
          MultidimensionalRange implicitRange =
              scheduledRange.takeLastDimensions(numOfImplicitInductions);

          scheduledEquationOp.setImplicitIndicesAttr(
              MultidimensionalRangeAttr::get(&getContext(), implicitRange));
        }
      }

      // Mark the old instance as obsolete.
      toBeErased.insert(matchedEquation);
    }
  }

  // Erase the old equation instances.
  for (MatchedEquationInstanceOp equation : toBeErased) {
    rewriter.eraseOp(equation);
  }

  return mlir::success();
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createSchedulingPass()
  {
    return std::make_unique<SchedulingPass>();
  }

  std::unique_ptr<mlir::Pass> createSchedulingPass(
      const SchedulingPassOptions& options)
  {
    return std::make_unique<SchedulingPass>(options);
  }
}
