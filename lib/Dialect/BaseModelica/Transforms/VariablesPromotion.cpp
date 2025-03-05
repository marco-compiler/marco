#include "marco/Dialect/BaseModelica/Transforms/VariablesPromotion.h"
#include "marco/Dialect/BaseModelica/Analysis/VariableAccessAnalysis.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Modeling/Dependency.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_VARIABLESPROMOTIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class VariablesPromotionPass
    : public mlir::bmodelica::impl::VariablesPromotionPassBase<
          VariablesPromotionPass>,
      public VariableAccessAnalysis::AnalysisProvider {
public:
  using VariablesPromotionPassBase<
      VariablesPromotionPass>::VariablesPromotionPassBase;

  void runOnOperation() override;

  std::optional<std::reference_wrapper<VariableAccessAnalysis>>
  getCachedVariableAccessAnalysis(EquationTemplateOp op) override;

private:
  mlir::LogicalResult processModelOp(ModelOp modelOp);

  mlir::LogicalResult cleanModelOp(ModelOp modelOp);

  std::optional<std::reference_wrapper<VariableAccessAnalysis>>
  getVariableAccessAnalysis(EquationTemplateOp equationTemplate,
                            mlir::SymbolTableCollection &symbolTableCollection);
};
} // namespace

void VariablesPromotionPass::runOnOperation() {
  llvm::SmallVector<ModelOp, 1> modelOps;

  walkClasses(getOperation(), [&](mlir::Operation *op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  auto runFn = [&](mlir::Operation *op) {
    auto modelOp = mlir::cast<ModelOp>(op);

    if (mlir::failed(processModelOp(modelOp))) {
      return mlir::failure();
    }

    if (mlir::failed(cleanModelOp(modelOp))) {
      return mlir::failure();
    }

    return mlir::success();
  };

  if (mlir::failed(
          mlir::failableParallelForEach(&getContext(), modelOps, runFn))) {
    return signalPassFailure();
  }
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
VariablesPromotionPass::getCachedVariableAccessAnalysis(EquationTemplateOp op) {
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

static IndexSet getVariableIndices(VariableOp variableOp) {
  IndexSet indices = variableOp.getIndices();

  if (indices.empty()) {
    // Scalar variable.
    indices += MultidimensionalRange(Range(0, 1));
  }

  return indices;
}

namespace {
struct VariableBridge {
  VariableBridge(mlir::SymbolRefAttr name, IndexSet indices)
      : name(name), indices(std::move(indices)) {}

  // Forbid copies to avoid dangling pointers by design.
  VariableBridge(const VariableBridge &other) = delete;
  VariableBridge(VariableBridge &&other) = delete;
  VariableBridge &operator=(const VariableBridge &other) = delete;
  VariableBridge &operator==(const VariableBridge &other) = delete;

  mlir::SymbolRefAttr name;
  IndexSet indices;
};

struct EquationBridge {
  EquationBridge(
      MatchedEquationInstanceOp op, mlir::SymbolTableCollection &symbolTable,
      VariableAccessAnalysis &accessAnalysis,
      llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge *> &variablesMap)
      : op(op), symbolTable(&symbolTable), accessAnalysis(&accessAnalysis),
        variablesMap(&variablesMap) {}

  // Forbid copies to avoid dangling pointers by design.
  EquationBridge(const EquationBridge &other) = delete;
  EquationBridge(EquationBridge &&other) = delete;
  EquationBridge &operator=(const EquationBridge &other) = delete;
  EquationBridge &operator==(const EquationBridge &other) = delete;

  MatchedEquationInstanceOp op;
  mlir::SymbolTableCollection *symbolTable;
  VariableAccessAnalysis *accessAnalysis;
  llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge *> *variablesMap;
};
} // namespace

namespace marco::modeling::dependency {
template <>
struct VariableTraits<::VariableBridge *> {
  using Variable = ::VariableBridge *;
  using Id = ::VariableBridge *;

  static Id getId(const Variable *variable) { return *variable; }

  static size_t getRank(const Variable *variable) {
    size_t rank = (*variable)->indices.rank();

    if (rank == 0) {
      return 1;
    }

    return rank;
  }

  static IndexSet getIndices(const Variable *variable) {
    const IndexSet &result = (*variable)->indices;

    if (result.empty()) {
      return {Point(0)};
    }

    return result;
  }
};

template <>
struct EquationTraits<::EquationBridge *> {
  using Equation = ::EquationBridge *;
  using Id = mlir::Operation *;

  static Id getId(const Equation *equation) {
    return (*equation)->op.getOperation();
  }

  static size_t getNumOfIterationVars(const Equation *equation) {
    auto numOfInductions =
        static_cast<uint64_t>((*equation)->op.getInductionVariables().size());

    if (numOfInductions == 0) {
      // Scalar equation.
      return 1;
    }

    return static_cast<size_t>(numOfInductions);
  }

  static IndexSet getIterationRanges(const Equation *equation) {
    IndexSet iterationSpace = (*equation)->op.getIterationSpace();

    if (iterationSpace.empty()) {
      // Scalar equation.
      iterationSpace += MultidimensionalRange(Range(0, 1));
    }

    return iterationSpace;
  }

  using VariableType = ::VariableBridge *;
  using AccessProperty = EquationPath;

  static std::vector<Access<VariableType, AccessProperty>>
  getAccesses(const Equation *equation) {
    std::vector<Access<VariableType, AccessProperty>> accesses;

    auto cachedAccesses = (*equation)->accessAnalysis->getAccesses(
        (*equation)->op, *(*equation)->symbolTable);

    if (cachedAccesses) {
      for (auto &access : *cachedAccesses) {
        auto accessFunction =
            getAccessFunction((*equation)->op.getContext(), access);

        auto variableIt =
            (*(*equation)->variablesMap).find(access.getVariable());

        if (variableIt != (*(*equation)->variablesMap).end()) {
          accesses.emplace_back(variableIt->getSecond(),
                                std::move(accessFunction), access.getPath());
        }
      }
    }

    return accesses;
  }

  static Access<VariableType, AccessProperty>
  getWrite(const Equation *equation) {
    IndexSet equationIndices = getIterationRanges(equation);

    auto accesses = (*equation)->accessAnalysis->getAccesses(
        (*equation)->op, *(*equation)->symbolTable);

    llvm::SmallVector<VariableAccess> writeAccesses;

    if (mlir::failed((*equation)->op.getWriteAccesses(
            writeAccesses, *(*equation)->symbolTable, equationIndices,
            *accesses))) {
      llvm_unreachable("Can't compute write accesses");
    }

    assert(!writeAccesses.empty());

    auto accessFunction =
        getAccessFunction((*equation)->op.getContext(), writeAccesses[0]);

    return Access((*(*equation)->variablesMap)[writeAccesses[0].getVariable()],
                  std::move(accessFunction), writeAccesses[0].getPath());
  }

  static std::vector<Access<VariableType, AccessProperty>>
  getReads(const Equation *equation) {
    IndexSet equationIndices = getIterationRanges(equation);

    auto accesses = (*equation)->accessAnalysis->getAccesses(
        (*equation)->op, *(*equation)->symbolTable);

    llvm::SmallVector<VariableAccess> readAccesses;

    if (!accesses) {
      llvm_unreachable("Can't compute read accesses");
      return {};
    }

    if (mlir::failed((*equation)->op.getReadAccesses(
            readAccesses, *(*equation)->symbolTable, equationIndices,
            *accesses))) {
      llvm_unreachable("Can't compute read accesses");
      return {};
    }

    std::vector<Access<VariableType, AccessProperty>> reads;

    for (const VariableAccess &readAccess : readAccesses) {
      auto variableIt =
          (*(*equation)->variablesMap).find(readAccess.getVariable());

      reads.emplace_back(
          variableIt->getSecond(),
          getAccessFunction((*equation)->op.getContext(), readAccess),
          readAccess.getPath());
    }

    return reads;
  }

  static std::unique_ptr<AccessFunction>
  getAccessFunction(mlir::MLIRContext *context, const VariableAccess &access) {
    const AccessFunction &accessFunction = access.getAccessFunction();

    if (accessFunction.getNumOfResults() == 0) {
      // Access to scalar variable.
      return AccessFunction::build(
          mlir::AffineMap::get(accessFunction.getNumOfDims(), 0,
                               mlir::getAffineConstantExpr(0, context)));
    }

    return accessFunction.clone();
  }
};
} // namespace marco::modeling::dependency

mlir::LogicalResult VariablesPromotionPass::processModelOp(ModelOp modelOp) {
  VariableAccessAnalysis::IRListener variableAccessListener(*this);
  mlir::IRRewriter rewriter(&getContext(), &variableAccessListener);
  mlir::SymbolTableCollection symbolTableCollection;

  // Retrieve the derivatives map.
  const DerivativesMap &derivativesMap = modelOp.getProperties().derivativesMap;

  // Collect the variables.
  llvm::SmallVector<VariableOp> variables;
  modelOp.collectVariables(variables);

  // Collect the equations.
  llvm::SmallVector<MatchedEquationInstanceOp> initialEquations;
  llvm::SmallVector<MatchedEquationInstanceOp> mainEquations;
  modelOp.collectInitialEquations(initialEquations);
  modelOp.collectMainEquations(mainEquations);

  // Determine the writes map of the 'initial conditions' model. This must be
  // used to avoid having different initial equations writing into the same
  // scalar variables.
  WritesMap<VariableOp, MatchedEquationInstanceOp> initialConditionsWritesMap;

  if (mlir::failed(getWritesMap(initialConditionsWritesMap, modelOp,
                                initialEquations, symbolTableCollection))) {
    return mlir::failure();
  }

  // Get the writes map of the 'main' model.
  WritesMap<VariableOp, MatchedEquationInstanceOp> mainWritesMap;

  if (mlir::failed(getWritesMap(mainWritesMap, modelOp, mainEquations,
                                symbolTableCollection))) {
    return mlir::failure();
  }

  // The variables that are already marked as parameters.
  llvm::DenseSet<VariableOp> parameters;

  for (VariableOp variableOp : variables) {
    if (variableOp.isReadOnly()) {
      parameters.insert(variableOp);
    }
  }

  // The variables that can may be promoted to parameters.
  llvm::DenseSet<VariableOp> candidateVariables;

  // The promotable indices of the candidate variables.
  // We need to keep both the information in order to correctly handle scalar
  // variables, which would otherwise result as always promotable if looking
  // only at this map.
  llvm::DenseMap<VariableOp, IndexSet> promotableVariablesIndices;

  // Determine the promotable equations by creating the dependency graph and
  // doing a post-order visit.

  using VectorDependencyGraph =
      ::marco::modeling::ArrayEquationsDependencyGraph<::VariableBridge *,
                                                       ::EquationBridge *>;

  using SCC = VectorDependencyGraph::SCC;

  VectorDependencyGraph vectorDependencyGraph(&getContext());

  llvm::SmallVector<std::unique_ptr<VariableBridge>> variableBridges;
  llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge *> variablesMap;
  llvm::SmallVector<std::unique_ptr<EquationBridge>> equationBridges;
  llvm::SmallVector<EquationBridge *> equationPtrs;

  for (VariableOp variableOp : variables) {
    auto symbolRefAttr = mlir::SymbolRefAttr::get(variableOp.getSymNameAttr());

    auto &bridge =
        variableBridges.emplace_back(std::make_unique<VariableBridge>(
            symbolRefAttr, getVariableIndices(variableOp)));

    variablesMap[symbolRefAttr] = bridge.get();
  }

  for (MatchedEquationInstanceOp equationOp : mainEquations) {
    auto accessAnalysis = getVariableAccessAnalysis(equationOp.getTemplate(),
                                                    symbolTableCollection);

    if (!accessAnalysis) {
      return mlir::failure();
    }

    auto &bridge =
        equationBridges.emplace_back(std::make_unique<EquationBridge>(
            equationOp, symbolTableCollection, *accessAnalysis, variablesMap));

    equationPtrs.push_back(bridge.get());
  }

  vectorDependencyGraph.addEquations(equationPtrs);

  ::marco::modeling::SCCsDependencyGraph<SCC> sccDependencyGraph;
  sccDependencyGraph.addSCCs(vectorDependencyGraph.getSCCs());

  auto scheduledSCCs = sccDependencyGraph.postOrder();

  llvm::DenseSet<MatchedEquationInstanceOp> promotableEquations;

  for (const auto &sccDescriptor : scheduledSCCs) {
    const SCC &scc = sccDependencyGraph[sccDescriptor];

    // Collect the equations of the SCC for a faster lookup.
    llvm::DenseSet<MatchedEquationInstanceOp> sccEquations;

    for (const auto &equationDescriptor : scc) {
      EquationBridge *equation =
          scc.getGraph()[equationDescriptor].getProperty();

      sccEquations.insert(equation->op);
    }

    // Check if the current SCC depends only on parametric variables or
    // variables that are written by the equations of the SCC.

    bool promotable = true;

    for (const auto &equationDescriptor : scc) {
      EquationBridge *equation =
          scc.getGraph()[equationDescriptor].getProperty();

      auto accesses = equation->accessAnalysis->getAccesses(
          equation->op, *equation->symbolTable);

      if (!accesses) {
        llvm_unreachable("Can't compute read accesses");
        return mlir::failure();
      }

      // Check if the equation uses the 'time' variable. If it does, then it
      // must not be promoted to an initial equation.
      bool timeUsage = false;

      equation->op.getTemplate().walk([&](TimeOp timeOp) { timeUsage = true; });

      if (timeUsage) {
        promotable = false;
        break;
      }

      // Do not promote the equation if it writes to a derivative variable.
      if (derivativesMap.getDerivedVariable(
              equation->op.getProperties().match.name)) {
        promotable = false;
        break;
      }

      // Check the accesses to the variables.
      promotable &= llvm::all_of(*accesses, [&](const VariableAccess &access) {
        auto readVariableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
            modelOp, access.getVariable());

        if (parameters.contains(readVariableOp)) {
          // If the read variable is a parameter, then there is no need for
          // additional analyses.
          return true;
        }

        const AccessFunction &accessFunction = access.getAccessFunction();
        IndexSet iterationSpace = equation->op.getIterationSpace();
        auto readIndices = accessFunction.map(iterationSpace);

        auto writingEquations =
            llvm::make_range(mainWritesMap.equal_range(readVariableOp));

        if (writingEquations.empty()) {
          // If there is no equation writing to the variable, then the variable
          // may be a state.
          return false;
        }

        return llvm::all_of(writingEquations, [&](const auto &entry) {
          MatchedEquationInstanceOp writingEquation = entry.second.second;
          const IndexSet &writtenIndices = entry.second.first;

          if (promotableEquations.contains(writingEquation)) {
            // The writing equation (and the scalar variables it writes to) has
            // already been marked as promotable.
            return true;
          }

          if (sccEquations.contains(writingEquation)) {
            // If the writing equation belongs to the current SCC, then the
            // whole SCC may still be turned into initial equations.
            return true;
          }

          if (!writtenIndices.empty() &&
              !writtenIndices.overlaps(readIndices)) {
            // Ignore the equation (consider it valid) if its written indices
            // don't overlap the read ones.
            return true;
          }

          return false;
        });
      });
    }

    if (promotable) {
      // Promote all the equations of the SCC.
      for (const auto &equationDescriptor : scc) {
        EquationBridge *equation =
            scc.getGraph()[equationDescriptor].getProperty();

        promotableEquations.insert(equation->op);

        auto writtenVariableOp =
            symbolTableCollection.lookupSymbolIn<VariableOp>(
                modelOp, equation->op.getProperties().match.name);

        std::optional<IndexSet> equationIndices =
            equation->op.getIterationSpace();

        candidateVariables.insert(writtenVariableOp);

        promotableVariablesIndices[writtenVariableOp] +=
            equation->op.getProperties().match.indices;
      }
    }
  }

  // Determine the promotable variables.
  // A variable can be promoted only if all the equations writing to it (and
  // thus all the scalar variables) are promotable.

  llvm::DenseSet<VariableOp> promotableVariables;

  for (VariableOp variableOp : variables) {
    if (!promotableVariables.contains(variableOp) &&
        candidateVariables.contains(variableOp) &&
        variableOp.getIndices() == promotableVariablesIndices[variableOp]) {
      promotableVariables.insert(variableOp);
    }
  }

  // Promote the variables (and the equations writing to them).
  for (VariableOp variableOp : promotableVariables) {
    // Change the variable type.
    auto newVariableType = variableOp.getVariableType().asParameter();
    variableOp.setType(newVariableType);

    // Determine the indices of the variable that are currently handled only by
    // equations that are not initial equations.
    IndexSet variableIndices;

    // Initially, consider all the variable indices.
    for (const auto &entry :
         llvm::make_range(mainWritesMap.equal_range(variableOp))) {
      variableIndices += entry.second.first;
    }

    // Then, remove the indices that are written by already existing initial
    // equations.
    for (const auto &entry :
         llvm::make_range(initialConditionsWritesMap.equal_range(variableOp))) {
      variableIndices -= entry.second.first;
    }

    // Convert the writing non-initial equations into initial equations.
    auto writingEquations =
        llvm::make_range(mainWritesMap.equal_range(variableOp));

    for (const auto &entry : writingEquations) {
      MatchedEquationInstanceOp equationOp = entry.second.second;
      IndexSet writingEquationIndices = equationOp.getIterationSpace();

      llvm::SmallVector<VariableAccess> accesses;
      llvm::SmallVector<VariableAccess> writeAccesses;

      if (mlir::failed(
              equationOp.getAccesses(accesses, symbolTableCollection))) {
        return mlir::failure();
      }

      if (mlir::failed(equationOp.getWriteAccesses(
              writeAccesses, symbolTableCollection, accesses))) {
        return mlir::failure();
      }

      llvm::sort(writeAccesses,
                 [](const VariableAccess &first, const VariableAccess &second) {
                   if (first.getAccessFunction().isInvertible() &&
                       !second.getAccessFunction().isInvertible()) {
                     return true;
                   }

                   if (!first.getAccessFunction().isInvertible() &&
                       second.getAccessFunction().isInvertible()) {
                     return false;
                   }

                   return first < second;
                 });

      // Restrict the indices to the ones not handled by the initial
      // equations.
      IndexSet writtenIndices =
          equationOp.getProperties().match.indices.intersect(variableIndices);

      // Determine if new initial equations should be created.
      bool shouldCreateInitialEquations;

      if (newVariableType.isScalar()) {
        // In case of scalar variables, the initial equation should be created
        // only if there is not one already writing its value.
        shouldCreateInitialEquations =
            initialConditionsWritesMap.find(variableOp) ==
            initialConditionsWritesMap.end();
      } else {
        shouldCreateInitialEquations = !writingEquationIndices.empty();
      }

      if (shouldCreateInitialEquations) {
        rewriter.setInsertionPoint(equationOp->getParentOfType<DynamicOp>());

        auto initialOp = rewriter.create<InitialOp>(modelOp.getLoc());

        rewriter.createBlock(&initialOp.getBodyRegion());
        rewriter.setInsertionPointToStart(initialOp.getBody());

        if (!writingEquationIndices.empty()) {
          // Get the indices of the equation that actually writes the scalar
          // variables of interest.
          writingEquationIndices =
              writeAccesses[0].getAccessFunction().inverseMap(
                  writtenIndices, writingEquationIndices);

          writingEquationIndices =
              writingEquationIndices.getCanonicalRepresentation();

          auto clonedEquationOp = mlir::cast<MatchedEquationInstanceOp>(
              rewriter.clone(*equationOp.getOperation()));

          clonedEquationOp.getProperties().setIndices(writingEquationIndices);

          clonedEquationOp.getProperties().match.indices =
              writeAccesses[0].getAccessFunction().map(writingEquationIndices);

          rewriter.eraseOp(equationOp);
        } else {
          auto clonedOp = mlir::cast<MatchedEquationInstanceOp>(
              rewriter.clone(*equationOp.getOperation()));

          clonedOp.getProperties().match.indices =
              writeAccesses[0].getAccessFunction().map(writingEquationIndices);

          rewriter.eraseOp(equationOp);
        }
      } else {
        rewriter.eraseOp(equationOp);
      }
    }
  }

  return mlir::success();
}

mlir::LogicalResult VariablesPromotionPass::cleanModelOp(ModelOp modelOp) {
  mlir::RewritePatternSet patterns(&getContext());
  ModelOp::getCleaningPatterns(patterns, &getContext());
  return mlir::applyPatternsAndFoldGreedily(modelOp, std::move(patterns));
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
VariablesPromotionPass::getVariableAccessAnalysis(
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

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createVariablesPromotionPass() {
  return std::make_unique<VariablesPromotionPass>();
}
} // namespace mlir::bmodelica
