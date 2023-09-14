#include "marco/Codegen/Transforms/Matching.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Analysis/DerivativesMap.h"
#include "marco/Codegen/Analysis/VariableAccessAnalysis.h"
#include "marco/Modeling/Matching.h"
#include <stack>

namespace mlir::modelica
{
#define GEN_PASS_DEF_MATCHINGPASS
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
  class MatchingPass
      : public mlir::modelica::impl::MatchingPassBase<MatchingPass>
  {
    public:
      using MatchingPassBase::MatchingPassBase;

      void runOnOperation() override
      {
        if (mlir::failed(processModelOp(getOperation()))) {
          return signalPassFailure();
        }

        markAnalysesPreserved<DerivativesMap>();
      }

    private:
      mlir::LogicalResult processModelOp(ModelOp modelOp);

      DerivativesMap& getDerivativesMap();

      std::optional<std::reference_wrapper<VariableAccessAnalysis>>
      getVariableAccessAnalysis(
          EquationInstanceOp equation,
          mlir::SymbolTableCollection& symbolTableCollection);

      mlir::LogicalResult match(
          mlir::OpBuilder& builder,
          ModelOp modelOp,
          llvm::ArrayRef<EquationInstanceOp> equations,
          mlir::SymbolTableCollection& symbolTable,
          llvm::function_ref<IndexSet(mlir::SymbolRefAttr)> matchableIndicesFn);
  };
}

mlir::LogicalResult MatchingPass::processModelOp(ModelOp modelOp)
{
  mlir::OpBuilder builder(modelOp);

  // Collect the equations.
  llvm::SmallVector<EquationInstanceOp> initialEquations;
  llvm::SmallVector<EquationInstanceOp> equations;
  modelOp.collectEquations(initialEquations, equations);

  // The symbol table collection to be used for caching.
  mlir::SymbolTableCollection symbolTableCollection;

  // Get the derivatives map.
  auto& derivativesMap = getDerivativesMap();

  // Perform the matching on the 'initial conditions' model.
  if (processICModel && !initialEquations.empty()) {
    auto matchableIndicesFn = [&](mlir::SymbolRefAttr variable) -> IndexSet {
      return getVariableIndices(modelOp, variable, symbolTableCollection);
    };

    if (mlir::failed(match(builder, modelOp, initialEquations,
                           symbolTableCollection, matchableIndicesFn))) {
      modelOp.emitError() << "Matching failed for the 'initial conditions' model";
      return mlir::failure();
    }
  }

  // Perform the matching on the 'main' model.
  if (processMainModel && !equations.empty()) {
    auto matchableIndicesFn = [&](mlir::SymbolRefAttr variable) -> IndexSet {
      assert(variable.getNestedReferences().empty());

      auto variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
          modelOp, variable.getRootReference());

      if (variableOp.isReadOnly()) {
        // Read-only variables are handled by initial equations.
        return {};
      }

      IndexSet variableIndices =
          getVariableIndices(modelOp, variable, symbolTableCollection);

      if (auto derivedIndices = derivativesMap.getDerivedIndices(variable)) {
        if (variableOp.getVariableType().isScalar()) {
          return {};
        }

        return variableIndices - derivedIndices->get();
      }

      return variableIndices;
    };

    if (mlir::failed(match(builder, modelOp, equations, symbolTableCollection,
                           matchableIndicesFn))) {
      modelOp.emitError() << "Matching failed for the 'main' model";
      return mlir::failure();
    }
  }

  return mlir::success();
}

DerivativesMap& MatchingPass::getDerivativesMap()
{
  if (auto analysis = getCachedAnalysis<DerivativesMap>()) {
    return *analysis;
  }

  auto& analysis = getAnalysis<DerivativesMap>();
  analysis.initialize();
  return analysis;
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
MatchingPass::getVariableAccessAnalysis(
    EquationInstanceOp equation,
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

    mlir::SymbolRefAttr name;
    IndexSet indices;
  };

  struct EquationBridge
  {
    EquationBridge(
        EquationInstanceOp op,
        mlir::SymbolTableCollection& symbolTable,
        VariableAccessAnalysis& accessAnalysis,
        llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge*>& variablesMap)
        : op(op),
          symbolTable(&symbolTable),
          accessAnalysis(&accessAnalysis),
          variablesMap(&variablesMap)
    {
    }

    EquationInstanceOp op;
    mlir::SymbolTableCollection* symbolTable;
    VariableAccessAnalysis* accessAnalysis;
    llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge*>* variablesMap;
  };
}

namespace marco::modeling::matching
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
      auto numOfExplicitInductions = static_cast<uint64_t>(
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

      if (cachedAccesses) {
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
      }

      return accesses;
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

mlir::LogicalResult MatchingPass::match(
    mlir::OpBuilder& builder,
    ModelOp modelOp,
    llvm::ArrayRef<EquationInstanceOp> equations,
    mlir::SymbolTableCollection& symbolTableCollection,
    llvm::function_ref<IndexSet(mlir::SymbolRefAttr)> matchableIndicesFn)
{
  using MatchingGraph =
      ::marco::modeling::MatchingGraph<::VariableBridge*, ::EquationBridge*>;

  llvm::SmallVector<std::unique_ptr<VariableBridge>> variableBridges;
  llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge*> variablesMap;
  llvm::SmallVector<std::unique_ptr<EquationBridge>> equationBridges;

  // Create the matching graph. We use the pointers to the real nodes in order
  // to speed up the copies.
  MatchingGraph matchingGraph(&getContext());

  for (VariableOp variableOp : modelOp.getVariables()) {
    auto symbolRefAttr = mlir::SymbolRefAttr::get(variableOp.getSymNameAttr());
    IndexSet indices = matchableIndicesFn(symbolRefAttr);

    if (!indices.empty()) {
      auto& bridge = variableBridges.emplace_back(
          std::make_unique<VariableBridge>(symbolRefAttr, std::move(indices)));

      variablesMap[symbolRefAttr] = bridge.get();
      matchingGraph.addVariable(bridge.get());
    }
  }

  for (EquationInstanceOp equation : equations) {
    auto accessAnalysis = getVariableAccessAnalysis(
        equation, symbolTableCollection);

    if (!accessAnalysis) {
      equation.emitOpError() << "Can't obtain access analysis";
      return mlir::failure();
    }

    auto& bridge = equationBridges.emplace_back(
        std::make_unique<EquationBridge>(
            equation, symbolTableCollection, *accessAnalysis, variablesMap));

    matchingGraph.addEquation(bridge.get());
  }

  auto numberOfScalarEquations = matchingGraph.getNumberOfScalarEquations();
  auto numberOfScalarVariables = matchingGraph.getNumberOfScalarVariables();

  if (numberOfScalarEquations < numberOfScalarVariables) {
    modelOp.emitError()
        << "Underdetermined model. Found " << numberOfScalarEquations
        << " scalar equations and " << numberOfScalarVariables
        << " scalar variables.";

    return mlir::failure();
  } else if (numberOfScalarEquations > numberOfScalarVariables) {
    modelOp.emitError()
        << "Overdetermined model. Found " << numberOfScalarEquations
        << " scalar equations and " << numberOfScalarVariables
        << " scalar variables.";

    return mlir::failure();
  }

  if (enableSimplificationAlgorithm) {
    // Apply the simplification algorithm to solve the obliged matches.
    if (!matchingGraph.simplify()) {
      modelOp.emitError()
          << "Inconsistency found during the matching simplification process";

      return mlir::failure();
    }
  }

  // Apply the full matching algorithm for the equations and variables that
  // are still unmatched.
  if (!matchingGraph.match()) {
    modelOp.emitError()
        << "Generic matching algorithm could not solve the matching problem";

    return mlir::failure();
  }

  // Keep track of the old instances to be erased.
  llvm::DenseSet<EquationInstanceOp> toBeErased;

  // Get the matching solution.
  using MatchingSolution = MatchingGraph::MatchingSolution;
  llvm::SmallVector<MatchingSolution> matchingSolutions;

  if (!matchingGraph.getMatch(matchingSolutions)) {
    modelOp.emitOpError() << "Not all the equations have been matched";
    return mlir::failure();
  }

  for (const MatchingSolution& solution : matchingSolutions) {
    const IndexSet& matchedEquationIndices = solution.getIndexes();

    // Scalar equations are masked as for-equations, so there should always be
    // some matched index.
    assert(!matchedEquationIndices.empty());

    const EquationPath& matchedPath = solution.getAccess();
    EquationBridge* equation = solution.getEquation();

    size_t numOfExplicitInductions =
        equation->op.getInductionVariables().size();

    size_t numOfImplicitInductions =
        equation->op.getNumOfImplicitInductionVariables();

    bool isScalarEquation = numOfExplicitInductions == 0;

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(equation->op);

    for (const MultidimensionalRange& matchedEquationRange :
         llvm::make_range(matchedEquationIndices.rangesBegin(),
                          matchedEquationIndices.rangesEnd())) {
      auto matchedEquationOp = builder.create<MatchedEquationInstanceOp>(
          equation->op.getLoc(),
          equation->op.getTemplate(), equation->op.getInitial(),
          EquationPathAttr::get(&getContext(), matchedPath));

      if (!isScalarEquation) {
        MultidimensionalRange explicitRange =
            matchedEquationRange.takeFirstDimensions(numOfExplicitInductions);

        matchedEquationOp.setIndicesAttr(
            MultidimensionalRangeAttr::get(&getContext(), explicitRange));
      }

      if (numOfImplicitInductions > 0) {
        MultidimensionalRange implicitRange =
            matchedEquationRange.takeLastDimensions(numOfImplicitInductions);

        matchedEquationOp.setImplicitIndicesAttr(
            MultidimensionalRangeAttr::get(&getContext(), implicitRange));
      }
    }

    // Mark the old instance as obsolete.
    toBeErased.insert(equation->op);
  }

  // Erase the old equation instances.
  for (EquationInstanceOp equation : toBeErased) {
    equation.erase();
  }

  return mlir::success();
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createMatchingPass()
  {
    return std::make_unique<MatchingPass>();
  }

  std::unique_ptr<mlir::Pass> createMatchingPass(
      const MatchingPassOptions& options)
  {
    return std::make_unique<MatchingPass>(options);
  }
}
