#include "marco/Codegen/Transforms/ModelLegalization.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation/Common.h"
#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include "marco/Codegen/Transforms/ModelSolving/Utils.h"
#include "marco/Codegen/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/ThreadPool.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_MODELLEGALIZATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace
{
  template<typename EqOp>
  struct EquationInterfaceMultipleValuesPattern : public mlir::OpRewritePattern<EqOp>
  {
    using mlir::OpRewritePattern<EqOp>::OpRewritePattern;

    virtual EquationInterface createEmptyEquation(mlir::OpBuilder& builder, mlir::Location loc) const = 0;

    mlir::LogicalResult matchAndRewrite(EqOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto terminator = mlir::cast<EquationSidesOp>(op.bodyBlock()->getTerminator());

      if (terminator.getLhsValues().size() != terminator.getRhsValues().size()) {
        return rewriter.notifyMatchFailure(op, "Different amount of values in left-hand and right-hand sides of the equation");
      }

      auto amountOfValues = terminator.getLhsValues().size();

      for (size_t i = 0; i < amountOfValues; ++i) {
        rewriter.setInsertionPointAfter(op);

        auto clone = createEmptyEquation(rewriter, loc);
        assert(clone.getBodyRegion().empty());
        mlir::Block* cloneBodyBlock = rewriter.createBlock(&clone.getBodyRegion());
        rewriter.setInsertionPointToStart(cloneBodyBlock);

        mlir::BlockAndValueMapping mapping;

        for (auto& originalOp : op.bodyBlock()->getOperations()) {
          if (mlir::isa<EquationSideOp>(originalOp)) {
            continue;
          }

          if (mlir::isa<EquationSidesOp>(originalOp)) {
            auto lhsOp = mlir::cast<EquationSideOp>(terminator.getLhs().getDefiningOp());
            auto rhsOp = mlir::cast<EquationSideOp>(terminator.getRhs().getDefiningOp());

            auto newLhsOp = rewriter.create<EquationSideOp>(lhsOp.getLoc(), mapping.lookup(terminator.getLhsValues()[i]));
            auto newRhsOp = rewriter.create<EquationSideOp>(rhsOp.getLoc(), mapping.lookup(terminator.getRhsValues()[i]));

            rewriter.create<EquationSidesOp>(terminator.getLoc(), newLhsOp, newRhsOp);
          } else {
            rewriter.clone(originalOp, mapping);
          }
        }
      }

      rewriter.eraseOp(op);
      return mlir::success();
    }
  };

  struct EquationOpMultipleValuesPattern : public EquationInterfaceMultipleValuesPattern<EquationOp>
  {
    using EquationInterfaceMultipleValuesPattern<EquationOp>::EquationInterfaceMultipleValuesPattern;

    EquationInterface createEmptyEquation(mlir::OpBuilder& builder, mlir::Location loc) const override
    {
      return builder.create<EquationOp>(loc);
    }
  };

  struct InitialEquationOpMultipleValuesPattern : public EquationInterfaceMultipleValuesPattern<InitialEquationOp>
  {
    using EquationInterfaceMultipleValuesPattern<InitialEquationOp>::EquationInterfaceMultipleValuesPattern;

    EquationInterface createEmptyEquation(mlir::OpBuilder& builder, mlir::Location loc) const override
    {
      return builder.create<InitialEquationOp>(loc);
    }
  };
}

static void createInitializingEquation(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    VariableOp variableOp,
    const IndexSet& indices,
    std::function<mlir::Value(mlir::OpBuilder&, mlir::Location)> valueCallback)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  for (const auto& range : llvm::make_range(indices.rangesBegin(), indices.rangesEnd())) {
    std::vector<mlir::Value> inductionVariables;

    for (unsigned int i = 0; i < range.rank(); ++i) {
      auto forOp = builder.create<ForEquationOp>(loc, range[i].getBegin(), range[i].getEnd() - 1, 1);
      inductionVariables.push_back(forOp.induction());
      builder.setInsertionPointToStart(forOp.bodyBlock());
    }

    auto equationOp = builder.create<EquationOp>(loc);
    assert(equationOp.getBodyRegion().empty());
    mlir::Block* bodyBlock = builder.createBlock(&equationOp.getBodyRegion());
    builder.setInsertionPointToStart(bodyBlock);

    mlir::Value value = valueCallback(builder, loc);

    std::vector<mlir::Value> currentIndices;

    for (unsigned int i = 0; i < variableOp.getVariableType().getRank(); ++i) {
      currentIndices.push_back(inductionVariables[i]);
    }

    mlir::Value variable = builder.create<VariableGetOp>(
        loc, variableOp.getVariableType().unwrap(), variableOp.getSymName());

    if (!currentIndices.empty()) {
      variable = builder.create<LoadOp>(loc, variable, currentIndices);
    }

    mlir::Value lhs = builder.create<EquationSideOp>(loc, variable);
    mlir::Value rhs = builder.create<EquationSideOp>(loc, value);
    builder.create<EquationSidesOp>(loc, lhs, rhs);
  }
}

static mlir::LogicalResult createDerivatives(
    mlir::OpBuilder& builder,
    ModelOp modelOp,
    mlir::SymbolTable& symbolTable,
    DerivativesMap& derivativesMap,
    const llvm::StringMap<IndexSet>& derivedIndices)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  for (const auto& entry : derivedIndices) {
    llvm::StringRef variableName = entry.getKey();
    auto variableOp = symbolTable.lookup<VariableOp>(variableName);
    auto variableType = variableOp.getVariableType();

    auto derType = variableType.withType(RealType::get(builder.getContext()));
    assert(derType.hasStaticShape());

    auto derivedVariableIndices = derivedIndices.find(variableName);
    assert(derivedVariableIndices != derivedIndices.end());

    // Create the variable and initialize it to zero.
    builder.setInsertionPointToStart(modelOp.bodyBlock());

    auto derivativeName = getNextFullDerVariableName(variableOp.getSymName(), 1);

    auto derVariableOp = builder.create<VariableOp>(
        variableOp.getLoc(), derivativeName, derType,
        variableOp.getDimensionsConstraints(),
        nullptr);

    derivativesMap.setDerivative(variableName, derVariableOp.getSymName());
    derivativesMap.setDerivedIndices(variableName, derivedVariableIndices->second);

    symbolTable.insert(derVariableOp);

    // Create the start value
    builder.setInsertionPointToEnd(modelOp.bodyBlock());

    auto startOp = builder.create<StartOp>(
        derVariableOp.getLoc(), derivativeName, false, !derType.isScalar());

    assert(startOp.getBodyRegion().empty());
    mlir::Block* bodyBlock = builder.createBlock(&startOp.getBodyRegion());
    builder.setInsertionPointToStart(bodyBlock);

    mlir::Value zero = builder.create<ConstantOp>(
        derVariableOp.getLoc(), RealAttr::get(builder.getContext(), 0));

    builder.create<YieldOp>(derVariableOp.getLoc(), zero);

    // We need to create additional equations in case of some non-derived indices.
    // If this is not done, then the matching process would fail by detecting an
    // underdetermined problem. An alternative would be to split each variable
    // according to the algebraic / differential nature of its indices, but that
    // is way too complicated with respect to the performance gain.
    builder.setInsertionPointToEnd(modelOp.bodyBlock());
    std::vector<Range> dimensions;

    if (derType.isScalar()) {
      dimensions.emplace_back(0, 1);
    } else {
      for (const auto& dimension : derType.getShape()) {
        dimensions.emplace_back(0, dimension);
      }
    }

    IndexSet allIndices{MultidimensionalRange(dimensions)};
    auto nonDerivedIndices = allIndices - derivedVariableIndices->second;

    createInitializingEquation(builder, derVariableOp.getLoc(), derVariableOp, nonDerivedIndices, [](mlir::OpBuilder& builder, mlir::Location loc) {
      mlir::Value zero = builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), 0));
      return zero;
    });
  }

  return mlir::success();
}

static void eraseValueInsideEquation(mlir::Value value)
{
  std::queue<mlir::Value> queue;
  queue.push(value);

  while (!queue.empty()) {
    std::vector<mlir::Value> valuesWithUses;
    mlir::Value current = queue.front();

    while (current != nullptr && !current.use_empty()) {
      valuesWithUses.push_back(current);
      queue.pop();

      if (queue.empty()) {
        current = nullptr;
      } else {
        current = queue.front();
      }
    }

    for (const auto& valueWithUses : valuesWithUses) {
      queue.push(valueWithUses);
    }

    if (current != nullptr) {
      assert(current.use_empty());

      if (auto op = current.getDefiningOp()) {
        for (auto operand : op->getOperands()) {
          queue.push(operand);
        }

        op->erase();
      }
    }

    queue.pop();
  }
}

static mlir::LogicalResult removeDerOps(
    mlir::OpBuilder& builder,
    ModelOp modelOp,
    const mlir::SymbolTable& symbolTable,
    const DerivativesMap& derivativesMap)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  auto appendIndexesFn = [](std::vector<mlir::Value>& destination, mlir::ValueRange indices) {
    for (size_t i = 0, e = indices.size(); i < e; ++i) {
      mlir::Value index = indices[e - 1 - i];
      destination.push_back(index);
    }
  };

  auto replaceDerOp = [&](DerOp derOp) {
    builder.setInsertionPoint(derOp);

    // If the value to be derived belongs to an array, then also the derived
    // value is stored within an array. Thus, we need to store its position.

    std::vector<mlir::Value> subscriptions;
    mlir::Operation* definingOp = derOp.getOperand().getDefiningOp();

    while (definingOp && !mlir::isa<VariableGetOp>(definingOp)) {
      assert(mlir::isa<LoadOp>(definingOp) || mlir::isa<SubscriptionOp>(definingOp));

      if (auto loadOp = mlir::dyn_cast<LoadOp>(definingOp)) {
        appendIndexesFn(subscriptions, loadOp.getIndices());
        definingOp = loadOp.getArray().getDefiningOp();
      } else {
        auto subscriptionOp = mlir::cast<SubscriptionOp>(definingOp);
        appendIndexesFn(subscriptions, subscriptionOp.getIndices());
        definingOp = subscriptionOp.getSource().getDefiningOp();
      }
    }

    assert(definingOp && mlir::isa<VariableGetOp>(definingOp));
    auto variableGetOp = mlir::cast<VariableGetOp>(definingOp);
    llvm::StringRef variableName = variableGetOp.getVariable();
    llvm::StringRef derivativeName = derivativesMap.getDerivative(variableName);

    auto derivativeVariableOp = symbolTable.lookup<VariableOp>(derivativeName);
    std::vector<mlir::Value> reverted(subscriptions.rbegin(), subscriptions.rend());

    mlir::Value derivative = builder.create<VariableGetOp>(
        derivativeVariableOp.getLoc(),
        derivativeVariableOp.getVariableType().unwrap(),
        derivativeVariableOp.getSymName());

    if (!subscriptions.empty()) {
      derivative = builder.create<SubscriptionOp>(derivative.getLoc(), derivative, reverted);
    }

    if (auto arrayType = derivative.getType().dyn_cast<ArrayType>(); arrayType && arrayType.isScalar()) {
      derivative = builder.create<LoadOp>(derivative.getLoc(), derivative);
    }

    derOp.replaceAllUsesWith(derivative);
  };

  modelOp.getBodyRegion().walk([&](EquationInterface equationInt) {
    std::vector<DerOp> derOps;

    equationInt.walk([&](DerOp derOp) {
      derOps.push_back(derOp);
    });

    for (auto& derOp : derOps) {
      replaceDerOp(derOp);
      eraseValueInsideEquation(derOp.getResult());
    }
  });

  for (AlgorithmOp algorithmOp : modelOp.getOps<AlgorithmOp>()) {
    std::vector<DerOp> derOps;

    algorithmOp.walk([&](DerOp derOp) {
      derOps.push_back(derOp);
    });

    for (auto& derOp : derOps) {
      replaceDerOp(derOp);
      derOp.erase();
    }
  }

  return mlir::success();
}

namespace
{
  class ModelLegalization
  {
    public:
      ModelLegalization(ModelOp modelOp, bool debugView)
        : modelOp(modelOp), debugView(debugView)
      {
      }

      mlir::LogicalResult run();

    private:
      void collectDerivedVariablesIndices(
        llvm::StringMap<IndexSet>& derivedIndices,
        const Equations<Equation>& equations);

      void collectDerivedVariablesIndices(
          llvm::StringMap<IndexSet>& derivedIndices,
          llvm::ArrayRef<AlgorithmOp> algorithmOps);

      // Convert the binding equations into equations or StartOps.
      mlir::LogicalResult convertBindingEquations(
          const mlir::SymbolTable& symbolTable);

      /// Add a StartOp for each variable not having one.
      mlir::LogicalResult addMissingStartOps(mlir::OpBuilder& builder);

      mlir::LogicalResult convertAlgorithmsIntoEquations(
          mlir::OpBuilder& builder, const mlir::SymbolTable& symbolTable);

      /// For each EquationOp, create an InitialEquationOp with the same body.
      mlir::LogicalResult cloneEquationsAsInitialEquations(
          mlir::OpBuilder& builder);

      /// Create the initial equations for the variables with a fixed
      /// initialization value.
      mlir::LogicalResult createInitialEquationsFromFixedStartOps(
          mlir::OpBuilder& builder, const mlir::SymbolTable& symbolTable);

      mlir::LogicalResult convertEquationsWithMultipleValues();

      mlir::LogicalResult convertToSingleEquationBody();

    private:
      ModelOp modelOp;
      bool debugView;

      llvm::ThreadPool threadPool;
  };
}

mlir::LogicalResult ModelLegalization::run()
{
  mlir::SymbolTable symbolTable(modelOp);
  mlir::OpBuilder builder(modelOp);

  // Convert the binding equations.
  if (mlir::failed(convertBindingEquations(symbolTable))) {
    return mlir::failure();
  }

  // The initial conditions are determined by resolving a separate model, with
  // indeed more equations than the model used during the simulation loop.
  Model<Equation> initialConditionsModel(modelOp);
  Model<Equation> mainModel(modelOp);

  initialConditionsModel.setVariables(
      discoverVariables(initialConditionsModel.getOperation()));

  initialConditionsModel.setEquations(
      discoverInitialEquations(
          initialConditionsModel.getOperation(),
          initialConditionsModel.getVariables()));

  mainModel.setVariables(discoverVariables(mainModel.getOperation()));

  mainModel.setEquations(
      discoverEquations(mainModel.getOperation(), mainModel.getVariables()));

  // A map that keeps track of which indices of a variable do appear under
  // the derivative operation.
  llvm::StringMap<IndexSet> derivedIndices;

  // Add a 'start' value of zero for the variables for which an explicit
  // 'start' value has not been provided.

  if (mlir::failed(addMissingStartOps(builder))) {
    return mlir::failure();
  }

  // Create the initial equations given by the start values having also
  // the fixed attribute set to 'true'.
  if (mlir::failed(createInitialEquationsFromFixedStartOps(
          builder, symbolTable))) {
    return mlir::failure();
  }

  // Determine which scalar variables do appear as argument to the derivative
  // operation.
  collectDerivedVariablesIndices(
      derivedIndices, initialConditionsModel.getEquations());

  collectDerivedVariablesIndices(derivedIndices, mainModel.getEquations());

  // Discover the derivatives inside the algorithms.
  llvm::SmallVector<AlgorithmOp> algorithmOps;

  for (AlgorithmOp algorithmOp : modelOp.getOps<AlgorithmOp>()) {
    algorithmOps.push_back(algorithmOp);
  }

  collectDerivedVariablesIndices(derivedIndices, algorithmOps);

  // Create the variables for the derivatives, together with the initial
  // equations needed to initialize them to zero.
  DerivativesMap derivativesMap;

  if (mlir::failed(readDerivativesMap(modelOp, derivativesMap))) {
    return mlir::failure();
  }

  if (mlir::failed(createDerivatives(
          builder, modelOp, symbolTable, derivativesMap, derivedIndices))) {
    return mlir::failure();
  }

  // The derivatives mapping is now complete, thus we can set the derivatives
  // map inside the models.
  initialConditionsModel.setDerivativesMap(derivativesMap);
  mainModel.setDerivativesMap(derivativesMap);

  // Remove the derivative operations.
  if (mlir::failed(removeDerOps(builder, modelOp, symbolTable, derivativesMap))) {
    return mlir::failure();
  }

  // Convert the algorithms into equations.
  if (mlir::failed(convertAlgorithmsIntoEquations(builder, symbolTable))) {
    return mlir::failure();
  }

  // Clone the equations as initial equations, in order to use them when
  // computing the initial values of the variables.

  if (mlir::failed(cloneEquationsAsInitialEquations(builder))) {
    return mlir::failure();
  }

  if (mlir::failed(convertEquationsWithMultipleValues())) {
    return mlir::failure();
  }

  // Split the loops containing more than one operation within their bodies.
  if (mlir::failed(convertToSingleEquationBody())) {
    return mlir::failure();
  }

  // We need to perform again the discovery process for both variables and
  // equations. Variables may have changed due to derivatives, while new
  // equations may have been created for multiple reasons (e.g. algorithms).

  initialConditionsModel.setVariables(
      discoverVariables(initialConditionsModel.getOperation()));

  initialConditionsModel.setEquations(
      discoverInitialEquations(
          initialConditionsModel.getOperation(),
          initialConditionsModel.getVariables()));

  mainModel.setVariables(discoverVariables(mainModel.getOperation()));

  mainModel.setEquations(
      discoverEquations(mainModel.getOperation(), mainModel.getVariables()));

  // Store the information about derivatives in form of attribute.
  ModelSolvingIROptions irOptions;
  irOptions.mergeAndSortRanges = debugView;

  writeDerivativesMap(builder, modelOp, symbolTable, derivativesMap, irOptions);

  return mlir::success();
}

static void collectDerivedVariablesIndices(
    llvm::StringMap<IndexSet>& derivedIndices,
    const std::unique_ptr<Equation>& equation,
    std::mutex& mutex)
{
  auto accesses = equation->getAccesses();

  equation->getOperation().walk([&](DerOp derOp) {
    auto it = llvm::find_if(accesses, [&](const auto& access) {
      auto value = equation->getValueAtPath(access.getPath());
      return value == derOp.getOperand();
    });

    assert(it != accesses.end());
    const auto& access = *it;
    auto indices = access.getAccessFunction().map(equation->getIterationRanges());
    llvm::StringRef variableName = access.getVariable()->getDefiningOp().getSymName();

    std::lock_guard<std::mutex> lock(mutex);
    derivedIndices[variableName] += indices;
  });
}

void ModelLegalization::collectDerivedVariablesIndices(
    llvm::StringMap<IndexSet>& derivedIndices,
    const Equations<Equation>& equations)
{
  std::mutex mutex;

  size_t numOfEquations = equations.size();
  std::atomic_size_t currentEquation = 0;

  // Function to process a chunk of equations.
  auto mapFn = [&]() {
    size_t i = currentEquation++;

    while (i < numOfEquations) {
      ::collectDerivedVariablesIndices(derivedIndices, equations[i], mutex);
      i = currentEquation++;
    }
  };

  // Shard the work among multiple threads.
  unsigned int numOfThreads = threadPool.getThreadCount();
  llvm::ThreadPoolTaskGroup tasks(threadPool);

  for (unsigned int i = 0; i < numOfThreads; ++i) {
    tasks.async(mapFn);
  }

  // Wait for all the tasks to finish.
  tasks.wait();
}

static void collectDerivedVariablesIndices(
    llvm::StringMap<IndexSet>& derivedIndices,
    AlgorithmOp algorithmOp,
    std::mutex& mutex)
{
  algorithmOp.walk([&](DerOp derOp) {
    mlir::Operation* definingOp = derOp.getOperand().getDefiningOp();

    while (definingOp && !mlir::isa<VariableGetOp>(definingOp)) {
      if (auto loadOp = mlir::dyn_cast<LoadOp>(definingOp)) {
        definingOp = loadOp.getArray().getDefiningOp();
      } else if (auto subscriptionOp = mlir::dyn_cast<SubscriptionOp>(definingOp)) {
        definingOp = subscriptionOp.getSource().getDefiningOp();
      } else {
        break;
      }
    }

    assert(definingOp && mlir::isa<VariableGetOp>(definingOp));
    auto variableGetOp = mlir::cast<VariableGetOp>(definingOp);
    IndexSet indices;

    if (auto arrayType = variableGetOp.getType().dyn_cast<ArrayType>()) {
      assert(arrayType.hasStaticShape());
      std::vector<Range> ranges;

      for (const auto& dimension : arrayType.getShape()) {
        ranges.emplace_back(0, dimension);
      }

      indices += MultidimensionalRange(ranges);
    } else {
      indices += Point(0);
    }

    llvm::StringRef variableName = variableGetOp.getVariable();

    std::lock_guard<std::mutex> lock(mutex);
    derivedIndices[variableName] += indices;
  });
}

void ModelLegalization::collectDerivedVariablesIndices(
    llvm::StringMap<IndexSet>& derivedIndices,
    llvm::ArrayRef<AlgorithmOp> algorithmOps)
{
  std::mutex mutex;

  size_t numOfAlgorithms = algorithmOps.size();
  std::atomic_size_t currentAlgorithm = 0;

  // Function to process a chunk of algorithms.
  auto mapFn = [&]() {
    size_t i = currentAlgorithm++;

    while (i < numOfAlgorithms) {
      ::collectDerivedVariablesIndices(derivedIndices, algorithmOps[i], mutex);
      i = currentAlgorithm++;
    }
  };

  // Shard the work among multiple threads.
  unsigned int numOfThreads = threadPool.getThreadCount();
  llvm::ThreadPoolTaskGroup tasks(threadPool);

  for (unsigned int i = 0; i < numOfThreads; ++i) {
    tasks.async(mapFn);
  }

  // Wait for all the tasks to finish.
  tasks.wait();
}

namespace
{
  struct BindingEquationOpPattern
      : public mlir::OpRewritePattern<BindingEquationOp>
  {
    BindingEquationOpPattern(
        mlir::MLIRContext* context,
        const mlir::SymbolTable& symbolTable)
        : mlir::OpRewritePattern<BindingEquationOp>(context),
          symbolTable(&symbolTable)
    {
    }

    protected:
      const mlir::SymbolTable* symbolTable;
  };

  struct BindingEquationOpToEquationOpPattern : public BindingEquationOpPattern
  {
    using BindingEquationOpPattern::BindingEquationOpPattern;

    mlir::LogicalResult match(BindingEquationOp op) const override
    {
      auto variable = symbolTable->lookup<VariableOp>(op.getVariable());
      return mlir::LogicalResult::success(!variable.isReadOnly());
    }

    void rewrite(
        BindingEquationOp op,
        mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      auto yieldOp = mlir::cast<YieldOp>(
          op.getBodyRegion().back().getTerminator());

      assert(yieldOp.getValues().size() == 1);
      mlir::Value expression = yieldOp.getValues()[0];

      auto variable = symbolTable->lookup<VariableOp>(op.getVariable());
      auto variableType = variable.getVariableType();
      mlir::Type unwrappedType = variableType.unwrap();
      auto expressionType = expression.getType();

      std::vector<mlir::Value> inductionVariables;

      if (auto variableArrayType = unwrappedType.dyn_cast<ArrayType>()) {
        unsigned int expressionRank = 0;

        if (auto expressionArrayType = expressionType.dyn_cast<ArrayType>()) {
          expressionRank = expressionArrayType.getRank();
        }

        auto variableRank = variableArrayType.getRank();
        assert(variableArrayType.hasStaticShape());
        assert(expressionRank == variableRank);

        for (unsigned int i = 0; i < variableRank; ++i) {
          auto forEquationOp = rewriter.create<ForEquationOp>(
              loc, 0, variableArrayType.getShape()[i] - 1, 1);

          inductionVariables.push_back(forEquationOp.induction());
          rewriter.setInsertionPointToStart(forEquationOp.bodyBlock());
        }
      }

      auto equationOp = rewriter.create<EquationOp>(loc);
      assert(equationOp.getBodyRegion().empty());

      rewriter.setInsertionPointToStart(&op.getBodyRegion().front());

      mlir::Value lhsValue = rewriter.create<VariableGetOp>(
          loc, unwrappedType, op.getVariable());

      if (!inductionVariables.empty()) {
        lhsValue = rewriter.create<LoadOp>(loc, lhsValue, inductionVariables);
      }

      mlir::Value rhsValue = expression;
      rewriter.setInsertionPointAfterValue(rhsValue);

      if (!inductionVariables.empty()) {
        rhsValue = rewriter.create<LoadOp>(loc, rhsValue, inductionVariables);
      }

      rewriter.setInsertionPoint(yieldOp);

      mlir::Value lhsTuple = rewriter.create<EquationSideOp>(loc, lhsValue);
      mlir::Value rhsTuple = rewriter.create<EquationSideOp>(loc, rhsValue);
      rewriter.create<EquationSidesOp>(loc, lhsTuple, rhsTuple);

      rewriter.eraseOp(yieldOp);

      rewriter.inlineRegionBefore(
          op.getBodyRegion(),
          equationOp.getBodyRegion(),
          equationOp.getBodyRegion().end());

      rewriter.eraseOp(op);
    }
  };

  struct BindingEquationOpToStartOpPattern : public BindingEquationOpPattern
  {
    using BindingEquationOpPattern::BindingEquationOpPattern;

    mlir::LogicalResult match(BindingEquationOp op) const override
    {
      auto variable = symbolTable->lookup<VariableOp>(op.getVariable());
      return mlir::LogicalResult::success(variable.isReadOnly());
    }

    void rewrite(
        BindingEquationOp op,
        mlir::PatternRewriter& rewriter) const override
    {
      auto startOp = rewriter.replaceOpWithNewOp<StartOp>(
          op, op.getVariable(), true, false);

      assert(startOp.getBodyRegion().empty());

      rewriter.inlineRegionBefore(
          op.getBodyRegion(),
          startOp.getBodyRegion(),
          startOp.getBodyRegion().end());
    }
  };
}

mlir::LogicalResult ModelLegalization::convertBindingEquations(
    const mlir::SymbolTable& symbolTable)
{
  mlir::ConversionTarget target(*modelOp.getContext());
  target.addIllegalOp<BindingEquationOp>();

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  mlir::RewritePatternSet patterns(modelOp.getContext());

  patterns.insert<
      BindingEquationOpToEquationOpPattern,
      BindingEquationOpToStartOpPattern>(modelOp.getContext(), symbolTable);

  return applyPartialConversion(modelOp, target, std::move(patterns));
}

mlir::LogicalResult ModelLegalization::addMissingStartOps(
    mlir::OpBuilder& builder)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  llvm::StringMap<StartOp> startOps;

  for (StartOp startOp : modelOp.getOps<StartOp>()) {
    startOps[startOp.getVariable()] = startOp;
  }

  for (VariableOp variableOp : modelOp.getOps<VariableOp>()) {
    if (startOps.find(variableOp.getSymName()) == startOps.end()) {
      builder.setInsertionPointToEnd(modelOp.bodyBlock());
      VariableType variableType = variableOp.getVariableType();
      bool each = !variableType.isScalar();

      auto startOp = builder.create<StartOp>(
          variableOp.getLoc(), variableOp.getSymName(), false, each);

      assert(startOp.getBodyRegion().empty());
      mlir::Block* bodyBlock = builder.createBlock(&startOp.getBodyRegion());
      builder.setInsertionPointToStart(bodyBlock);

      mlir::Value zero = builder.create<ConstantOp>(
          variableOp.getLoc(), getZeroAttr(variableType.getElementType()));

      builder.create<YieldOp>(variableOp.getLoc(), zero);
    }
  }

  return mlir::success();
}

namespace
{
  template<typename Op>
  class AlgorithmInterfacePattern : public mlir::OpRewritePattern<Op>
  {
    public:
      AlgorithmInterfacePattern(
          mlir::MLIRContext* context,
          const mlir::SymbolTable& symbolTable,
          const llvm::StringMap<StartOp>& startOps)
          : mlir::OpRewritePattern<Op>(context),
            symbolTable(&symbolTable),
            startOps(&startOps)
      {
      }

      mlir::LogicalResult matchAndRewrite(
          Op op,
          mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        auto algorithmInt = mlir::cast<AlgorithmInterface>(op.getOperation());
        auto modelOp = algorithmInt->template getParentOfType<ModelOp>();
        auto moduleOp = modelOp->template getParentOfType<mlir::ModuleOp>();

        // Determine the read and written variables.
        llvm::DenseSet<VariableOp> readVariables;
        llvm::DenseSet<VariableOp> writtenVariables;

        algorithmInt.walk([&](VariableGetOp getOp) {
          VariableOp variableOp =
              symbolTable->lookup<VariableOp>(getOp.getVariable());

          if (variableOp.getVariableType().isScalar()) {
            readVariables.insert(variableOp);
          } else {
            bool isRead, isWritten;
            std::tie(isRead, isWritten) = determineReadWrite(getOp.getResult());

            if (isRead) {
              readVariables.insert(variableOp);
            } else if (isWritten) {
              writtenVariables.insert(variableOp);
            }
          }
        });

        algorithmInt.walk([&](VariableSetOp setOp) {
          VariableOp variableOp =
              symbolTable->lookup<VariableOp>(setOp.getVariable());

          writtenVariables.insert(variableOp);
        });

        // Determine the input and output variables of the function.
        // If a variable is read, but not written, then it will be an argument
        // of the function. All written variables are results of the function.
        llvm::SmallVector<VariableOp> inputVariables;
        llvm::SmallVector<VariableOp> outputVariables(
            writtenVariables.begin(), writtenVariables.end());

        for (VariableOp readVariable : readVariables) {
          if (!writtenVariables.contains(readVariable)) {
            inputVariables.push_back(readVariable);
          }
        }

        // Obtain a unique name for the function to be created.
        std::string functionName = getFunctionName(modelOp);

        // Create the function.
        rewriter.setInsertionPointToEnd(moduleOp.getBody());

        auto functionOp = rewriter.create<FunctionOp>(loc, functionName);
        mlir::Block* entryBlock = rewriter.createBlock(&functionOp.getBody());

        // Declare the variables.
        rewriter.setInsertionPointToStart(entryBlock);
        mlir::BlockAndValueMapping mapping;

        for (VariableOp variableOp : inputVariables) {
          auto clonedVariableOp = mlir::cast<VariableOp>(
              rewriter.clone(*variableOp.getOperation(), mapping));

          auto originalVariableType = variableOp.getVariableType();

          clonedVariableOp.setType(VariableType::get(
              originalVariableType.getShape(),
              originalVariableType.getElementType(),
              VariabilityProperty::none,
              IOProperty::input));
        }

        for (VariableOp variableOp : outputVariables) {
          auto clonedVariableOp = mlir::cast<VariableOp>(
              rewriter.clone(*variableOp.getOperation(), mapping));

          auto originalVariableType = variableOp.getVariableType();

          clonedVariableOp.setType(VariableType::get(
              originalVariableType.getShape(),
              originalVariableType.getElementType(),
              VariabilityProperty::none,
              IOProperty::output));
        }

        // Set the default value of the output variables.
        for (VariableOp variableOp : outputVariables) {
          auto startOpIt = startOps->find(variableOp.getSymName());
          assert(startOpIt != startOps->end());
          StartOp startOp = startOpIt->getValue();

          auto defaultOp = rewriter.create<DefaultOp>(
              startOp.getLoc(), variableOp.getSymName());

          rewriter.cloneRegionBefore(
              startOp.getBodyRegion(),
              defaultOp.getBodyRegion(),
              defaultOp.getBodyRegion().end());

          if (startOp.getEach()) {
            mlir::OpBuilder::InsertionGuard guard(rewriter);

            auto yieldOp = mlir::cast<YieldOp>(
                defaultOp.getBodyRegion().back().getTerminator());

            assert(yieldOp.getValues().size() == 1);
            rewriter.setInsertionPoint(yieldOp);

            mlir::Value array = rewriter.create<ArrayBroadcastOp>(
                yieldOp.getLoc(),
                variableOp.getVariableType().unwrap(),
                yieldOp.getValues()[0]);

            rewriter.replaceOpWithNewOp<YieldOp>(yieldOp, array);
          }
        }

        // Create the algorithm inside the function and move the original body
        // into it.
        rewriter.setInsertionPointToEnd(entryBlock);
        auto algorithmOp = rewriter.create<AlgorithmOp>(loc);

        rewriter.inlineRegionBefore(
            algorithmInt->getRegion(0),
            algorithmOp.getBodyRegion(),
            algorithmOp.getBodyRegion().end());

        // Create the equation containing the call to the function.
        rewriter.setInsertionPointToEnd(modelOp.bodyBlock());
        mlir::Region* equationRegion = createEquation(rewriter, loc);
        rewriter.setInsertionPointToStart(&equationRegion->front());

        llvm::SmallVector<mlir::Value> inputVariableGetOps;
        llvm::SmallVector<mlir::Value> outputVariableGetOps;

        for (VariableOp inputVariable : inputVariables) {
          inputVariableGetOps.push_back(rewriter.create<VariableGetOp>(
              loc,
              inputVariable.getVariableType().unwrap(),
              inputVariable.getSymName()));
        }

        for (VariableOp outputVariable : outputVariables) {
          outputVariableGetOps.push_back(rewriter.create<VariableGetOp>(
              loc,
              outputVariable.getVariableType().unwrap(),
              outputVariable.getSymName()));
        }

        auto callOp = rewriter.create<CallOp>(
            loc, functionName,
            mlir::ValueRange(outputVariableGetOps).getTypes(),
            inputVariableGetOps);

        mlir::Value lhs = rewriter.create<EquationSideOp>(
            loc, outputVariableGetOps);

        mlir::Value rhs = rewriter.create<EquationSideOp>(
            loc, callOp.getResults());

        rewriter.create<EquationSidesOp>(loc, lhs, rhs);

        // Erase the algorithm.
        rewriter.eraseOp(op);

        return mlir::success();
      }

      virtual std::string getFunctionName(ModelOp modelOp) const = 0;

      virtual mlir::Region* createEquation(
          mlir::OpBuilder& builder, mlir::Location loc) const = 0;

    private:
      /// Determine if an array is read or written.
      /// The return value consists in pair of boolean values, respectively
      /// indicating whether the array is read and written.
      std::pair<bool, bool> determineReadWrite(mlir::Value array) const
      {
        assert(array.getType().isa<ArrayType>());

        bool read = false;
        bool write = false;

        std::stack<mlir::Value> aliases;
        aliases.push(array);

        auto shouldStopEarly = [&read, &write]() {
          // Stop early if both a read and write have been found.
          return read && write;
        };

        // Keep the vector outside the loop, in order to avoid a stack overflow
        llvm::SmallVector<mlir::SideEffects::EffectInstance<
            mlir::MemoryEffects::Effect>> effects;

        while (!aliases.empty() && !shouldStopEarly()) {
          mlir::Value alias = aliases.top();
          aliases.pop();

          std::stack<mlir::Operation*> ops;

          for (const auto& user : alias.getUsers()) {
            ops.push(user);
          }

          while (!ops.empty() && !shouldStopEarly()) {
            mlir::Operation* op = ops.top();
            ops.pop();

            effects.clear();

            if (auto memoryInterface =
                    mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
              memoryInterface.getEffectsOnValue(alias, effects);

              read |= llvm::any_of(effects, [](const auto& effect) {
                return mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect());
              });

              write |= llvm::any_of(effects, [](const auto& effect) {
                return mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect());
              });
            } else if (auto viewInterface =
                           mlir::dyn_cast<mlir::ViewLikeOpInterface>(op)) {
              if (viewInterface.getViewSource() == alias) {
                for (const auto& result : viewInterface->getResults()) {
                  aliases.push(result);
                }
              }
            }
          }
        }

        return std::make_pair(read, write);
      }

    private:
      const mlir::SymbolTable* symbolTable;
      const llvm::StringMap<StartOp>* startOps;
  };

  class AlgorithmOpPattern : public AlgorithmInterfacePattern<AlgorithmOp>
  {
    public:
      AlgorithmOpPattern(
          mlir::MLIRContext* context,
          const mlir::SymbolTable& symbolTable,
          const llvm::StringMap<StartOp>& startOps,
          size_t& functionsCounter)
          : AlgorithmInterfacePattern(context, symbolTable, startOps),
            functionsCounter(&functionsCounter)
      {
      }

      std::string getFunctionName(ModelOp modelOp) const override
      {
        return modelOp.getSymName().str() +
            "_algorithm_" + std::to_string((*functionsCounter)++);
      }

      mlir::Region* createEquation(
          mlir::OpBuilder& builder, mlir::Location loc) const override
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        auto op = builder.create<EquationOp>(loc);
        assert(op.getBodyRegion().empty());
        builder.createBlock(&op.getBodyRegion());
        return &op.getBodyRegion();
      }

    private:
      size_t* functionsCounter;
  };

  class InitialAlgorithmOpPattern
      : public AlgorithmInterfacePattern<InitialAlgorithmOp>
  {
    public:
      InitialAlgorithmOpPattern(
          mlir::MLIRContext* context,
          const mlir::SymbolTable& symbolTable,
          const llvm::StringMap<StartOp>& startOps,
          size_t& functionsCounter)
          : AlgorithmInterfacePattern(context, symbolTable, startOps),
            functionsCounter(&functionsCounter)
      {
      }

      std::string getFunctionName(ModelOp modelOp) const override
      {
        return modelOp.getSymName().str() +
            "_initial_algorithm_" + std::to_string((*functionsCounter)++);
      }

      mlir::Region* createEquation(
          mlir::OpBuilder& builder, mlir::Location loc) const override
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        auto op = builder.create<InitialEquationOp>(loc);
        assert(op.getBodyRegion().empty());
        builder.createBlock(&op.getBodyRegion());
        return &op.getBodyRegion();
      }

    private:
      size_t* functionsCounter;
  };
}

mlir::LogicalResult ModelLegalization::convertAlgorithmsIntoEquations(
    mlir::OpBuilder& builder, const mlir::SymbolTable& symbolTable)
{
  mlir::ConversionTarget target(*modelOp.getContext());
  target.addIllegalOp<InitialAlgorithmOp>();

  target.addDynamicallyLegalOp<AlgorithmOp>([](AlgorithmOp op) {
    return !mlir::isa<ModelOp>(op->getParentOp());
  });

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  mlir::RewritePatternSet patterns(modelOp.getContext());

  // Map the StartOps.
  llvm::StringMap<StartOp> startOps;

  for (StartOp startOp : modelOp.getOps<StartOp>()) {
    startOps[startOp.getVariable()] = startOp;
  }

  // Counters for uniqueness of functions.
  size_t algorithmsCounter = 0;
  size_t initialAlgorithmsCounter = 0;

  patterns.insert<AlgorithmOpPattern>(
      modelOp.getContext(), symbolTable, startOps, algorithmsCounter);

  patterns.insert<InitialAlgorithmOpPattern>(
      modelOp.getContext(), symbolTable, startOps, initialAlgorithmsCounter);

  auto res = applyPartialConversion(modelOp, target, std::move(patterns));

  if (mlir::failed(res)) {
    llvm::errs() << "FAIL\n";
  }

  return res;
}

mlir::LogicalResult ModelLegalization::cloneEquationsAsInitialEquations(
    mlir::OpBuilder& builder)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  // Collect the equations.
  std::vector<EquationOp> equationOps;

  modelOp.bodyBlock()->walk([&](EquationOp equationOp) {
    equationOps.push_back(equationOp);
  });

  for (auto& equationOp : equationOps) {
    // The new initial equation is placed right after the original equation.
    // In this way, there is no need to clone also the wrapping loops.
    builder.setInsertionPointAfter(equationOp);

    mlir::BlockAndValueMapping mapping;

    // Create the initial equation and clone the original equation body
    auto initialEquationOp =
        builder.create<InitialEquationOp>(equationOp.getLoc());

    initialEquationOp->setAttrs(equationOp->getAttrDictionary());

    assert(initialEquationOp.getBodyRegion().empty());

    mlir::Block* bodyBlock = builder.createBlock(
        &initialEquationOp.getBodyRegion());

    builder.setInsertionPointToStart(bodyBlock);

    for (auto& op : equationOp.bodyBlock()->getOperations()) {
      builder.clone(op, mapping);
    }
  }

  return mlir::success();
}

mlir::LogicalResult ModelLegalization::createInitialEquationsFromFixedStartOps(
    mlir::OpBuilder& builder, const mlir::SymbolTable& symbolTable)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  // Collect the start operations having the 'fixed' attribute set to true.
  std::vector<StartOp> startOps;

  for (StartOp startOp : modelOp.getOps<StartOp>()) {
    if (startOp.getFixed()) {
      startOps.push_back(startOp);
    }
  }

  // Process each StartOp.
  for (auto& startOp : startOps) {
    builder.setInsertionPointToEnd(modelOp.bodyBlock());

    mlir::Location loc = startOp.getLoc();
    auto variable = symbolTable.lookup<VariableOp>(startOp.getVariable());
    auto variableType = variable.getVariableType();

    unsigned int expressionRank = 0;

    auto yieldOp =  mlir::cast<YieldOp>(
        startOp.getBodyRegion().back().getTerminator());

    mlir::Value expressionValue = yieldOp.getValues()[0];

    if (auto expressionArrayType =
            expressionValue.getType().dyn_cast<ArrayType>()) {
      expressionRank = expressionArrayType.getRank();
    }

    auto variableRank = variableType.getRank();
    assert(expressionRank == 0 || expressionRank == variableRank);

    std::vector<mlir::Value> inductionVariables;

    for (unsigned int i = 0; i < variableRank - expressionRank; ++i) {
      auto forEquationOp = builder.create<ForEquationOp>(
          loc, 0, variableType.getShape()[i] - 1, 1);

      inductionVariables.push_back(forEquationOp.induction());
      builder.setInsertionPointToStart(forEquationOp.bodyBlock());
    }

    auto equationOp = builder.create<InitialEquationOp>(loc);
    assert(equationOp.getBodyRegion().empty());

    mlir::Block* equationBodyBlock = builder.createBlock(
        &equationOp.getBodyRegion());

    builder.setInsertionPointToStart(equationBodyBlock);

    // Left-hand side.
    mlir::Value lhsValue = builder.create<VariableGetOp>(
        startOp.getLoc(), variableType.unwrap(), startOp.getVariable());

    if (!inductionVariables.empty()) {
      if (inductionVariables.size() ==
          static_cast<size_t>(variableType.getRank())) {
        lhsValue = builder.create<LoadOp>(loc, lhsValue, inductionVariables);
      } else {
        lhsValue = builder.create<SubscriptionOp>(
            loc, lhsValue, inductionVariables);
      }
    }

    // Clone the operations.
    mlir::BlockAndValueMapping mapping;

    for (auto& op : startOp.getOps()) {
      if (!mlir::isa<YieldOp>(op)) {
        builder.clone(op, mapping);
      }
    }

    // Right-hand side.
    mlir::Value rhsValue = mapping.lookup(yieldOp.getValues()[0]);

    // Create the assignment.
    mlir::Value lhsTuple = builder.create<EquationSideOp>(loc, lhsValue);
    mlir::Value rhsTuple = builder.create<EquationSideOp>(loc, rhsValue);
    builder.create<EquationSidesOp>(loc, lhsTuple, rhsTuple);
  }

  return mlir::success();
}

mlir::LogicalResult ModelLegalization::convertEquationsWithMultipleValues()
{
  mlir::ConversionTarget target(*modelOp.getContext());

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  target.addDynamicallyLegalOp<EquationOp>([](EquationOp op) {
    auto terminator = mlir::cast<EquationSidesOp>(op.bodyBlock()->getTerminator());
    return terminator.getLhsValues().size() == 1 && terminator.getRhsValues().size() == 1;
  });

  target.addDynamicallyLegalOp<InitialEquationOp>([](InitialEquationOp op) {
    auto terminator = mlir::cast<EquationSidesOp>(op.bodyBlock()->getTerminator());
    return terminator.getLhsValues().size() == 1 && terminator.getRhsValues().size() == 1;
  });

  mlir::RewritePatternSet patterns(modelOp.getContext());
  patterns.insert<EquationOpMultipleValuesPattern>(modelOp.getContext());
  patterns.insert<InitialEquationOpMultipleValuesPattern>(modelOp.getContext());

  return applyPartialConversion(modelOp, target, std::move(patterns));
}

mlir::LogicalResult ModelLegalization::convertToSingleEquationBody()
{
  mlir::OpBuilder builder(modelOp);
  std::vector<EquationInterface> equations;

  // Collect all the equations inside the region
  modelOp.walk([&](EquationInterface op) {
    equations.push_back(op);
  });

  mlir::BlockAndValueMapping mapping;

  for (auto& equation : equations) {
    builder.setInsertionPointToEnd(modelOp.bodyBlock());
    std::vector<ForEquationOp> parents;

    // Collect the wrapping loops
    auto parent = equation->getParentOfType<ForEquationOp>();

    while (parent != nullptr) {
      parents.push_back(parent);
      parent = parent->getParentOfType<ForEquationOp>();
    }

    if (parents.empty()) {
      builder.setInsertionPointAfter(equation);
    } else {
      builder.setInsertionPointAfter(parents.back());
    }

    // Clone them starting from the outermost one
    for (size_t i = 0, e = parents.size(); i < e; ++i) {
      ForEquationOp oldParent = parents[e - i - 1];

      auto newParent = builder.create<ForEquationOp>(
          oldParent.getLoc(),
          oldParent.getFromAttr().getInt(),
          oldParent.getToAttr().getInt(),
          oldParent.getStepAttr().getInt());

      mapping.map(oldParent.induction(), newParent.induction());
      builder.setInsertionPointToStart(newParent.bodyBlock());
    }

    builder.clone(*equation.getOperation(), mapping);
  }

  // Erase the old equations
  for (auto& equation : equations) {
    auto parent = equation->getParentOfType<ForEquationOp>();
    equation.erase();

    while (parent != nullptr && parent.bodyBlock()->empty()) {
      auto newParent = parent->getParentOfType<ForEquationOp>();
      parent.erase();
      parent = newParent;
    }
  }

  return mlir::success();
}

namespace
{
  class ModelLegalizationPass
      : public mlir::modelica::impl::ModelLegalizationPassBase<
          ModelLegalizationPass>
  {
    public:
      using ModelLegalizationPassBase::ModelLegalizationPassBase;

      void runOnOperation() override
      {
        mlir::OpBuilder builder(getOperation());
        llvm::SmallVector<ModelOp, 1> modelOps;

        for (ModelOp modelOp : getOperation().getOps<ModelOp>()) {
          if (modelOp.getSymName() == modelName) {
            modelOps.push_back(modelOp);
          }
        }

        for (ModelOp modelOp : modelOps) {
          ModelLegalization instance(modelOp, debugView);

          if (mlir::failed(instance.run())) {
            return signalPassFailure();
          }
        }
      }
  };
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createModelLegalizationPass()
  {
    return std::make_unique<ModelLegalizationPass>();
  }

  std::unique_ptr<mlir::Pass> createModelLegalizationPass(
      const ModelLegalizationPassOptions& options)
  {
    return std::make_unique<ModelLegalizationPass>(options);
  }
}
