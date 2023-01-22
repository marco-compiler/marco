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

/// Determine if an array is read or written.
/// The return value consists in pair of boolean values, respectively
/// indicating whether the array is read and written.
static std::pair<bool, bool> determineReadWrite(mlir::Value array)
{
  assert(array.getType().isa<ArrayType>());

  bool read = false;
  bool write = false;

  std::stack<mlir::Value> aliases;
  aliases.push(array);

  auto shouldStopEarly = [&]() {
    // Stop early if both a read and write have been found
    return read && write;
  };

  // Keep the vector outside the loop, in order to avoid a stack overflow
  llvm::SmallVector<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> effects;

  while (!aliases.empty() && !shouldStopEarly()) {
    auto alias = aliases.top();
    aliases.pop();

    std::stack<mlir::Operation*> ops;

    for (const auto& user : alias.getUsers()) {
      ops.push(user);
    }

    while (!ops.empty() && !shouldStopEarly()) {
      auto* op = ops.top();
      ops.pop();

      effects.clear();

      if (auto memoryInterface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
        memoryInterface.getEffectsOnValue(alias, effects);

        read |= llvm::any_of(effects, [](const auto& effect) {
          return mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect());
        });

        write |= llvm::any_of(effects, [](const auto& effect) {
          return mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect());
        });
      } else if (auto viewInterface = mlir::dyn_cast<mlir::ViewLikeOpInterface>(op)) {
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

static StartOp getStartOperation(ModelOp modelOp, unsigned int varNumber)
{
  mlir::Value variable = modelOp.getBodyRegion().getArgument(varNumber);

  for (auto op : modelOp.getBodyRegion().getOps<StartOp>()) {
    if (op.getVariable() == variable) {
      return op;
    }
  }

  return nullptr;
}

static void createInitializingEquation(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    mlir::Value variable,
    const IndexSet& indices,
    std::function<mlir::Value(mlir::OpBuilder&, mlir::Location)> valueCallback)
{
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

    for (unsigned int i = 0; i <variable.getType().cast<ArrayType>().getRank(); ++i) {
      currentIndices.push_back(inductionVariables[i]);
    }

    mlir::Value scalarVariable = builder.create<LoadOp>(loc, variable, currentIndices);
    mlir::Value lhs = builder.create<EquationSideOp>(loc, scalarVariable);
    mlir::Value rhs = builder.create<EquationSideOp>(loc, value);
    builder.create<EquationSidesOp>(loc, lhs, rhs);
  }
}

static unsigned int addDerivativeToRegions(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    llvm::ArrayRef<mlir::Region*> regions,
    ArrayType derType,
    const IndexSet& derivedIndices)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  assert(!regions.empty());

  assert(llvm::all_of(regions, [&](const auto& region) {
    return region->getNumArguments() == regions[0]->getNumArguments();
  }));

  std::vector<Range> dimensions;

  if (derType.isScalar()) {
    dimensions.emplace_back(0, 1);
  } else {
    for (const auto& dimension : derType.getShape()) {
      dimensions.emplace_back(0, dimension);
    }
  }

  IndexSet allIndices(MultidimensionalRange(std::move(dimensions)));
  auto nonDerivedIndices = allIndices - derivedIndices;

  unsigned int newArgNumber = regions[0]->getNumArguments();

  for (auto& region : regions) {
    assert(region->hasOneBlock());

    mlir::Value derivative = region->addArgument(derType, loc);

    // We need to create additional equations in case of some non-derived indices.
    // If this is not done, then the matching process would fail by detecting an
    // underdetermined problem. An alternative would be to split each variable
    // according to the algebraic / differential nature of its indices, but that
    // is way too complicated with respect to the performance gain.

    createInitializingEquation(builder, loc, derivative, nonDerivedIndices, [](mlir::OpBuilder& builder, mlir::Location loc) {
      mlir::Value zero = builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), 0));
      return zero;
    });
  }

  return newArgNumber;
}

static mlir::LogicalResult createDerivatives(
    mlir::OpBuilder& builder,
    ModelOp modelOp,
    DerivativesMap& derivativesMap,
    const std::map<unsigned int, IndexSet>& derivedIndices)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  // The list of the new variables
  std::vector<mlir::Value> variables;
  auto terminator = mlir::cast<YieldOp>(modelOp.getVarsRegion().back().getTerminator());

  for (auto variable : terminator.getValues()) {
    variables.push_back(variable);
  }

  // Create the new variables for the derivatives
  llvm::SmallVector<unsigned int> derivedVariablesOrdered;

  for (const auto& variable : derivedIndices) {
    derivedVariablesOrdered.push_back(variable.first);
  }

  llvm::sort(derivedVariablesOrdered);

  for (const auto& argNumber : derivedVariablesOrdered) {
    auto variable = terminator.getValues()[argNumber];
    auto memberCreateOp = variable.getDefiningOp<MemberCreateOp>();
    auto variableMemberType = memberCreateOp.getMemberType();

    auto derType = ArrayType::get(variableMemberType.getShape(), RealType::get(builder.getContext()));
    assert(derType.hasStaticShape());

    auto derivedVariableIndices = derivedIndices.find(argNumber);
    assert(derivedVariableIndices != derivedIndices.end());

    auto derArgNumber = addDerivativeToRegions(builder, modelOp.getLoc(), &modelOp.getBodyRegion(), derType, derivedVariableIndices->second);
    derivativesMap.setDerivative(argNumber, derArgNumber);
    derivativesMap.setDerivedIndices(argNumber, derivedVariableIndices->second);

    // Create the variable and initialize it at zero
    builder.setInsertionPoint(terminator);

    auto derivativeName = getNextFullDerVariableName(memberCreateOp.getSymName(), 1);

    auto derMemberOp = builder.create<MemberCreateOp>(
        memberCreateOp.getLoc(), derivativeName, MemberType::wrap(derType), llvm::None);

    variables.push_back(derMemberOp);

    // Create the start value
    builder.setInsertionPointToStart(modelOp.bodyBlock());

    auto startOp = builder.create<StartOp>(
        derMemberOp.getLoc(),
        modelOp.getBodyRegion().getArgument(derArgNumber),
        builder.getBoolAttr(false),
        builder.getBoolAttr(!derType.isScalar()));

    assert(startOp.getBodyRegion().empty());
    mlir::Block* bodyBlock = builder.createBlock(&startOp.getBodyRegion());
    builder.setInsertionPointToStart(bodyBlock);

    mlir::Value zero = builder.create<ConstantOp>(
        derMemberOp.getLoc(), RealAttr::get(builder.getContext(), 0));

    builder.create<YieldOp>(derMemberOp.getLoc(), zero);
  }

  // Update the terminator with the new derivatives
  builder.setInsertionPointToEnd(&modelOp.getVarsRegion().back());
  builder.create<YieldOp>(terminator.getLoc(), variables);
  terminator.erase();

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
    mlir::Value operand = derOp.getOperand();

    while (!operand.isa<mlir::BlockArgument>()) {
      mlir::Operation* definingOp = operand.getDefiningOp();
      assert(mlir::isa<LoadOp>(definingOp) || mlir::isa<SubscriptionOp>(definingOp));

      if (auto loadOp = mlir::dyn_cast<LoadOp>(definingOp)) {
        appendIndexesFn(subscriptions, loadOp.getIndices());
        operand = loadOp.getArray();
      } else {
        auto subscriptionOp = mlir::cast<SubscriptionOp>(definingOp);
        appendIndexesFn(subscriptions, subscriptionOp.getIndices());
        operand = subscriptionOp.getSource();
      }
    }

    auto variableArgNumber = operand.cast<mlir::BlockArgument>().getArgNumber();
    auto derivativeArgNumber = derivativesMap.getDerivative(variableArgNumber);
    mlir::Value derivative = modelOp.getBodyRegion().getArgument(derivativeArgNumber);

    std::vector<mlir::Value> reverted(subscriptions.rbegin(), subscriptions.rend());

    if (!subscriptions.empty()) {
      derivative = builder.create<SubscriptionOp>(derivative.getLoc(), derivative, reverted);
    }

    if (auto arrayType = derivative.getType().cast<ArrayType>(); arrayType.isScalar()) {
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

  modelOp.walk([&](AlgorithmOp algorithmOp) {
    std::vector<DerOp> derOps;

    algorithmOp.walk([&](DerOp derOp) {
      derOps.push_back(derOp);
    });

    for (auto& derOp : derOps) {
      replaceDerOp(derOp);
      derOp.erase();
    }
  });

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
        std::map<unsigned int, IndexSet>& derivedIndices,
        const Equations<Equation>& equations);

      void collectDerivedVariablesIndices(
          std::map<unsigned int, IndexSet>& derivedIndices,
          llvm::ArrayRef<AlgorithmOp> algorithmOps);

      // Convert the binding equations into equations or StartOps.
      mlir::LogicalResult convertBindingEquations();

      /// Add a StartOp for each variable not having one.
      mlir::LogicalResult addMissingStartOps(mlir::OpBuilder& builder);

      mlir::LogicalResult convertAlgorithmsIntoEquations(
          mlir::OpBuilder& builder);

      /// For each EquationOp, create an InitialEquationOp with the same body.
      mlir::LogicalResult cloneEquationsAsInitialEquations(
          mlir::OpBuilder& builder);

      /// Create the initial equations for the variables with a fixed
      /// initialization value.
      mlir::LogicalResult convertFixedStartOpsToInitialEquations(
          mlir::OpBuilder& builder);

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
  // Convert the binding equations.
  if (mlir::failed(convertBindingEquations())) {
    return mlir::failure();
  }

  mlir::OpBuilder builder(modelOp);

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
  std::map<unsigned int, IndexSet> derivedIndices;

  // Add a 'start' value of zero for the variables for which an explicit
  // 'start' value has not been provided.

  if (mlir::failed(addMissingStartOps(builder))) {
    return mlir::failure();
  }

  // Create the initial equations given by the start values having also
  // the fixed attribute set to 'true'.
  if (mlir::failed(convertFixedStartOpsToInitialEquations(builder))) {
    return mlir::failure();
  }

  // Determine which scalar variables do appear as argument to the derivative
  // operation.
  collectDerivedVariablesIndices(
      derivedIndices, initialConditionsModel.getEquations());

  collectDerivedVariablesIndices(derivedIndices, mainModel.getEquations());

  // Discover the derivatives inside the algorithms.
  llvm::SmallVector<AlgorithmOp> algorithmOps;

  modelOp.walk([&](AlgorithmOp algorithmOp) {
    algorithmOps.push_back(algorithmOp);
  });

  collectDerivedVariablesIndices(derivedIndices, algorithmOps);

  // Create the variables for the derivatives, together with the initial
  // equations needed to initialize them to zero.
  DerivativesMap derivativesMap;

  if (mlir::failed(readDerivativesMap(modelOp, derivativesMap))) {
    return mlir::failure();
  }

  if (mlir::failed(createDerivatives(
          builder, modelOp, derivativesMap, derivedIndices))) {
    return mlir::failure();
  }

  // The derivatives mapping is now complete, thus we can set the derivatives
  // map inside the models.
  initialConditionsModel.setDerivativesMap(derivativesMap);
  mainModel.setDerivativesMap(derivativesMap);

  // Remove the derivative operations.
  if (mlir::failed(removeDerOps(builder, modelOp, derivativesMap))) {
    return mlir::failure();
  }

  // Convert the algorithms into equations.
  if (mlir::failed(convertAlgorithmsIntoEquations(builder))) {
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

  writeDerivativesMap(builder, modelOp, derivativesMap, irOptions);

  return mlir::success();
}

static void collectDerivedVariablesIndices(
    std::map<unsigned int, IndexSet>& derivedIndices,
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
    auto argNumber = access.getVariable()->getValue().cast<mlir::BlockArgument>().getArgNumber();

    std::lock_guard<std::mutex> lock(mutex);
    derivedIndices[argNumber] += indices;
  });
}

void ModelLegalization::collectDerivedVariablesIndices(
    std::map<unsigned int, IndexSet>& derivedIndices,
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
    std::map<unsigned int, IndexSet>& derivedIndices,
    AlgorithmOp algorithmOp,
    std::mutex& mutex)
{
  algorithmOp.walk([&](DerOp derOp) {
    mlir::Value value = derOp.getOperand();

    while (!value.isa<mlir::BlockArgument>()) {
      mlir::Operation* definingOp = value.getDefiningOp();

      if (auto loadOp = mlir::dyn_cast<LoadOp>(definingOp)) {
        value = loadOp.getArray();
      } else if (auto subscriptionOp = mlir::dyn_cast<SubscriptionOp>(definingOp)) {
        value = subscriptionOp.getSource();
      } else {
        break;
      }
    }

    assert(value.isa<mlir::BlockArgument>());
    IndexSet indices;

    if (auto arrayType = value.getType().dyn_cast<ArrayType>(); arrayType && !arrayType.isScalar()) {
      assert(arrayType.hasStaticShape());
      std::vector<Range> ranges;

      for (const auto& dimension : arrayType.getShape()) {
        ranges.emplace_back(0, dimension);
      }

      indices += MultidimensionalRange(ranges);
    } else {
      indices += Point(0);
    }

    auto varNumber = value.cast<mlir::BlockArgument>().getArgNumber();

    std::lock_guard<std::mutex> lock(mutex);
    derivedIndices[varNumber] += indices;
  });
}

void ModelLegalization::collectDerivedVariablesIndices(
    std::map<unsigned int, IndexSet>& derivedIndices,
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
  struct BindingEquationOpToEquationOpPattern
      : public mlir::OpRewritePattern<BindingEquationOp>
  {
    using mlir::OpRewritePattern<BindingEquationOp>::OpRewritePattern;

    mlir::LogicalResult match(BindingEquationOp op) const override
    {
      auto modelOp = op->getParentOfType<ModelOp>();
      auto variables = modelOp.getVariableDeclarationOps();

      unsigned int argNumber =
          op.getVariable().cast<mlir::BlockArgument>().getArgNumber();

      return mlir::LogicalResult::success(!variables[argNumber].isReadOnly());
    }

    void rewrite(
        BindingEquationOp op,
        mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location loc = op->getLoc();

      auto yieldOp = mlir::cast<YieldOp>(
          op.getBodyRegion().back().getTerminator());

      assert(yieldOp.getValues().size() == 1);
      mlir::Value expression = yieldOp.getValues()[0];

      auto variableType = op.getVariable().getType();
      auto expressionType = expression.getType();

      std::vector<mlir::Value> inductionVariables;

      if (auto variableArrayType = variableType.dyn_cast<ArrayType>()) {
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

      mlir::BlockAndValueMapping mapping;

      rewriter.cloneRegionBefore(
          op.getBodyRegion(),
          equationOp.getBodyRegion(),
          equationOp.getBodyRegion().end(),
          mapping);

      rewriter.eraseOp(equationOp.getBodyRegion().back().getTerminator());
      rewriter.setInsertionPointToStart(&equationOp.getBodyRegion().back());

      mlir::Value lhsValue = rewriter.create<LoadOp>(
          loc, op.getVariable(), inductionVariables);

      mlir::Value rhsValue = mapping.lookup(expression);
      rewriter.setInsertionPointAfterValue(rhsValue);

      if (!inductionVariables.empty()) {
        rhsValue = rewriter.create<LoadOp>(loc, rhsValue, inductionVariables);
      }

      rewriter.setInsertionPointToEnd(&equationOp.getBodyRegion().back());
      mlir::Value lhsTuple = rewriter.create<EquationSideOp>(loc, lhsValue);
      mlir::Value rhsTuple = rewriter.create<EquationSideOp>(loc, rhsValue);
      rewriter.create<EquationSidesOp>(loc, lhsTuple, rhsTuple);

      rewriter.eraseOp(op);
    }
  };

  struct BindingEquationOpToStartOpPattern
      : public mlir::OpRewritePattern<BindingEquationOp>
  {
    using mlir::OpRewritePattern<BindingEquationOp>::OpRewritePattern;

    mlir::LogicalResult match(BindingEquationOp op) const override
    {
      auto modelOp = op->getParentOfType<ModelOp>();
      auto variables = modelOp.getVariableDeclarationOps();

      unsigned int argNumber =
          op.getVariable().cast<mlir::BlockArgument>().getArgNumber();

      return mlir::LogicalResult::success(variables[argNumber].isReadOnly());
    }

    void rewrite(
        BindingEquationOp op,
        mlir::PatternRewriter& rewriter) const override
    {
      auto startOp = rewriter.replaceOpWithNewOp<StartOp>(
          op, op.getVariable(), true, false);

      assert(startOp.getBodyRegion().empty());

      mlir::BlockAndValueMapping mapping;
      mapping.map(op.getVariable(), startOp.getVariable());

      rewriter.cloneRegionBefore(
          op.getBodyRegion(),
          startOp.getBodyRegion(),
          startOp.getBodyRegion().end(),
          mapping);
    }
  };
}

mlir::LogicalResult ModelLegalization::convertBindingEquations()
{
  mlir::ConversionTarget target(*modelOp.getContext());
  target.addIllegalOp<BindingEquationOp>();

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  mlir::RewritePatternSet patterns(modelOp.getContext());
  patterns.insert<
      BindingEquationOpToEquationOpPattern,
      BindingEquationOpToStartOpPattern>(modelOp.getContext());

  return applyPartialConversion(modelOp, target, std::move(patterns));
}

mlir::LogicalResult ModelLegalization::addMissingStartOps(mlir::OpBuilder& builder)
{
  std::vector<bool> startOpExistence(modelOp.getBodyRegion().getNumArguments(), false);

  modelOp.getBodyRegion().walk([&](StartOp startOp) {
    auto argNumber = startOp.getVariable().cast<mlir::BlockArgument>().getArgNumber();
    startOpExistence[argNumber] = true;
  });

  for (const auto& existence : llvm::enumerate(startOpExistence)) {
    if (!existence.value()) {
      builder.setInsertionPointToEnd(modelOp.bodyBlock());
      auto variable = modelOp.getBodyRegion().getArgument(existence.index());
      auto arrayType = variable.getType().cast<ArrayType>();
      bool each = false;

      if (!arrayType.isScalar()) {
        each = true;
      }

      auto startOp = builder.create<StartOp>(
          variable.getLoc(),
          variable,
          builder.getBoolAttr(false),
          builder.getBoolAttr(each));

      assert(startOp.getBodyRegion().empty());
      mlir::Block* bodyBlock = builder.createBlock(&startOp.getBodyRegion());
      builder.setInsertionPointToStart(bodyBlock);

      mlir::Value zero = builder.create<ConstantOp>(
          variable.getLoc(), getZeroAttr(arrayType.getElementType()));

      builder.create<YieldOp>(variable.getLoc(), zero);
    }
  }

  return mlir::success();
}

mlir::LogicalResult ModelLegalization::convertAlgorithmsIntoEquations(mlir::OpBuilder& builder)
{
  auto module = modelOp->getParentOfType<mlir::ModuleOp>();

  // Collect all the algorithms
  std::vector<AlgorithmInterface> algorithms;

  modelOp.getBodyRegion().walk([&](AlgorithmInterface op) {
    algorithms.push_back(op);
  });

  // Convert them one by one
  size_t algorithmsCount = 0;

  for (auto& algorithm : algorithms) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());

    auto loc = algorithm.getLoc();

    auto functionName = getUniqueSymbolName(module, [&]() -> std::string {
      return modelOp.getSymName().str() + "_algorithm_" + std::to_string(algorithmsCount++);
    });

    // Create the function.
    // At first, assume that all the variables are read and written.
    // We will then prune both the lists according to which variables
    // are effectively accessed or not.

    std::vector<mlir::Type> variableTypes;

    for (const auto& type : modelOp.getBodyRegion().getArgumentTypes()) {
      auto arrayType = type.cast<ArrayType>();

      if (arrayType.isScalar()) {
        variableTypes.push_back(arrayType.getElementType());
      } else {
        variableTypes.push_back(arrayType);
      }
    }

    auto functionType = builder.getFunctionType(variableTypes, variableTypes);
    auto functionOp = builder.create<FunctionOp>(loc, functionName, functionType);

    assert(functionOp.getBody().empty());
    mlir::Block* functionBlock = builder.createBlock(&functionOp.getBody());
    builder.setInsertionPointToStart(functionBlock);

    mlir::BlockAndValueMapping mapping;

    // Temporarily create the arrays as in the original model region, so that
    // we can seamlessly clone the operations of the algorithm.
    std::vector<mlir::Value> arrays;

    for (const auto& arg : modelOp.getBodyRegion().getArguments()) {
      mlir::Value array = builder.create<AllocOp>(loc, arg.getType(), llvm::None);
      arrays.push_back(array);
      mapping.map(arg, array);
    }

    // Clone the operations of the algorithm into the function's body
    for (auto& op : algorithm.getBodyRegion().getOps()) {
      builder.clone(op, mapping);
    }

    std::set<size_t> removedInputVars;
    std::set<size_t> removedOutputVars;

    for (auto& array : llvm::enumerate(arrays)) {
      bool isRead = false;
      bool isWritten = false;

      std::tie(isRead, isWritten) = determineReadWrite(array.value());

      if (!isRead && !isWritten) {
        removedInputVars.insert(array.index());
        removedOutputVars.insert(array.index());

      } else if (isWritten) {
        // If a variable is written, then it must not be taken as input.
        // Instead, it must be initialized with its 'start' value.
        removedInputVars.insert(array.index());

        builder.setInsertionPointAfterValue(array.value());
        auto startOp = getStartOperation(modelOp, array.index());
        assert(startOp != nullptr);

        for (auto& op : startOp.getBodyRegion().getOps()) {
          if (auto yieldOp = mlir::dyn_cast<YieldOp>(op)) {
            mlir::Value valueToBeStored = mapping.lookup(yieldOp.getValues()[0]);
            mlir::Value destination = mapping.lookup(startOp.getVariable());

            if (startOp.getEach()) {
              builder.create<ArrayFillOp>(startOp.getLoc(), destination, valueToBeStored);
            } else {
              builder.create<StoreOp>(startOp.getLoc(), valueToBeStored, destination, llvm::None);
            }
          } else {
            builder.clone(op, mapping);
          }
        }

      } else if (isRead) {
        // If a variable is just read, then it must be kept only as input.
        removedOutputVars.insert(array.index());
      }
    }

    auto arrayReplacer = [&](mlir::Value array, mlir::Value member) {
      auto arrayType = array.getType().cast<ArrayType>();
      bool isScalar = arrayType.isScalar();

      if (isScalar) {
        for (auto* user : llvm::make_early_inc_range(array.getUsers())) {
          assert(mlir::isa<LoadOp>(user) || mlir::isa<StoreOp>(user));
          builder.setInsertionPoint(user);

          if (auto loadOp = mlir::dyn_cast<LoadOp>(user)) {
            mlir::Value replacement = builder.create<MemberLoadOp>(loadOp.getLoc(), member);
            loadOp.replaceAllUsesWith(replacement);
            loadOp.erase();

          } else if (auto storeOp = mlir::dyn_cast<StoreOp>(user)) {
            builder.create<MemberStoreOp>(storeOp.getLoc(), member, storeOp.getValue());
            storeOp.erase();
          }
        }
      } else {
        mlir::Value replacement = builder.create<MemberLoadOp>(loc, member);
        array.replaceAllUsesWith(replacement);
      }
    };

    for (auto& array : llvm::enumerate(arrays)) {
      bool removedFromInput = removedInputVars.find(array.index()) != removedInputVars.end();
      bool removedFromOutput = removedOutputVars.find(array.index()) != removedOutputVars.end();

      builder.setInsertionPointToStart(functionBlock);

      if (removedFromOutput && !removedFromInput) {
        // Input variable
        auto memberType = MemberType::wrap(
            array.value().getType(),
            VariabilityProperty::none,
            IOProperty::input);

        mlir::Value member = builder.create<MemberCreateOp>(
            loc, "arg_" + std::to_string(array.index()), memberType, llvm::None);

        arrayReplacer(array.value(), member);

      } else if (removedFromInput && !removedFromOutput) {
        // Output variable
        auto memberType = MemberType::wrap(
            array.value().getType(),
            VariabilityProperty::none,
            IOProperty::output);

        mlir::Value member = builder.create<MemberCreateOp>(
            loc, "result_" + std::to_string(array.index()), memberType, llvm::None);

        arrayReplacer(array.value(), member);
      }

      array.value().getDefiningOp()->erase();
    }

    // Update the function signature according to the removed arguments and results
    std::vector<mlir::Type> newFunctionArgs;
    std::vector<mlir::Type> newFunctionResults;

    for (const auto& arg : llvm::enumerate(functionType.getInputs())) {
      if (removedInputVars.find(arg.index()) == removedInputVars.end()) {
        newFunctionArgs.push_back(arg.value());
      }
    }

    for (const auto& result : llvm::enumerate(functionType.getResults())) {
      if (removedOutputVars.find(result.index()) == removedOutputVars.end()) {
        newFunctionResults.push_back(result.value());
      }
    }

    functionOp.setFunctionTypeAttr(
        mlir::TypeAttr::get(builder.getFunctionType(newFunctionArgs, newFunctionResults)));

    // Create the equation calling the function
    builder.setInsertionPointToEnd(modelOp.bodyBlock());

    if (mlir::isa<AlgorithmOp>(algorithm)) {
      auto dummyEquationOp = builder.create<EquationOp>(loc);
      assert(dummyEquationOp.getBodyRegion().empty());
      mlir::Block* dummyEquationBody = builder.createBlock(&dummyEquationOp.getBodyRegion());
      builder.setInsertionPointToStart(dummyEquationBody);
    } else {
      assert(mlir::isa<InitialAlgorithmOp>(algorithm));
      auto dummyEquationOp = builder.create<InitialEquationOp>(loc);
      assert(dummyEquationOp.getBodyRegion().empty());
      mlir::Block* dummyEquationBody = builder.createBlock(&dummyEquationOp.getBodyRegion());
      builder.setInsertionPointToStart(dummyEquationBody);
    }

    std::vector<mlir::Value> callArgs;
    std::vector<mlir::Value> callResults;

    auto unboxFn = [](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value) -> mlir::Value {
      auto arrayType = value.getType().cast<ArrayType>();

      if (arrayType.isScalar()) {
        return builder.create<LoadOp>(loc, value, llvm::None);
      } else {
        return value;
      }
    };

    for (const auto& var : llvm::enumerate(modelOp.getBodyRegion().getArguments())) {
      if (removedInputVars.find(var.index()) == removedInputVars.end()) {
        callArgs.push_back(unboxFn(builder, loc, var.value()));
      }
    }

    for (const auto& var : llvm::enumerate(modelOp.getBodyRegion().getArguments())) {
      if (removedOutputVars.find(var.index()) == removedOutputVars.end()) {
        callResults.push_back(unboxFn(builder, loc, var.value()));
      }
    }

    auto callOp = builder.create<CallOp>(
        loc, functionName, mlir::ValueRange(callResults).getTypes(), callArgs);

    mlir::Value lhs = builder.create<EquationSideOp>(loc, callResults);
    mlir::Value rhs = builder.create<EquationSideOp>(loc, callOp.getResults());
    builder.create<EquationSidesOp>(loc, lhs, rhs);

    // The algorithm has been converted, so we can now remove it
    algorithm->erase();
  }

  return mlir::success();
}

mlir::LogicalResult ModelLegalization::cloneEquationsAsInitialEquations(mlir::OpBuilder& builder)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  // Collect the equations
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
    auto initialEquationOp = builder.create<InitialEquationOp>(equationOp.getLoc());
    initialEquationOp->setAttrs(equationOp->getAttrDictionary());

    assert(initialEquationOp.getBodyRegion().empty());
    mlir::Block* bodyBlock = builder.createBlock(&initialEquationOp.getBodyRegion());
    builder.setInsertionPointToStart(bodyBlock);

    for (auto& op : equationOp.bodyBlock()->getOperations()) {
      builder.clone(op, mapping);
    }
  }

  return mlir::success();
}

mlir::LogicalResult ModelLegalization::convertFixedStartOpsToInitialEquations(
    mlir::OpBuilder& builder)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  // Collect the start operations having the 'fixed' attribute set to true.
  std::vector<StartOp> startOps;

  modelOp.getBodyRegion().walk([&](StartOp startOp) {
    if (startOp.getFixed()) {
      startOps.push_back(startOp);
    }
  });

  for (auto& startOp : startOps) {
    builder.setInsertionPointToEnd(modelOp.bodyBlock());

    mlir::Location loc = startOp.getLoc();
    auto memberArrayType = startOp.getVariable().getType().cast<ArrayType>();

    unsigned int expressionRank = 0;

    auto yieldOp =  mlir::cast<YieldOp>(
        startOp.getBodyRegion().back().getTerminator());

    mlir::Value expressionValue = yieldOp.getValues()[0];

    if (auto expressionArrayType =
            expressionValue.getType().dyn_cast<ArrayType>()) {
      expressionRank = expressionArrayType.getRank();
    }

    auto variableRank = memberArrayType.getRank();
    assert(expressionRank == 0 || expressionRank == variableRank);

    std::vector<mlir::Value> inductionVariables;

    for (unsigned int i = 0; i < variableRank - expressionRank; ++i) {
      auto forEquationOp = builder.create<ForEquationOp>(
          loc, 0, memberArrayType.getShape()[i] - 1, 1);

      inductionVariables.push_back(forEquationOp.induction());
      builder.setInsertionPointToStart(forEquationOp.bodyBlock());
    }

    auto equationOp = builder.create<InitialEquationOp>(loc);
    assert(equationOp.getBodyRegion().empty());

    mlir::Block* equationBodyBlock = builder.createBlock(
        &equationOp.getBodyRegion());

    builder.setInsertionPointToStart(equationBodyBlock);

    // Clone the operations.
    mlir::BlockAndValueMapping mapping;

    for (auto& op : startOp.getBody()->getOperations()) {
      if (!mlir::isa<YieldOp>(op)) {
        builder.clone(op, mapping);
      }
    }

    // Left-hand side.
    mlir::Value lhsValue = startOp.getVariable();
    auto lhsArrayType = lhsValue.getType().dyn_cast<ArrayType>();

    if (inductionVariables.size() ==
        static_cast<size_t>(lhsArrayType.getRank())) {
      lhsValue = builder.create<LoadOp>(loc, lhsValue, inductionVariables);
    } else if (!inductionVariables.empty()) {
      lhsValue = builder.create<SubscriptionOp>(
          loc, lhsValue, inductionVariables);
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
  class ModelLegalizationPass : public mlir::modelica::impl::ModelLegalizationPassBase<ModelLegalizationPass>
  {
    public:
      using ModelLegalizationPassBase::ModelLegalizationPassBase;

      void runOnOperation() override
      {
        mlir::OpBuilder builder(getOperation());
        llvm::SmallVector<ModelOp, 1> modelOps;

        getOperation()->walk([&](ModelOp modelOp) {
          if (modelOp.getSymName() == modelName) {
            modelOps.push_back(modelOp);
          }
        });

        for (const auto& modelOp : modelOps) {
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

  std::unique_ptr<mlir::Pass> createModelLegalizationPass(const ModelLegalizationPassOptions& options)
  {
    return std::make_unique<ModelLegalizationPass>(options);
  }
}
