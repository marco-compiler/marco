#include "marco/Codegen/Transforms/ModelSolving/ModelSolving.h"
#include "marco/Codegen/Transforms/ModelSolving/Cycles.h"
#include "marco/Codegen/Transforms/ModelSolving/Equation.h"
#include "marco/Codegen/Transforms/ModelSolving/EquationImpl.h"
#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include "marco/Codegen/Transforms/ModelSolving/ModelConverter.h"
#include "marco/Codegen/Transforms/ModelSolving/Scheduling.h"
#include "marco/Codegen/Transforms/ModelSolving/TypeConverter.h"
#include "marco/Codegen/Transforms/ModelSolving/VariablesMap.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation/Common.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include <cassert>
#include <map>
#include <memory>
#include <queue>

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace
{
  struct EquationOpMultipleValuesPattern : public mlir::OpRewritePattern<EquationOp>
  {
    using mlir::OpRewritePattern<EquationOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(EquationOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto terminator = mlir::cast<EquationSidesOp>(op.bodyBlock()->getTerminator());

      if (terminator.lhsValues().size() != terminator.rhsValues().size()) {
        return rewriter.notifyMatchFailure(op, "Different amount of values in left-hand and right-hand sides of the equation");
      }

      auto amountOfValues = terminator.lhsValues().size();

      for (size_t i = 0; i < amountOfValues; ++i) {
        rewriter.setInsertionPointAfter(op);

        auto clone = rewriter.create<EquationOp>(loc);
        assert(clone.bodyRegion().empty());
        mlir::Block* cloneBodyBlock = rewriter.createBlock(&clone.bodyRegion());
        rewriter.setInsertionPointToStart(cloneBodyBlock);

        mlir::BlockAndValueMapping mapping;

        for (auto& originalOp : op.bodyBlock()->getOperations()) {
          if (mlir::isa<EquationSideOp>(originalOp)) {
            continue;
          }

          if (mlir::isa<EquationSidesOp>(originalOp)) {
            auto lhsOp = mlir::cast<EquationSideOp>(terminator.lhs().getDefiningOp());
            auto rhsOp = mlir::cast<EquationSideOp>(terminator.rhs().getDefiningOp());

            auto newLhsOp = rewriter.create<EquationSideOp>(lhsOp.getLoc(), mapping.lookup(terminator.lhsValues()[i]));
            auto newRhsOp = rewriter.create<EquationSideOp>(rhsOp.getLoc(), mapping.lookup(terminator.rhsValues()[i]));

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
}

static void collectDerivedVariablesIndices(
    std::map<unsigned int, IndexSet>& derivedIndices,
    const Equations<Equation>& equations)
{
  for (const auto& equation : equations) {
    auto accesses = equation->getAccesses();

    equation->getOperation().walk([&](DerOp derOp) {
      auto it = llvm::find_if(accesses, [&](const auto& access) {
        auto value = equation->getValueAtPath(access.getPath());
        return value == derOp.operand();
      });

      assert(it != accesses.end());
      const auto& access = *it;
      auto indices = access.getAccessFunction().map(equation->getIterationRanges());
      auto argNumber = access.getVariable()->getValue().cast<mlir::BlockArgument>().getArgNumber();
      derivedIndices[argNumber] += indices;
    });
  }
}

static void createInitializingEquation(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    mlir::Value variable,
    const IndexSet& indices,
    std::function<mlir::Value(mlir::OpBuilder&, mlir::Location)> valueCallback)
{
  for (const auto& range : indices) {
    std::vector<mlir::Value> inductionVariables;

    for (unsigned int i = 0; i < range.rank(); ++i) {
      auto forOp = builder.create<ForEquationOp>(loc, range[i].getBegin(), range[i].getEnd() - 1);
      inductionVariables.push_back(forOp.induction());
      builder.setInsertionPointToStart(forOp.bodyBlock());
    }

    auto equationOp = builder.create<EquationOp>(loc);
    assert(equationOp.bodyRegion().empty());
    mlir::Block* bodyBlock = builder.createBlock(&equationOp.bodyRegion());
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

    mlir::Value derivative = region->addArgument(derType);

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

static mlir::LogicalResult createDerivatives(
    mlir::OpBuilder& builder,
    ModelOp modelOp,
    DerivativesMap& derivativesMap,
    const std::map<unsigned int, IndexSet>& derivedIndices)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  // Collect the regions to be modified
  llvm::SmallVector<mlir::Region*, 2> regions;
  regions.push_back(&modelOp.equationsRegion());
  regions.push_back(&modelOp.initialEquationsRegion());

  // Ensure that all regions have the same arguments
  assert(llvm::all_of(regions, [&](const auto& region) {
    return region->getArgumentTypes() == regions[0]->getArgumentTypes();
  }));

  // The list of the new variables
  std::vector<mlir::Value> variables;
  auto terminator = mlir::cast<YieldOp>(modelOp.initRegion().back().getTerminator());

  for (auto variable : terminator.values()) {
    variables.push_back(variable);
  }

  // Create the new variables for the derivatives
  llvm::SmallVector<unsigned int> derivedVariablesOrdered;

  for (const auto& variable : derivedIndices) {
    derivedVariablesOrdered.push_back(variable.first);
  }

  llvm::sort(derivedVariablesOrdered);

  for (const auto& argNumber : derivedVariablesOrdered) {
    auto variable = terminator.values()[argNumber];
    auto memberCreateOp = variable.getDefiningOp<MemberCreateOp>();
    auto variableMemberType = memberCreateOp.getMemberType();

    auto derType = ArrayType::get(builder.getContext(), RealType::get(builder.getContext()), variableMemberType.getShape());
    assert(derType.hasConstantShape());

    auto derivedVariableIndices = derivedIndices.find(argNumber);
    assert(derivedVariableIndices != derivedIndices.end());

    auto derArgNumber = addDerivativeToRegions(builder, modelOp.getLoc(), regions, derType, derivedVariableIndices->second);
    derivativesMap.setDerivative(argNumber, derArgNumber);
    derivativesMap.setDerivedIndices(argNumber, derivedVariableIndices->second);

    // Create the variable and initialize it at zero
    builder.setInsertionPoint(terminator);

    auto derivativeName = getNextFullDerVariableName(memberCreateOp.name(), 1);

    auto derMemberOp = builder.create<MemberCreateOp>(
        memberCreateOp.getLoc(), derivativeName, MemberType::wrap(derType), llvm::None);

    variables.push_back(derMemberOp);

    mlir::Value zero = builder.create<ConstantOp>(derMemberOp.getLoc(), RealAttr::get(builder.getContext(), 0));

    if (derType.isScalar()) {
      builder.create<MemberStoreOp>(derMemberOp.getLoc(), derMemberOp, zero);
    } else {
      mlir::Value derivative = builder.create<MemberLoadOp>(derMemberOp.getLoc(), derMemberOp);
      builder.create<ArrayFillOp>(derMemberOp.getLoc(), derivative, zero);
    }

    /*
    // Create the initial equations needed to initialize the derivative to zero
    builder.setInsertionPointToEnd(modelOp.initialEquationsBlock());
    mlir::Value derivative = modelOp.initialEquationsRegion().getArgument(derArgNumber);

    createInitializingEquation(
        builder, modelOp.getLoc(), derivative, derivativesMap.getDerivedIndices(argNumber),
        [](mlir::OpBuilder& builder, mlir::Location loc) {
          mlir::Value zero = builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), 0));
          return zero;
        });
        */
  }

  builder.setInsertionPointToEnd(&modelOp.initRegion().back());
  builder.create<YieldOp>(terminator.getLoc(), variables);
  terminator.erase();

  return mlir::success();
}

static mlir::LogicalResult removeDerOps(
    mlir::OpBuilder& builder,
    ModelOp modelOp,
    const DerivativesMap& derivativesMap)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  // Collect the regions to be modified
  llvm::SmallVector<mlir::Region*, 2> regions;
  regions.push_back(&modelOp.equationsRegion());
  regions.push_back(&modelOp.initialEquationsRegion());

  auto appendIndexesFn = [](std::vector<mlir::Value>& destination, mlir::ValueRange indices) {
    for (size_t i = 0, e = indices.size(); i < e; ++i) {
      mlir::Value index = indices[e - 1 - i];
      destination.push_back(index);
    }
  };

  for (auto& region : regions) {
    region->walk([&](EquationOp equationOp) {
      std::vector<DerOp> derOps;

      equationOp.walk([&](DerOp derOp) {
        derOps.push_back(derOp);
      });

      for (auto& derOp : derOps) {
        builder.setInsertionPoint(derOp);

        // If the value to be derived belongs to an array, then also the derived
        // value is stored within an array. Thus, we need to store its position.

        std::vector<mlir::Value> subscriptions;
        mlir::Value operand = derOp.operand();

        while (!operand.isa<mlir::BlockArgument>()) {
          mlir::Operation* definingOp = operand.getDefiningOp();
          assert(mlir::isa<LoadOp>(definingOp) || mlir::isa<SubscriptionOp>(definingOp));

          if (auto loadOp = mlir::dyn_cast<LoadOp>(definingOp)) {
            appendIndexesFn(subscriptions, loadOp.indices());
            operand = loadOp.array();
          } else {
            auto subscriptionOp = mlir::cast<SubscriptionOp>(definingOp);
            appendIndexesFn(subscriptions, subscriptionOp.indices());
            operand = subscriptionOp.source();
          }
        }

        auto variableArgNumber = operand.cast<mlir::BlockArgument>().getArgNumber();
        auto derivativeArgNumber = derivativesMap.getDerivative(variableArgNumber);
        mlir::Value derivative = region->getArgument(derivativeArgNumber);

        std::vector<mlir::Value> reverted(subscriptions.rbegin(), subscriptions.rend());

        if (!subscriptions.empty()) {
          derivative = builder.create<SubscriptionOp>(derivative.getLoc(), derivative, reverted);
        }

        if (auto arrayType = derivative.getType().cast<ArrayType>(); arrayType.isScalar()) {
          derivative = builder.create<LoadOp>(derivative.getLoc(), derivative);
        }

        derOp.replaceAllUsesWith(derivative);
        eraseValueInsideEquation(derOp.getResult());
      }
    });
  }

  return mlir::success();
}

namespace
{
  /// Model solving pass.
  /// Its objective is to convert a descriptive (and thus not sequential) model into an
  /// algorithmic one and to create the functions to be called during the simulation.
  class ModelSolvingPass: public mlir::PassWrapper<ModelSolvingPass, mlir::OperationPass<mlir::ModuleOp>>
  {
    public:
      explicit ModelSolvingPass(ModelSolvingOptions options, unsigned int bitWidth)
          : options(std::move(options)),
            bitWidth(std::move(bitWidth))
      {
      }

      void getDependentDialects(mlir::DialectRegistry& registry) const override
      {
        registry.insert<ModelicaDialect>();
        registry.insert<mlir::ida::IDADialect>();
        registry.insert<mlir::scf::SCFDialect>();
        registry.insert<mlir::LLVM::LLVMDialect>();
      }

      void runOnOperation() override
      {
        llvm::SmallVector<ModelOp, 1> models;

        getOperation().walk([&](ModelOp op) {
          models.push_back(op);
        });

        if (models.size() > 1) {
          // There must be at most one ModelOp inside the module
          return signalPassFailure();
        }

        mlir::OpBuilder builder(models[0]);

        // Copy the equations into the initial equations' region, in order to use
        // them when computing the initial values of the variables.

        if (mlir::failed(copyEquationsAmongInitialEquations(builder, models[0]))) {
          return signalPassFailure();
        }

        if (mlir::failed(convertEquationsWithMultipleValues())) {
          return signalPassFailure();
        }

        // Split the loops containing more than one operation within their bodies
        if (mlir::failed(convertToSingleEquationBody(models[0]))) {
          return signalPassFailure();
        }

        // The initial conditions are determined by resolving a separate model, with
        // indeed more equations than the model used during the simulation loop.
        Model<Equation> initialModel(models[0]);
        Model<Equation> model(models[0]);

        initialModel.setVariables(discoverVariables(initialModel.getOperation().initialEquationsRegion()));
        initialModel.setEquations(discoverEquations(initialModel.getOperation().initialEquationsRegion(), initialModel.getVariables()));

        model.setVariables(discoverVariables(model.getOperation().equationsRegion()));
        model.setEquations(discoverEquations(model.getOperation().equationsRegion(), model.getVariables()));

        // Determine which scalar variables do appear as argument to the derivative operation
        std::map<unsigned int, IndexSet> derivedIndices;
        collectDerivedVariablesIndices(derivedIndices, initialModel.getEquations());
        collectDerivedVariablesIndices(derivedIndices, model.getEquations());

        // The variable splitting may have caused a variable list change and equations splitting.
        // For this reason we need to perform again the discovery process.

        initialModel.setVariables(discoverVariables(initialModel.getOperation().initialEquationsRegion()));
        initialModel.setEquations(discoverEquations(initialModel.getOperation().initialEquationsRegion(), initialModel.getVariables()));

        model.setVariables(discoverVariables(model.getOperation().equationsRegion()));
        model.setEquations(discoverEquations(model.getOperation().equationsRegion(), model.getVariables()));

        // Create the variables for the derivatives, together with the initial equations needed to
        // initialize them to zero.
        DerivativesMap derivativesMap;

        if (mlir::failed(createDerivatives(builder, models[0], derivativesMap, derivedIndices))) {
          return signalPassFailure();
        }

        // The derivatives mapping is now complete, thus we can set the derivatives map inside the models
        initialModel.setDerivativesMap(derivativesMap);
        model.setDerivativesMap(derivativesMap);

        // Now that the derivatives have been converted to variables, we need perform a new scan
        // of the variables so that they become available inside the model.
        initialModel.setVariables(discoverVariables(initialModel.getOperation().initialEquationsRegion()));
        model.setVariables(discoverVariables(model.getOperation().equationsRegion()));

        // Additional equations are introduced in case of partially derived arrays. Thus, we need
        // to perform again the discovery of the equations.

        initialModel.setEquations(discoverEquations(initialModel.getOperation().initialEquationsRegion(), initialModel.getVariables()));
        model.setEquations(discoverEquations(model.getOperation().equationsRegion(), model.getVariables()));

        // Remove the derivative operations
        if (mlir::failed(removeDerOps(builder, models[0], derivativesMap))) {
          return signalPassFailure();
        }

        // Solve the initial conditions problem
        Model<ScheduledEquationsBlock> scheduledInitialModel(initialModel.getOperation());

        if (mlir::failed(solveInitialConditionsModel(builder, scheduledInitialModel, initialModel))) {
          initialModel.getOperation().emitError("Can't solve the initialization problem");
          return signalPassFailure();
        }

        // Solve the main model
        Model<ScheduledEquationsBlock> scheduledModel(model.getOperation());

        if (mlir::failed(solveMainModel(builder, scheduledModel, model))) {
          model.getOperation().emitError("Can't solve the main model");
          return signalPassFailure();
        }

        // Create the simulation functions
        mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
        marco::codegen::TypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);
        ModelConverter modelConverter(options, typeConverter);

        if (mlir::failed(modelConverter.convertInitialModel(builder, scheduledInitialModel))) {
          return signalPassFailure();
        }

        if (mlir::failed(modelConverter.convertMainModel(builder, scheduledModel))) {
          return signalPassFailure();
        }

        // Erase the model operation, which has been converted to algorithmic code
        models[0].erase();
      }

    private:
      /// Copy the equations declared into the 'equations' region into the 'initial equations' region.
      mlir::LogicalResult copyEquationsAmongInitialEquations(mlir::OpBuilder& builder, ModelOp modelOp)
      {
        if (!modelOp.hasEquationsBlock()) {
          // There is no equation to be copied
          return mlir::success();
        }

        mlir::OpBuilder::InsertionGuard guard(builder);

        if (!modelOp.hasInitialEquationsBlock()) {
          builder.createBlock(&modelOp.initialEquationsRegion(), {}, modelOp.equationsRegion().getArgumentTypes());
        }

        // Map the variables declared into the equations region to the ones declared into the initial equations' region
        mlir::BlockAndValueMapping mapping;
        auto originalVariables = modelOp.equationsRegion().getArguments();
        auto mappedVariables = modelOp.initialEquationsRegion().getArguments();
        assert(originalVariables.size() == mappedVariables.size());

        for (const auto& [original, mapped] : llvm::zip(originalVariables, mappedVariables)) {
          mapping.map(original, mapped);
        }

        // Clone the equations
        builder.setInsertionPointToEnd(modelOp.initialEquationsBlock());

        for (auto& op : modelOp.equationsBlock()->getOperations()) {
          builder.clone(op, mapping);
        }

        return mlir::success();
      }

      mlir::LogicalResult convertEquationsWithMultipleValues()
      {
        mlir::ConversionTarget target(getContext());

        target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
          return true;
        });

        target.addDynamicallyLegalOp<EquationOp>([](EquationOp op) {
          auto terminator = mlir::cast<EquationSidesOp>(op.bodyBlock()->getTerminator());
          return terminator.lhsValues().size() == 1 && terminator.rhsValues().size() == 1;
        });

        mlir::OwningRewritePatternList patterns(&getContext());
        patterns.insert<EquationOpMultipleValuesPattern>(&getContext());

        return applyPartialConversion(getOperation(), target, std::move(patterns));
      }

      mlir::LogicalResult convertToSingleEquationBody(ModelOp modelOp)
      {
        llvm::SmallVector<EquationOp> equations;

        for (auto op : modelOp.equationsBlock()->getOps<EquationOp>()) {
          equations.push_back(op);
        }

        mlir::OpBuilder builder(modelOp);

        mlir::BlockAndValueMapping mapping;

        for (auto& equationOp : equations) {
          builder.setInsertionPointToEnd(modelOp.equationsBlock());
          std::vector<ForEquationOp> parents;

          ForEquationOp parent = equationOp->getParentOfType<ForEquationOp>();

          while (parent != nullptr) {
            parents.push_back(parent);
            parent = parent->getParentOfType<ForEquationOp>();
          }

          for (size_t i = 0, e = parents.size(); i < e; ++i) {
            auto clonedParent = mlir::cast<ForEquationOp>(builder.clone(*parents[e - i - 1].getOperation(), mapping));
            builder.setInsertionPointToEnd(clonedParent.bodyBlock());
          }

          builder.clone(*equationOp.getOperation(), mapping);
        }

        for (auto& equationOp : equations) {
          ForEquationOp parent = equationOp->getParentOfType<ForEquationOp>();
          equationOp.erase();

          while (parent != nullptr && parent.bodyBlock()->empty()) {
            ForEquationOp newParent = parent->getParentOfType<ForEquationOp>();
            parent.erase();
            parent = newParent;
          }
        }

        return mlir::success();
      }

      mlir::LogicalResult splitEquations(mlir::OpBuilder& builder, Model<MatchedEquation>& model)
      {
        Equations<MatchedEquation> equations;

        for (const auto& equation : model.getEquations()) {
          auto write = equation->getWrite();
          auto iterationRanges = equation->getIterationRanges();
          auto writtenIndices = write.getAccessFunction().map(iterationRanges);

          IndexSet result;

          for (const auto& access : equation->getAccesses()) {
            if (access.getPath() == write.getPath()) {
              continue;
            }

            if (access.getVariable() != write.getVariable()) {
              continue;
            }

            auto accessedIndices = access.getAccessFunction().map(iterationRanges);

            if (!accessedIndices.overlaps(writtenIndices)) {
              continue;
            }

            result += write.getAccessFunction().inverseMap(
                IndexSet(accessedIndices.intersect(writtenIndices)),
                IndexSet(iterationRanges));
          }

          for (const auto& range : result) {
            auto clone = Equation::build(equation->getOperation(), equation->getVariables());

            auto matchedClone = std::make_unique<MatchedEquation>(
                std::move(clone), range, write.getPath());

            equations.add(std::move(matchedClone));
          }

          for (const auto& range : IndexSet(iterationRanges) - result) {
            auto clone = Equation::build(equation->getOperation(), equation->getVariables());

            auto matchedClone = std::make_unique<MatchedEquation>(
                std::move(clone), range, write.getPath());

            equations.add(std::move(matchedClone));
          }
        }

        model.setEquations(equations);
        return mlir::success();
      }

      mlir::LogicalResult solveInitialConditionsModel(
          mlir::OpBuilder& builder,
          Model<ScheduledEquationsBlock>& result,
          const Model<Equation>& model)
      {
        // Matching process
        Model<MatchedEquation> matchedModel(model.getOperation());
        matchedModel.setDerivativesMap(model.getDerivativesMap());

        auto matchableIndicesFn = [&](const Variable& variable) -> IndexSet {
          IndexSet matchableIndices(variable.getIndices());

          if (variable.isConstant()) {
            matchableIndices.clear();
            return matchableIndices;
          }

          return matchableIndices;
        };

        if (auto res = match(matchedModel, model, matchableIndicesFn); mlir::failed(res)) {
          return res;
        }

        if (auto res = splitEquations(builder, matchedModel); mlir::failed(res)) {
          return res;
        }

        // Resolve the algebraic loops
        if (auto res = solveCycles(matchedModel, builder); mlir::failed(res)) {
          if (options.solver != Solver::ida) {
            // Check if the selected solver can deal with cycles. If not, fail.
            return res;
          }
        }

        // Schedule the equations
        if (auto res = schedule(result, matchedModel); mlir::failed(res)) {
          return res;
        }

        result.setDerivativesMap(model.getDerivativesMap());
        return mlir::success();
      }

      mlir::LogicalResult solveMainModel(
          mlir::OpBuilder& builder,
          Model<ScheduledEquationsBlock>& result,
          const Model<Equation>& model)
      {
        // Matching process
        Model<MatchedEquation> matchedModel(model.getOperation());
        matchedModel.setDerivativesMap(model.getDerivativesMap());

        auto matchableIndicesFn = [&](const Variable& variable) -> IndexSet {
          IndexSet matchableIndices(variable.getIndices());

          if (variable.isConstant()) {
            matchableIndices.clear();
            return matchableIndices;
          }

          auto argNumber = variable.getValue().cast<mlir::BlockArgument>().getArgNumber();

          if (auto derivativesMap = matchedModel.getDerivativesMap(); derivativesMap.hasDerivative(argNumber)) {
            matchableIndices -= matchedModel.getDerivativesMap().getDerivedIndices(argNumber);
          }

          return matchableIndices;
        };

        if (auto res = match(matchedModel, model, matchableIndicesFn); mlir::failed(res)) {
          return res;
        }

        if (auto res = splitEquations(builder, matchedModel); mlir::failed(res)) {
          return res;
        }

        // Resolve the algebraic loops
        if (auto res = solveCycles(matchedModel, builder); mlir::failed(res)) {
          if (options.solver != Solver::ida) {
            // Check if the selected solver can deal with cycles. If not, fail.
            return res;
          }
        }

        // Schedule the equations
        if (auto res = schedule(result, matchedModel); mlir::failed(res)) {
          return res;
        }

        result.setDerivativesMap(model.getDerivativesMap());
        return mlir::success();
      }

    private:
      ModelSolvingOptions options;
      unsigned int bitWidth;
  };
}

namespace marco::codegen
{
  std::unique_ptr<mlir::Pass> createModelSolvingPass(ModelSolvingOptions options, unsigned int bitWidth)
  {
    return std::make_unique<ModelSolvingPass>(options, bitWidth);
  }
}
