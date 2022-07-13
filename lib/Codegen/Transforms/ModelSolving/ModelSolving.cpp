#include "marco/Codegen/Transforms/ModelSolving.h"
#include "marco/Codegen/Transforms/ModelSolving/Cycles.h"
#include "marco/Codegen/Transforms/ModelSolving/Equation.h"
#include "marco/Codegen/Transforms/ModelSolving/EquationImpl.h"
#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include "marco/Codegen/Transforms/ModelSolving/ModelConverter.h"
#include "marco/Codegen/Transforms/ModelSolving/Scheduling.h"
#include "marco/Codegen/Transforms/ModelSolving/TypeConverter.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation/Common.h"
#include "marco/Codegen/Utils.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include <cassert>
#include <map>
#include <memory>
#include <queue>

#include "marco/Codegen/Transforms/PassDetail.h"

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

static void collectDerivedVariablesIndices(
    std::map<unsigned int, IndexSet>& derivedIndices,
    const Equations<Equation>& equations)
{
  for (const auto& equation : equations) {
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
      derivedIndices[argNumber] += indices;
    });
  }
}

static void collectDerivedVariablesIndices(
    std::map<unsigned int, IndexSet>& derivedIndices,
    AlgorithmOp algorithmOp)
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
    IndexSet derivedIndices;

    if (auto arrayType = value.getType().dyn_cast<ArrayType>()) {
      assert(arrayType.hasConstantShape());
      std::vector<Range> ranges;

      for (const auto& dimension : arrayType.getShape()) {
        ranges.emplace_back(0, dimension);
      }

      derivedIndices += MultidimensionalRange(ranges);
    } else {
      derivedIndices += Point(0);
    }
  });
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

    auto derType = ArrayType::get(builder.getContext(), RealType::get(builder.getContext()), variableMemberType.getShape());
    assert(derType.hasConstantShape());

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

  modelOp.getBodyRegion().walk([&](EquationInterface equationInt) {
    std::vector<DerOp> derOps;

    equationInt.walk([&](DerOp derOp) {
      derOps.push_back(derOp);
    });

    for (auto& derOp : derOps) {
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
      eraseValueInsideEquation(derOp.getResult());
    }
  });

  return mlir::success();
}

namespace
{
  struct FuncOpTypesPattern : public mlir::OpConversionPattern<mlir::func::FuncOp>
  {
    using mlir::OpConversionPattern<mlir::func::FuncOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(mlir::func::FuncOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Type, 3> resultTypes;
      llvm::SmallVector<mlir::Type, 3> argTypes;

      for (const auto& type : op.getFunctionType().getResults()) {
        resultTypes.push_back(getTypeConverter()->convertType(type));
      }

      for (const auto& type : op.getFunctionType().getInputs()) {
        argTypes.push_back(getTypeConverter()->convertType(type));
      }

      auto functionType = rewriter.getFunctionType(argTypes, resultTypes);
      auto newOp = rewriter.replaceOpWithNewOp<mlir::func::FuncOp>(op, op.getSymName(), functionType);

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      mlir::BlockAndValueMapping mapping;

      // Clone the blocks structure
      for (auto& block : llvm::enumerate(op.getBody())) {
        if (block.index() == 0) {
          mapping.map(&block.value(), entryBlock);
        } else {
          std::vector<mlir::Location> argLocations;

          for (const auto& arg : block.value().getArguments()) {
            argLocations.push_back(arg.getLoc());
          }

          auto signatureConversion = typeConverter->convertBlockSignature(&block.value());

          if (!signatureConversion) {
            return mlir::failure();
          }

          mlir::Block* clonedBlock = rewriter.createBlock(
              &newOp.getBody(),
              newOp.getBody().end(),
              signatureConversion->getConvertedTypes(),
              argLocations);

          mapping.map(&block.value(), clonedBlock);
        }
      }

      for (auto& block : op.getBody().getBlocks()) {
        mlir::Block* clonedBlock = mapping.lookup(&block);
        rewriter.setInsertionPointToStart(clonedBlock);

        // Cast the block arguments
        for (const auto& [original, cloned] : llvm::zip(block.getArguments(), clonedBlock->getArguments())) {
          mlir::Value arg = typeConverter->materializeSourceConversion(
              rewriter, cloned.getLoc(), original.getType(), cloned);

          mapping.map(original, arg);
        }

        // Clone the operations
        for (auto& bodyOp : block.getOperations()) {
          if (auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(bodyOp)) {
            std::vector<mlir::Value> returnValues;

            for (auto returnValue : returnOp.operands()) {
              returnValues.push_back(getTypeConverter()->materializeTargetConversion(
                  rewriter, returnOp.getLoc(),
                  getTypeConverter()->convertType(returnValue.getType()),
                  mapping.lookup(returnValue)));
            }

            rewriter.create<mlir::func::ReturnOp>(returnOp.getLoc(), returnValues);
          } else {
            rewriter.clone(bodyOp, mapping);
          }
        }
      }

      return mlir::success();
    }
  };

  struct CallOpTypesPattern : public mlir::OpConversionPattern<mlir::func::CallOp>
  {
    using mlir::OpConversionPattern<mlir::func::CallOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(mlir::func::CallOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 3> values;

      for (const auto& operand : op.operands()) {
        values.push_back(getTypeConverter()->materializeTargetConversion(
            rewriter, operand.getLoc(), getTypeConverter()->convertType(operand.getType()), operand));
      }

      llvm::SmallVector<mlir::Type, 3> resultTypes;

      for (const auto& type : op.getResults().getTypes()) {
        resultTypes.push_back(getTypeConverter()->convertType(type));
      }

      auto newOp = rewriter.create<mlir::func::CallOp>(op.getLoc(), op.getCallee(), resultTypes, values);

      llvm::SmallVector<mlir::Value, 3> results;

      for (const auto& [oldResult, newResult] : llvm::zip(op.getResults(), newOp.getResults())) {
        if (oldResult.getType() != newResult.getType()) {
          results.push_back(getTypeConverter()->materializeSourceConversion(
              rewriter, newResult.getLoc(), oldResult.getType(), newResult));
        } else {
          results.push_back(newResult);
        }
      }

      rewriter.replaceOp(op, results);
      return mlir::success();
    }
  };
}

namespace
{
  /// Model solving pass.
  /// Its objective is to convert a descriptive (and thus not sequential) model into an
  /// algorithmic one and to create the functions to be called during the simulation.
  class ModelSolvingPass: public ModelSolvingBase<ModelSolvingPass>
  {
    public:
      ModelSolvingPass(ModelSolvingOptions options, unsigned int bitWidth)
          : options(std::move(options)),
            bitWidth(std::move(bitWidth))
      {
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

        // A map that keeps track of which indices of a variable do appear under
        // the derivative operation.
        std::map<unsigned int, IndexSet> derivedIndices;

        // Discover the derivatives inside the algorithms
        models[0].walk([&](AlgorithmOp algorithmOp) {
          collectDerivedVariablesIndices(derivedIndices, algorithmOp);
        });

        // Convert the algorithms into equations
        if (mlir::failed(convertAlgorithmsIntoEquations(builder, models[0]))) {
          return signalPassFailure();
        }

        // Clone the equations as initial equations, in order to use them when
        // computing the initial values of the variables.

        if (mlir::failed(cloneEquationsAsInitialEquations(builder, models[0]))) {
          return signalPassFailure();
        }

        // Create the initial equations given by the start values having also
        // the fixed attribute set to true.

        if (mlir::failed(convertFixedStartOps(builder, models[0]))) {
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

        initialModel.setVariables(discoverVariables(initialModel.getOperation()));
        initialModel.setEquations(discoverInitialEquations(initialModel.getOperation(), initialModel.getVariables()));

        model.setVariables(discoverVariables(model.getOperation()));
        model.setEquations(discoverEquations(model.getOperation(), model.getVariables()));

        // Determine which scalar variables do appear as argument to the derivative operation
        collectDerivedVariablesIndices(derivedIndices, initialModel.getEquations());
        collectDerivedVariablesIndices(derivedIndices, model.getEquations());

        // The variable splitting may have caused a variable list change and equations splitting.
        // For this reason we need to perform again the discovery process.

        initialModel.setVariables(discoverVariables(initialModel.getOperation()));
        initialModel.setEquations(discoverInitialEquations(initialModel.getOperation(), initialModel.getVariables()));

        model.setVariables(discoverVariables(model.getOperation()));
        model.setEquations(discoverEquations(model.getOperation(), model.getVariables()));

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
        initialModel.setVariables(discoverVariables(initialModel.getOperation()));
        model.setVariables(discoverVariables(model.getOperation()));

        // Additional equations are introduced in case of partially derived arrays. Thus, we need
        // to perform again the discovery of the equations.

        initialModel.setEquations(discoverInitialEquations(initialModel.getOperation(), initialModel.getVariables()));
        model.setEquations(discoverEquations(model.getOperation(), model.getVariables()));

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

        if (mlir::failed(modelConverter.createGetModelNameFunction(builder, models[0]))) {
          models[0].emitError("Could not create the '" + ModelConverter::getModelNameFunctionName + "' function");
          return signalPassFailure();
        }

        if (mlir::failed(modelConverter.createInitFunction(builder, models[0]))) {
          models[0].emitError("Could not create the '" + ModelConverter::initFunctionName + "' function");
          return signalPassFailure();
        }

        if (mlir::failed(modelConverter.createDeinitFunction(builder, models[0]))) {
          models[0].emitError("Could not create the '" + ModelConverter::deinitFunctionName + "' function");
          return signalPassFailure();
        }

        if (options.emitMain) {
          if (mlir::failed(modelConverter.createMainFunction(builder, models[0]))) {
            models[0].emitError("Could not create the '" + ModelConverter::mainFunctionName + "' function");
            return signalPassFailure();
          }
        }

        if (mlir::failed(modelConverter.convertInitialModel(builder, scheduledInitialModel))) {
          return signalPassFailure();
        }

        if (mlir::failed(modelConverter.convertMainModel(builder, scheduledModel))) {
          return signalPassFailure();
        }

        // Erase the model operation, which has been converted to algorithmic code
        models[0].erase();

        // Convert the functions having a Modelica type within their signature.
        if (mlir::failed(convertFuncOps())) {
          return signalPassFailure();
        }
      }

    private:
      mlir::LogicalResult convertAlgorithmsIntoEquations(mlir::OpBuilder& builder, ModelOp modelOp)
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

          auto functionType = builder.getFunctionType(
              modelOp.getBodyRegion().getArgumentTypes(),
              modelOp.getBodyRegion().getArgumentTypes());

          auto functionOp = builder.create<FunctionOp>(loc, functionName, functionType);

          assert(functionOp.getBody().empty());
          mlir::Block* functionBlock = builder.createBlock(&functionOp.getBody());
          builder.setInsertionPointToStart(functionBlock);

          // Create the input and output members of the function
          std::vector<mlir::Value> inputMemberCreateOps;
          std::vector<mlir::Value> outputMemberCreateOps;

          std::vector<mlir::Value> inputMemberLoadOps;
          std::vector<mlir::Value> outputMemberLoadOps;

          for (const auto& type : llvm::enumerate(functionType.getInputs())) {
            auto memberType = MemberType::wrap(type.value(), false, IOProperty::input);
            auto memberCreateOp = builder.create<MemberCreateOp>(loc, "arg_" + std::to_string(type.index()), memberType, llvm::None);
            inputMemberCreateOps.push_back(memberCreateOp);
          }

          for (const auto& type : llvm::enumerate(functionType.getResults())) {
            auto memberType = MemberType::wrap(type.value(), false, IOProperty::output);
            auto memberCreateOp = builder.create<MemberCreateOp>(loc, "result_" + std::to_string(type.index()), memberType, llvm::None);
            outputMemberCreateOps.push_back(memberCreateOp);
          }

          mlir::BlockAndValueMapping mapping;

          for (const auto& member : llvm::enumerate(inputMemberCreateOps)) {
            auto memberLoadOp = builder.create<MemberLoadOp>(loc, member.value());
            inputMemberLoadOps.push_back(memberLoadOp);
          }

          for (const auto& member : llvm::enumerate(outputMemberCreateOps)) {
            auto memberLoadOp = builder.create<MemberLoadOp>(loc, member.value());
            outputMemberLoadOps.push_back(memberLoadOp);
            mapping.map(modelOp.getBodyRegion().getArgument(member.index()), memberLoadOp);
          }

          // Clone the operations of the algorithm into the function's body
          for (auto& op : algorithm.bodyBlock()->getOperations()) {
            builder.clone(op, mapping);
          }

          // Remove the output variables that are never written
          std::set<size_t> removedOutputVars;

          for (const auto& outputVar : llvm::enumerate(outputMemberLoadOps)) {
            if (outputVar.value().use_empty()) {
              removedOutputVars.insert(outputVar.index());
              outputVar.value().getDefiningOp()->erase();
              outputMemberCreateOps[outputVar.index()].getDefiningOp()->erase();
            } else {
              builder.setInsertionPointAfterValue(inputMemberLoadOps[outputVar.index()]);
              builder.create<MemberStoreOp>(loc, outputMemberCreateOps[outputVar.index()], inputMemberLoadOps[outputVar.index()]);
            }
          }

          // Remove the input variables that are never read
          std::set<size_t> removedInputVars;

          for (const auto& inputVar : llvm::enumerate(inputMemberLoadOps)) {
            if (inputVar.value().use_empty()) {
              removedInputVars.insert(inputVar.index());
              inputVar.value().getDefiningOp()->erase();
              inputMemberCreateOps[inputVar.index()].getDefiningOp()->erase();
            }
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

          for (const auto& var : llvm::enumerate(modelOp.getBodyRegion().getArguments())) {
            if (removedInputVars.find(var.index()) == removedInputVars.end()) {
              callArgs.push_back(var.value());
            }
          }

          for (const auto& var : llvm::enumerate(modelOp.getBodyRegion().getArguments())) {
            if (removedOutputVars.find(var.index()) == removedOutputVars.end()) {
              callResults.push_back(var.value());
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

      /// For each EquationOp, create an InitialEquationOp with the same body
      mlir::LogicalResult cloneEquationsAsInitialEquations(mlir::OpBuilder& builder, ModelOp modelOp)
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
          assert(initialEquationOp.getBodyRegion().empty());
          mlir::Block* bodyBlock = builder.createBlock(&initialEquationOp.getBodyRegion());
          builder.setInsertionPointToStart(bodyBlock);

          for (auto& op : equationOp.bodyBlock()->getOperations()) {
            builder.clone(op, mapping);
          }
        }

        return mlir::success();
      }

      mlir::LogicalResult convertFixedStartOps(mlir::OpBuilder& builder, ModelOp modelOp)
      {
        mlir::OpBuilder::InsertionGuard guard(builder);

        // Collect the start operations having the 'fixed' attribute set to true
        std::vector<StartOp> startOps;

        modelOp.getBodyRegion().walk([&](StartOp startOp) {
          if (startOp.getFixed()) {
            startOps.push_back(startOp);
          }
        });

        for (auto& startOp : startOps) {
          builder.setInsertionPointToEnd(modelOp.bodyBlock());

          auto loc = startOp.getLoc();
          auto memberArrayType = startOp.getVariable().getType().cast<ArrayType>();

          unsigned int expressionRank = 0;
          auto yieldOp =  mlir::cast<YieldOp>(startOp.getBodyRegion().back().getTerminator());
          mlir::Value expressionValue = yieldOp.getValues()[0];

          if (auto expressionArrayType = expressionValue.getType().dyn_cast<ArrayType>()) {
            expressionRank = expressionArrayType.getRank();
          }

          auto memberRank = memberArrayType.getRank();
          assert(expressionRank == 0 || expressionRank == memberRank);

          std::vector<mlir::Value> inductionVariables;

          for (unsigned int i = 0; i < memberRank - expressionRank; ++i) {
            auto forEquationOp = builder.create<ForEquationOp>(loc, 0, memberArrayType.getShape()[i] - 1);
            inductionVariables.push_back(forEquationOp.induction());
            builder.setInsertionPointToStart(forEquationOp.bodyBlock());
          }

          auto equationOp = builder.create<InitialEquationOp>(loc);
          assert(equationOp.getBodyRegion().empty());
          mlir::Block* equationBodyBlock = builder.createBlock(&equationOp.getBodyRegion());
          builder.setInsertionPointToStart(equationBodyBlock);

          // Clone the operations
          mlir::BlockAndValueMapping mapping;

          for (auto& op : startOp.getBody()->getOperations()) {
            if (!mlir::isa<YieldOp>(op)) {
              builder.clone(op, mapping);
            }
          }

          // Left-hand side
          mlir::Value lhsValue = startOp.getVariable();

          if (lhsValue.getType().isa<ArrayType>()) {
            lhsValue = builder.create<LoadOp>(loc, lhsValue, inductionVariables);
          }

          // Right-hand side
          mlir::Value rhsValue = mapping.lookup(yieldOp.getValues()[0]);

          // Create the assignment
          mlir::Value lhsTuple = builder.create<EquationSideOp>(loc, lhsValue);
          mlir::Value rhsTuple = builder.create<EquationSideOp>(loc, rhsValue);
          builder.create<EquationSidesOp>(loc, lhsTuple, rhsTuple);
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
          return terminator.getLhsValues().size() == 1 && terminator.getRhsValues().size() == 1;
        });

        target.addDynamicallyLegalOp<InitialEquationOp>([](InitialEquationOp op) {
          auto terminator = mlir::cast<EquationSidesOp>(op.bodyBlock()->getTerminator());
          return terminator.getLhsValues().size() == 1 && terminator.getRhsValues().size() == 1;
        });

        mlir::RewritePatternSet patterns(&getContext());
        patterns.insert<EquationOpMultipleValuesPattern>(&getContext());
        patterns.insert<InitialEquationOpMultipleValuesPattern>(&getContext());

        return applyPartialConversion(getOperation(), target, std::move(patterns));
      }

      mlir::LogicalResult convertToSingleEquationBody(ModelOp modelOp)
      {
        mlir::OpBuilder builder(modelOp);
        std::vector<EquationInterface> equations;

        // Collect all the equations inside the region
        for (auto op : modelOp.getBodyRegion().getOps<EquationInterface>()) {
          equations.push_back(op);
        }

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

          // Clone them starting from the outermost one
          for (size_t i = 0, e = parents.size(); i < e; ++i) {
            auto clonedParent = mlir::cast<ForEquationOp>(builder.clone(*parents[e - i - 1].getOperation(), mapping));
            builder.setInsertionPointToEnd(clonedParent.bodyBlock());
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

          // Remove the derived indices
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

      mlir::LogicalResult convertFuncOps()
      {
        mlir::ConversionTarget target(getContext());

        mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
        marco::codegen::TypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);

        target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType());
        });

        target.addDynamicallyLegalOp<mlir::func::CallOp>([&](mlir::func::CallOp op) {
          for (const auto& type : op.operands().getTypes()) {
            if (!typeConverter.isLegal(type)) {
              return false;
            }
          }

          for (const auto& type : op.getResults().getTypes()) {
            if (!typeConverter.isLegal(type)) {
              return false;
            }
          }

          return true;
        });

        target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
          return true;
        });

        mlir::RewritePatternSet patterns(&getContext());
        patterns.insert<FuncOpTypesPattern>(typeConverter, &getContext());
        patterns.insert<CallOpTypesPattern>(typeConverter, &getContext());

        return applyPartialConversion(getOperation(), target, std::move(patterns));
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
