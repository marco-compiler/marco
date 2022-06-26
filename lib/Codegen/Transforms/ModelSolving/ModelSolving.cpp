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
  struct EquationOpMultipleValuesPattern : public mlir::OpRewritePattern<EquationOp>
  {
    using mlir::OpRewritePattern<EquationOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(EquationOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto terminator = mlir::cast<EquationSidesOp>(op.bodyBlock()->getTerminator());

      if (terminator.getLhsValues().size() != terminator.getRhsValues().size()) {
        return rewriter.notifyMatchFailure(op, "Different amount of values in left-hand and right-hand sides of the equation");
      }

      auto amountOfValues = terminator.getLhsValues().size();

      for (size_t i = 0; i < amountOfValues; ++i) {
        rewriter.setInsertionPointAfter(op);

        auto clone = rewriter.create<EquationOp>(loc);
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

  // Collect the regions to be modified
  llvm::SmallVector<mlir::Region*, 2> regions;
  regions.push_back(&modelOp.getEquationsRegion());
  regions.push_back(&modelOp.getInitialEquationsRegion());

  // Ensure that all regions have the same arguments
  assert(llvm::all_of(regions, [&](const auto& region) {
    return region->getArgumentTypes() == regions[0]->getArgumentTypes();
  }));

  // The list of the new variables
  std::vector<mlir::Value> variables;
  auto terminator = mlir::cast<YieldOp>(modelOp.getInitRegion().back().getTerminator());

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

    auto derArgNumber = addDerivativeToRegions(builder, modelOp.getLoc(), regions, derType, derivedVariableIndices->second);
    derivativesMap.setDerivative(argNumber, derArgNumber);
    derivativesMap.setDerivedIndices(argNumber, derivedVariableIndices->second);

    // Create the variable and initialize it at zero
    builder.setInsertionPoint(terminator);

    auto derivativeName = getNextFullDerVariableName(memberCreateOp.getSymName(), 1);

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

  builder.setInsertionPointToEnd(&modelOp.getInitRegion().back());
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
  regions.push_back(&modelOp.getEquationsRegion());
  regions.push_back(&modelOp.getInitialEquationsRegion());

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

        initialModel.setVariables(discoverVariables(initialModel.getOperation().getInitialEquationsRegion()));
        initialModel.setEquations(discoverEquations(initialModel.getOperation().getInitialEquationsRegion(), initialModel.getVariables()));

        model.setVariables(discoverVariables(model.getOperation().getEquationsRegion()));
        model.setEquations(discoverEquations(model.getOperation().getEquationsRegion(), model.getVariables()));

        // Determine which scalar variables do appear as argument to the derivative operation
        std::map<unsigned int, IndexSet> derivedIndices;
        collectDerivedVariablesIndices(derivedIndices, initialModel.getEquations());
        collectDerivedVariablesIndices(derivedIndices, model.getEquations());

        // The variable splitting may have caused a variable list change and equations splitting.
        // For this reason we need to perform again the discovery process.

        initialModel.setVariables(discoverVariables(initialModel.getOperation().getInitialEquationsRegion()));
        initialModel.setEquations(discoverEquations(initialModel.getOperation().getInitialEquationsRegion(), initialModel.getVariables()));

        model.setVariables(discoverVariables(model.getOperation().getEquationsRegion()));
        model.setEquations(discoverEquations(model.getOperation().getEquationsRegion(), model.getVariables()));

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
        initialModel.setVariables(discoverVariables(initialModel.getOperation().getInitialEquationsRegion()));
        model.setVariables(discoverVariables(model.getOperation().getEquationsRegion()));

        // Additional equations are introduced in case of partially derived arrays. Thus, we need
        // to perform again the discovery of the equations.

        initialModel.setEquations(discoverEquations(initialModel.getOperation().getInitialEquationsRegion(), initialModel.getVariables()));
        model.setEquations(discoverEquations(model.getOperation().getEquationsRegion(), model.getVariables()));

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
      /// Copy the equations declared into the 'equations' region into the 'initial equations' region.
      mlir::LogicalResult copyEquationsAmongInitialEquations(mlir::OpBuilder& builder, ModelOp modelOp)
      {
        if (!modelOp.hasEquationsBlock()) {
          // There is no equation to be copied
          return mlir::success();
        }

        mlir::OpBuilder::InsertionGuard guard(builder);

        if (!modelOp.hasInitialEquationsBlock()) {
          llvm::SmallVector<mlir::Location> variableLocations;

          for (const auto& variable : modelOp.getEquationsRegion().getArguments()) {
            variableLocations.push_back(variable.getLoc());
          }

          builder.createBlock(
              &modelOp.getInitialEquationsRegion(),
              {},
              modelOp.getEquationsRegion().getArgumentTypes(),
              variableLocations);
        }

        // Map the variables declared into the equations region to the ones declared into the initial equations' region
        mlir::BlockAndValueMapping mapping;
        auto originalVariables = modelOp.getEquationsRegion().getArguments();
        auto mappedVariables = modelOp.getInitialEquationsRegion().getArguments();
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
          return terminator.getLhsValues().size() == 1 && terminator.getRhsValues().size() == 1;
        });

        mlir::RewritePatternSet patterns(&getContext());
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
