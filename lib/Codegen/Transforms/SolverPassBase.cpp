#include "marco/Codegen/Transforms/SolverPassBase.h"
#include "marco/Codegen/Transforms/ModelSolving/Utils.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

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

      // Clone the blocks structure.
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

        // Cast the block arguments.
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

static IndexSet getFilteredIndices(
    mlir::Type variableType,
    llvm::ArrayRef<VariableFilter::Filter> filters)
{
  IndexSet result;

  auto arrayType = variableType.cast<ArrayType>();
  assert(arrayType.hasStaticShape());

  for (const auto& filter : filters) {
    if (!filter.isVisible()) {
      continue;
    }

    auto filterRanges = filter.getRanges();
    assert(filterRanges.size() == static_cast<size_t>(arrayType.getRank()));

    std::vector<Range> ranges;

    for (const auto& range : llvm::enumerate(filterRanges)) {
      // In Modelica, arrays are 1-based. If present, we need to lower by 1
      // the value given by the variable filter.

      auto lowerBound = range.value().hasLowerBound()
          ? range.value().getLowerBound() - 1 : 0;

      auto upperBound = range.value().hasUpperBound()
          ? range.value().getUpperBound()
          : arrayType.getShape()[range.index()];

      ranges.emplace_back(lowerBound, upperBound);
    }

    if (ranges.empty()) {
      // Scalar value.
      ranges.emplace_back(0, 1);
    }

    result += MultidimensionalRange(std::move(ranges));
  }

  return result;
}

static llvm::SmallVector<mlir::simulation::MultidimensionalRangeAttr>
getSimulationMultidimensionalRanges(
    mlir::MLIRContext* context, const IndexSet& indices)
{
  llvm::SmallVector<mlir::simulation::MultidimensionalRangeAttr> result;
  IndexSet canonicalIndices = indices.getCanonicalRepresentation();

  for (const MultidimensionalRange& multidimensionalRange :
       llvm::make_range(canonicalIndices.rangesBegin(),
                        canonicalIndices.rangesEnd())) {
    llvm::SmallVector<std::pair<int64_t, int64_t>> ranges;

    for (unsigned int i = 0, e = multidimensionalRange.rank(); i < e; ++i) {
      ranges.emplace_back(
          multidimensionalRange[i].getBegin(),
          multidimensionalRange[i].getEnd());
    }

    result.push_back(mlir::simulation::MultidimensionalRangeAttr::get(
        context, ranges));
  }

  return result;
}

namespace mlir::modelica::impl
{
  ModelSolver::ModelSolver() = default;

  ModelSolver::~ModelSolver() = default;

  mlir::LogicalResult ModelSolver::convert(
      ModelOp modelOp,
      const VariableFilter& variablesFilter,
      bool processICModel,
      bool processMainModel)
  {
    mlir::OpBuilder builder(modelOp);

    // Parse the derivatives map.
    DerivativesMap derivativesMap;

    if (mlir::failed(readDerivativesMap(modelOp, derivativesMap))) {
      return mlir::failure();
    }

    mlir::simulation::ModuleOp simulationModuleOp = createSimulationModule(
        builder, modelOp, derivativesMap, variablesFilter);

    if (processICModel) {
      // Obtain the scheduled model.
      Model<ScheduledEquationsBlock> model(modelOp);
      model.setVariables(discoverVariables(modelOp));
      model.setDerivativesMap(derivativesMap);

      auto equationsFilter = [](EquationInterface op) {
        return mlir::isa<InitialEquationOp>(op);
      };

      if (mlir::failed(readSchedulingAttributes(model, equationsFilter))) {
        return mlir::failure();
      }

      // Create the simulation functions.
      if (mlir::failed(solveICModel(builder, simulationModuleOp, model))) {
        return mlir::failure();
      }
    }

    if (processMainModel) {
      // Obtain the scheduled model.
      Model<ScheduledEquationsBlock> model(modelOp);
      model.setVariables(discoverVariables(modelOp));
      model.setDerivativesMap(derivativesMap);

      auto equationsFilter = [](EquationInterface op) {
        return mlir::isa<EquationOp>(op);
      };

      if (mlir::failed(readSchedulingAttributes(model, equationsFilter))) {
        return mlir::failure();
      }

      // Create the simulation functions.
      if (mlir::failed(solveMainModel(builder, simulationModuleOp, model))) {
        return mlir::failure();
      }
    }

    if (mlir::failed(createInitFunction(
            builder, modelOp, simulationModuleOp))) {
      return mlir::failure();
    }

    if (mlir::failed(createDeinitFunction(
            builder, modelOp, simulationModuleOp))) {
      return mlir::failure();
    }

    if (mlir::failed(createVariableGetterFunctions(
            builder, modelOp, simulationModuleOp))) {
      return mlir::failure();
    }

    // Erase the model operation, which has been converted to algorithmic code.
    modelOp.erase();

    return mlir::success();
  }

  mlir::simulation::ModuleOp ModelSolver::createSimulationModule(
      mlir::OpBuilder& builder,
      ModelOp modelOp,
      const DerivativesMap& derivativesMap,
      const VariableFilter& variablesFilter)
  {
    auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(moduleOp.getBody());

    llvm::SmallVector<mlir::Attribute> variables;

    for (const auto& variable :
         llvm::enumerate(modelOp.getVariableDeclarationOps())) {
      MemberCreateOp memberCreateOp = variable.value();

      // Determine the printable indices.
      bool printable = true;

      IndexSet printableIndices = getPrintableIndices(
          modelOp, derivativesMap, variablesFilter, variable.index());

      auto simulationPrintableIndices = getSimulationMultidimensionalRanges(
          builder.getContext(), printableIndices);

      if (!memberCreateOp.getMemberType().hasRank()) {
        if (printableIndices.empty()) {
          printable = false;
        }

        simulationPrintableIndices.clear();
      }

      // Create the variable.
      variables.push_back(mlir::simulation::VariableAttr::get(
          builder.getContext(),
          memberCreateOp.getMemberType().toArrayType(),
          memberCreateOp.getSymName(),
          memberCreateOp.getMemberType().getShape(),
          printable,
          simulationPrintableIndices));
    }

    llvm::SmallVector<mlir::Attribute> derivatives;

    for (size_t i = 0, e = variables.size(); i < e; ++i) {
      if (derivativesMap.hasDerivative(i)) {
        auto variableAttr = variables[i].cast<mlir::simulation::VariableAttr>();

        auto derivativeAttr = variables[derivativesMap.getDerivative(i)]
                                  .cast<mlir::simulation::VariableAttr>();

        derivatives.push_back(mlir::simulation::DerivativeAttr::get(
            builder.getContext(), variableAttr, derivativeAttr));
      }
    }

    return builder.create<mlir::simulation::ModuleOp>(
        moduleOp.getLoc(), modelOp.getSymName(), variables, derivatives);
  }

  IndexSet ModelSolver::getPrintableIndices(
      ModelOp modelOp,
      const DerivativesMap& derivativesMap,
      const VariableFilter& variablesFilter,
      unsigned int variable) const
  {
    llvm::SmallVector<llvm::StringRef> names = modelOp.getVariableNames();
    llvm::StringRef name = names[variable];

    auto arrayType = modelOp.getBodyRegion()
                         .getArgument(variable)
                         .getType()
                         .cast<ArrayType>();

    int64_t rank = arrayType.hasRank() ? arrayType.getRank() : 0;

    if (derivativesMap.isDerivative(variable)) {
      auto derivedVariable = derivativesMap.getDerivedVariable(variable);
      llvm::StringRef derivedVariableName = names[derivedVariable];
      auto filters = variablesFilter.getVariableDerInfo(derivedVariableName, rank);

      IndexSet filteredIndices = getFilteredIndices(arrayType, filters);
      IndexSet derivedIndices = derivativesMap.getDerivedIndices(derivedVariable);
      return filteredIndices.intersect(derivedIndices);
    }

    auto filters = variablesFilter.getVariableInfo(name, rank);
    return getFilteredIndices(arrayType, filters);
  }

  mlir::LogicalResult ModelSolver::createInitFunction(
      mlir::OpBuilder& builder,
      ModelOp modelOp,
      mlir::simulation::ModuleOp simulationModuleOp) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(simulationModuleOp.getBody());

    mlir::Location loc = modelOp.getLoc();

    auto initFunctionOp = builder.create<mlir::simulation::InitFunctionOp>(
        loc, simulationModuleOp.getVariablesTypes());

    mlir::Block* entryBlock = initFunctionOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    std::vector<MemberCreateOp> originalMembers;

    // The descriptors of the arrays that compose that runtime data structure.
    std::vector<mlir::Value> structVariables;

    mlir::BlockAndValueMapping membersOpsMapping;

    for (auto& op : modelOp.getVarsRegion().getOps()) {
      if (auto memberCreateOp = mlir::dyn_cast<MemberCreateOp>(op)) {
        auto arrayType = memberCreateOp.getMemberType().toArrayType();
        assert(arrayType.hasStaticShape());

        std::vector<mlir::Value> dynamicDimensions;

        for (const auto& dynamicDimension : memberCreateOp.getDynamicSizes()) {
          dynamicDimensions.push_back(membersOpsMapping.lookup(dynamicDimension));
        }

        mlir::Value array = builder.create<AllocOp>(
            memberCreateOp.getLoc(), arrayType, dynamicDimensions);

        originalMembers.push_back(memberCreateOp);
        structVariables.push_back(array);

        membersOpsMapping.map(memberCreateOp.getResult(), array);
      }
    }

    // Map the body arguments to the new arrays.
    mlir::ValueRange bodyArgs = modelOp.getBodyRegion().getArguments();
    mlir::BlockAndValueMapping startOpsMapping;

    for (const auto& [arg, array] : llvm::zip(bodyArgs, structVariables)) {
      startOpsMapping.map(arg, array);
    }

    // Keep track of the variables for which a start value has been provided.
    llvm::SmallVector<bool> initializedVars(bodyArgs.size(), false);

    modelOp.getBodyRegion().walk([&](StartOp startOp) {
      unsigned int argNumber = startOp.getVariable().cast<mlir::BlockArgument>().getArgNumber();

      // Note that parameters must be set independently of the 'fixed'
      // attribute being true or false.

      if (startOp.getFixed() && !originalMembers[argNumber].isReadOnly()) {
        return;
      }

      builder.setInsertionPointAfterValue(structVariables[argNumber]);

      for (auto& op : startOp.getBodyRegion().getOps()) {
        if (auto yieldOp = mlir::dyn_cast<YieldOp>(op)) {
          mlir::Value valueToBeStored = startOpsMapping.lookup(yieldOp.getValues()[0]);
          mlir::Value destination = startOpsMapping.lookup(startOp.getVariable());

          if (startOp.getEach()) {
            builder.create<ArrayFillOp>(startOp.getLoc(), destination, valueToBeStored);
          } else {
            builder.create<StoreOp>(startOp.getLoc(), valueToBeStored, destination, llvm::None);
          }
        } else {
          builder.clone(op, startOpsMapping);
        }
      }

      // Set the variable as initialized.
      initializedVars[argNumber] = true;
    });

    // The variables without a 'start' attribute must be initialized to zero.
    for (const auto& initialized : llvm::enumerate(initializedVars)) {
      if (initialized.value()) {
        continue;
      }

      mlir::Value destination = structVariables[initialized.index()];
      auto arrayType = destination.getType().cast<ArrayType>();

      builder.setInsertionPointAfterValue(destination);

      mlir::Value zero = builder.create<ConstantOp>(
          destination.getLoc(), getZeroAttr(arrayType.getElementType()));

      if (arrayType.isScalar()) {
        builder.create<StoreOp>(destination.getLoc(), zero, destination, llvm::None);
      } else {
        builder.create<ArrayFillOp>(destination.getLoc(), destination, zero);
      }
    }

    // Create the runtime data structure.
    builder.setInsertionPointToEnd(&initFunctionOp.getBodyRegion().back());
    builder.create<mlir::simulation::YieldOp>(loc, structVariables);

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createDeinitFunction(
      mlir::OpBuilder& builder,
      ModelOp modelOp,
      mlir::simulation::ModuleOp simulationModuleOp) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(simulationModuleOp.getBody());

    mlir::Location loc = modelOp.getLoc();

    auto deinitFunctionOp = builder.create<mlir::simulation::DeinitFunctionOp>(
        loc, simulationModuleOp.getVariablesTypes());

    mlir::Block* entryBlock = deinitFunctionOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    for (mlir::Value var : deinitFunctionOp.getVariables()) {
      builder.create<FreeOp>(loc, var);
    }

    builder.create<mlir::simulation::YieldOp>(loc, llvm::None);

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createVariableGetterFunctions(
      mlir::OpBuilder& builder,
      mlir::modelica::ModelOp modelOp,
      mlir::simulation::ModuleOp simulationModuleOp) const
  {
    mlir::Location loc = modelOp.getLoc();

    // Map the variables by their types, so that we can create just a single
    // getter for all the variables with the same type.
    llvm::DenseMap<
        mlir::Type,
        llvm::SmallVector<mlir::Attribute>> variablesMap;

    for (mlir::simulation::VariableAttr variable :
         simulationModuleOp.getVariables()
             .getAsRange<mlir::simulation::VariableAttr>()) {
      variablesMap[variable.getType()].push_back(variable);
    }

    for (const auto& entry : variablesMap) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(simulationModuleOp.getBody());

      auto arrayType = entry.getFirst().cast<ArrayType>();
      int64_t rank = arrayType.hasRank() ? arrayType.getRank() : 0;

      auto getterOp = builder.create<mlir::simulation::VariableGetterOp>(
          loc, entry.getSecond(), arrayType.getElementType(), arrayType, rank);

      mlir::Block* entryBlock = getterOp.addEntryBlock();
      builder.setInsertionPointToStart(entryBlock);

      mlir::Value result = builder.create<LoadOp>(
          loc, getterOp.getVariable(), getterOp.getIndices());

      builder.create<mlir::simulation::YieldOp>(loc, result);
    }

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::legalizeFuncOps(
      mlir::ModuleOp moduleOp,
      mlir::TypeConverter& typeConverter) const
  {
    mlir::ConversionTarget target(*moduleOp->getContext());

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

    mlir::RewritePatternSet patterns(moduleOp->getContext());
    patterns.insert<FuncOpTypesPattern>(typeConverter, moduleOp->getContext());
    patterns.insert<CallOpTypesPattern>(typeConverter, moduleOp->getContext());

    return applyPartialConversion(moduleOp, target, std::move(patterns));
  }
}

//===---------------------------------------------------------------------===//
// ModelConversionTestPass

namespace mlir::modelica
{
#define GEN_PASS_DEF_MODELCONVERSIONTESTPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

namespace
{
  class TestSolver : public mlir::modelica::impl::ModelSolver
  {
    public:
      mlir::LogicalResult solveICModel(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp,
          const marco::codegen::Model<
              marco::codegen::ScheduledEquationsBlock>& model) override
      {
        // Do nothing.
        return mlir::success();
      }

      mlir::LogicalResult solveMainModel(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp,
          const marco::codegen::Model<
              marco::codegen::ScheduledEquationsBlock>& model) override
      {
        // Do nothing.
        return mlir::success();
      }
  };

  class ModelConversionTestPass
      : public mlir::modelica::impl::ModelConversionTestPassBase<
          ModelConversionTestPass>
  {
    public:
      using ModelConversionTestPassBase::ModelConversionTestPassBase;

      void runOnOperation() override
      {
        mlir::ModuleOp module = getOperation();
        std::vector<ModelOp> modelOps;

        module.walk([&](ModelOp modelOp) {
          modelOps.push_back(modelOp);
        });

        assert(llvm::count_if(modelOps, [&](ModelOp modelOp) {
                 return modelOp.getSymName() == model;
               }) <= 1 && "More than one model matches the requested model name, but only one can be converted into a simulation");

        TestSolver solver;

        auto expectedVariablesFilter = marco::VariableFilter::fromString(variablesFilter);
        std::unique_ptr<marco::VariableFilter> variablesFilterInstance;

        if (!expectedVariablesFilter) {
          getOperation().emitWarning("Invalid variable filter string. No filtering will take place");
          variablesFilterInstance = std::make_unique<marco::VariableFilter>();
        } else {
          variablesFilterInstance = std::make_unique<marco::VariableFilter>(std::move(*expectedVariablesFilter));
        }

        for (ModelOp modelOp : modelOps) {
          if (mlir::failed(solver.convert(
                  modelOp, *variablesFilterInstance,
                  processICModel, processMainModel))) {
            return signalPassFailure();
          }
        }
      }
  };
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createModelConversionTestPass()
  {
    return std::make_unique<ModelConversionTestPass>();
  }

  std::unique_ptr<mlir::Pass> createModelConversionTestPass(
      const ModelConversionTestPassOptions& options)
  {
    return std::make_unique<ModelConversionTestPass>(options);
  }
}
