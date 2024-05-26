#include "marco/Dialect/BaseModelica/Transforms/AutomaticDifferentiation.h"
#include "marco/Dialect/BaseModelica/Transforms/AutomaticDifferentiation/ForwardAD.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelicaDialect.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_AUTOMATICDIFFERENTIATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

namespace
{
  class AutomaticDifferentiationPass
      : public impl::AutomaticDifferentiationPassBase<
          AutomaticDifferentiationPass>
  {
    public:
      using AutomaticDifferentiationPassBase<AutomaticDifferentiationPass>
          ::AutomaticDifferentiationPassBase;

      void runOnOperation() override;

    private:
      mlir::LogicalResult createTimeDerFunctions();

      mlir::LogicalResult createPartialDerFunctions();

      mlir::LogicalResult convertPartialDerFunction(
          mlir::OpBuilder& builder,
          ad::forward::State& state,
          DerFunctionOp derFunctionOp);
  };
}

void AutomaticDifferentiationPass::runOnOperation()
{
  if (mlir::failed(createTimeDerFunctions())) {
    return signalPassFailure();
  }

  if (mlir::failed(createPartialDerFunctions())) {
    return signalPassFailure();
  }
}

mlir::LogicalResult AutomaticDifferentiationPass::createTimeDerFunctions()
{
  auto moduleOp = getOperation();
  mlir::OpBuilder builder(moduleOp);
  ad::forward::State state;

  llvm::SmallVector<FunctionOp, 3> toBeDerived;

  moduleOp->walk([&](FunctionOp op) {
    if (op.getDerivative()) {
      toBeDerived.push_back(op);
    }
  });

  for (FunctionOp functionOp : toBeDerived) {
    auto derivativeAttr = functionOp.getDerivativeAttr();
    uint64_t order = functionOp.getTimeDerivativeOrder();

    if (!ad::forward::createFunctionTimeDerivative(
            builder, state, functionOp, order,
            derivativeAttr.getName(), derivativeAttr.getOrder())) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult AutomaticDifferentiationPass::createPartialDerFunctions()
{
  auto moduleOp = getOperation();
  mlir::OpBuilder builder(moduleOp);

  llvm::SmallVector<DerFunctionOp> toBeProcessed;

  moduleOp->walk([&](DerFunctionOp op) {
    toBeProcessed.push_back(op);
  });

  ad::forward::State state;

  for (DerFunctionOp derFunctionOp : toBeProcessed) {
    if (mlir::failed(convertPartialDerFunction(
            builder, state, derFunctionOp))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult AutomaticDifferentiationPass::convertPartialDerFunction(
    mlir::OpBuilder& builder,
    ad::forward::State& state,
    DerFunctionOp derFunctionOp)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(derFunctionOp);

  // Get the function to be derived.
  auto moduleOp = derFunctionOp->getParentOfType<mlir::ModuleOp>();

  auto baseFunctionOp = resolveSymbol<FunctionOp>(
      moduleOp, state.getSymbolTableCollection(),
      derFunctionOp.getDerivedFunction());

  assert(baseFunctionOp && "Can't find the function to be derived");

  // Create the derivative template function.
  size_t numOfIndependentVars = derFunctionOp.getIndependentVars().size();
  std::optional<FunctionOp> templateFunction = baseFunctionOp;

  for (size_t order = 0; order < numOfIndependentVars; ++order) {
    std::string derivativeName =
        ad::forward::getPartialDerFunctionName(templateFunction->getSymName());

    auto newTemplateFunction = ad::forward::createFunctionPartialDerivative(
        builder, state, *templateFunction, derivativeName);

    if (order != 0) {
      // Erase the temporary template functions.
      mlir::Operation* templateParentSymbolTable =
          (*templateFunction)->getParentWithTrait<mlir::OpTrait::SymbolTable>();

      state.getSymbolTableCollection()
          .getSymbolTable(templateParentSymbolTable)
          .erase(*templateFunction);
    }

    if (!newTemplateFunction) {
      // The template function could not be generated.
      return mlir::failure();
    }

    templateFunction = newTemplateFunction;
  }

  mlir::Location loc = derFunctionOp.getLoc();

  // Get the symbol table.
  mlir::Operation* parentSymbolTable =
      derFunctionOp->getParentWithTrait<mlir::OpTrait::SymbolTable>();

  auto& symbolTable =
      state.getSymbolTableCollection().getSymbolTable(parentSymbolTable);

  // Create the derived function.
  builder.setInsertionPointAfter(derFunctionOp);

  auto derivedFunctionOp = builder.create<FunctionOp>(
      derFunctionOp.getLoc(), derFunctionOp.getSymName());

  builder.setInsertionPointToStart(
      builder.createBlock(&derivedFunctionOp.getBodyRegion()));

  // Declare the variables.
  llvm::SmallVector<VariableOp> inputVariables;
  llvm::SmallVector<VariableOp> outputVariables;

  for (VariableOp variableOp : baseFunctionOp.getVariables()) {
    if (variableOp.isInput() || variableOp.isOutput()) {
      auto clonedVariableOp = mlir::cast<VariableOp>(
          builder.clone(*variableOp.getOperation()));

      if (clonedVariableOp.isInput()) {
        inputVariables.push_back(clonedVariableOp);
      } else if (clonedVariableOp.isOutput()){
        outputVariables.push_back(clonedVariableOp);
      }
    }
  }

  // Create the function body.
  auto algorithmOp = builder.create<AlgorithmOp>(loc);

  builder.setInsertionPointToStart(
      builder.createBlock(&algorithmOp.getBodyRegion()));

  // Call the template function.
  llvm::SmallVector<mlir::Value> args;

  // Forward the input variables.
  for (VariableOp variableOp : inputVariables) {
    args.push_back(builder.create<VariableGetOp>(
        loc, variableOp.getVariableType().unwrap(), variableOp.getSymName()));
  }

  // Append the seeds.
  size_t numberOfSeeds = llvm::count_if(
      baseFunctionOp.getVariables(), [](VariableOp variableOp) {
        return variableOp.isInput();
      });

  // Pre-compute the positions of input variables.
  llvm::StringMap<size_t> baseInputVarsPositions;

  for (auto variable : llvm::enumerate(baseFunctionOp.getVariables())) {
    if (variable.value().isInput()) {
      baseInputVarsPositions[variable.value().getSymName()] = variable.index();
    }
  }

  llvm::SmallVector<mlir::Type> templateFunctionArgTypes =
      templateFunction->getArgumentTypes();

  size_t visitedTemplateArgs = 0;

  for (auto independentVariable :
       llvm::enumerate(derFunctionOp.getIndependentVars())) {
    auto independentVarName =
        independentVariable.value().cast<mlir::StringAttr>().getValue();

    assert(baseInputVarsPositions.contains(independentVarName));
    auto independentVarPos = baseInputVarsPositions[independentVarName];

    for (size_t i = 0; i < numberOfSeeds; ++i) {
      double seed = i == independentVarPos ? 1 : 0;
      auto argType = templateFunctionArgTypes[visitedTemplateArgs++];

      if (seed == 1 && argType.isa<mlir::ShapedType>()) {
        derFunctionOp.emitOpError()
            << "Can't compute a derivative with respect to an array";

        return mlir::failure();
      }

      if (auto tensorType = argType.dyn_cast<mlir::TensorType>()) {
        llvm::SmallVector<mlir::Value> dynamicSizes;

        for (int64_t dim = 0, rank = tensorType.getRank(); dim < rank; ++dim) {
          if (dim == mlir::ShapedType::kDynamic) {
            mlir::Value dimIndex =
                builder.create<ConstantOp>(loc, builder.getIndexAttr(dim));

            mlir::Value dimSize = builder.create<DimOp>(
                loc, args[i % baseInputVarsPositions.size()], dimIndex);

            dynamicSizes.push_back(dimSize);
          }
        }

        auto constantMaterializableType =
            mlir::dyn_cast<ConstantMaterializableTypeInterface>(
                tensorType.getElementType());

        if (!constantMaterializableType) {
          derFunctionOp.emitOpError() << "Can't create seed with type "
                                      << tensorType.getElementType();

          return mlir::failure();
        }

        mlir::Value seedValue =
            constantMaterializableType.materializeFloatConstant(
                builder, loc, seed);

        mlir::Value tensor = builder.create<TensorBroadcastOp>(
            loc, tensorType, seedValue, dynamicSizes);

        args.push_back(tensor);
      } else {
        auto constantMaterializableType =
            mlir::dyn_cast<ConstantMaterializableTypeInterface>(argType);

        if (!constantMaterializableType) {
          derFunctionOp.emitOpError()
              << "Can't create seed with type " << argType;

          return mlir::failure();
        }

        mlir::Value seedValue =
            constantMaterializableType.materializeFloatConstant(
                builder, loc, seed);

        args.push_back(seedValue);
      }
    }

    // The number of seeds increases exponentially with the derivative order.
    numberOfSeeds *= 2;
  }

  auto callOp = builder.create<CallOp>(loc, *templateFunction, args);
  assert(callOp->getNumResults() == outputVariables.size());

  for (const auto& [variable, result] :
       llvm::zip(outputVariables, callOp->getResults())) {
    builder.create<VariableSetOp>(loc, variable, result);
  }

  // Update the symbol table.
  symbolTable.erase(derFunctionOp);
  symbolTable.insert(derivedFunctionOp);

  return mlir::success();
}

namespace mlir::bmodelica
{
  std::unique_ptr<mlir::Pass> createAutomaticDifferentiationPass()
  {
    return std::make_unique<AutomaticDifferentiationPass>();
  }
}
