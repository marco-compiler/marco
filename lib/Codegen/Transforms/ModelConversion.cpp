#include "marco/Codegen/Transforms/ModelConversion.h"
#include "marco/Codegen/Conversion/IDAToLLVM/LLVMTypeConverter.h"
#include "marco/Codegen/Conversion/KINSOLToLLVM/LLVMTypeConverter.h"
#include "marco/Codegen/Conversion/ModelicaCommon/LLVMTypeConverter.h"
#include "marco/Codegen/Transforms/ModelSolving/Solvers/EulerForward.h"
#include "marco/Codegen/Transforms/ModelSolving/Solvers/IDA.h"
#include "marco/Codegen/Transforms/ModelSolving/DerivativesMap.h"
#include "marco/Codegen/Transforms/ModelSolving/Utils.h"
#include "marco/Codegen/Runtime.h"
#include "marco/VariableFilter/VariableFilter.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_MODELCONVERSIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

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

namespace
{
  class ModelConversionPass : public mlir::modelica::impl::ModelConversionPassBase<ModelConversionPass>
  {
    public:
      using ModelConversionPassBase::ModelConversionPassBase;

      void runOnOperation() override
      {
        if (mlir::failed(createSimulationHooks())) {
          mlir::emitError(getOperation().getLoc(), "Can't create the simulation hooks");
          return signalPassFailure();
        }

        if (mlir::failed(convertFuncOps())) {
          return signalPassFailure();
        }
      }

    private:
      mlir::LogicalResult createSimulationHooks();

      std::unique_ptr<ModelSolver> getSolver(
          mlir::LLVMTypeConverter& typeConverter,
          VariableFilter& variablesFilter);

      mlir::LogicalResult convertFuncOps();
  };
}

mlir::LogicalResult ModelConversionPass::createSimulationHooks()
{
  mlir::ModuleOp module = getOperation();
  std::vector<ModelOp> modelOps;

  module.walk([&](ModelOp modelOp) {
    modelOps.push_back(modelOp);
  });

  assert(llvm::count_if(modelOps, [&](ModelOp modelOp) {
           return modelOp.getSymName() == model;
         }) <= 1 && "More than one model matches the requested model name, but only one can be converted into a simulation");

  mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
  llvmLoweringOptions.dataLayout.reset(dataLayout);

  // Modelica types converter.
  mlir::modelica::LLVMTypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);

  // Add the conversions for the IDA types.
  mlir::ida::LLVMTypeConverter idaTypeConverter(&getContext(), llvmLoweringOptions);

  // Add the conversions for the KINSOL types.
  mlir::kinsol::LLVMTypeConverter kinsolTypeConverter(&getContext(), llvmLoweringOptions);

  typeConverter.addConversion([&](mlir::ida::InstanceType type) {
    return idaTypeConverter.convertType(type);
  });

  typeConverter.addConversion([&](mlir::ida::VariableType type) {
    return idaTypeConverter.convertType(type);
  });

  typeConverter.addConversion([&](mlir::ida::EquationType type) {
    return idaTypeConverter.convertType(type);
  });

  typeConverter.addConversion([&](mlir::kinsol::InstanceType type) {
    return kinsolTypeConverter.convertType(type);
  });

  typeConverter.addConversion([&](mlir::kinsol::VariableType type) {
    return kinsolTypeConverter.convertType(type);
  });

  typeConverter.addConversion([&](mlir::kinsol::EquationType type) {
    return kinsolTypeConverter.convertType(type);
  });

  for (auto& modelOp : modelOps) {
    if (modelOp.getSymName() != model) {
      modelOp.erase();
      continue;
    }

    auto expectedVariablesFilter = VariableFilter::fromString(variablesFilter);
    std::unique_ptr<VariableFilter> variablesFilterInstance;

    if (!expectedVariablesFilter) {
      modelOp.emitWarning("Invalid variable filter string. No filtering will take place");
      variablesFilterInstance = std::make_unique<VariableFilter>();
    } else {
      variablesFilterInstance = std::make_unique<VariableFilter>(std::move(*expectedVariablesFilter));
    }

    std::unique_ptr<ModelSolver> modelSolver = getSolver(typeConverter, *variablesFilterInstance);
    mlir::OpBuilder builder(modelOp);

    // Parse the derivatives map.
    DerivativesMap derivativesMap;

    if (mlir::failed(readDerivativesMap(modelOp, derivativesMap))) {
      return mlir::failure();
    }

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
      if (mlir::failed(modelSolver->solveICModel(builder, model))) {
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
      if (mlir::failed(modelSolver->solveMainModel(builder, model))) {
        return mlir::failure();
      }
    }

    if (mlir::failed(modelSolver->createGetModelNameFunction(builder, modelOp))) {
      modelOp.emitError("Could not create the '" + ModelSolver::getModelNameFunctionName + "' function");
      return mlir::failure();
    }

    if (mlir::failed(modelSolver->createGetNumOfVariablesFunction(builder, modelOp))) {
      modelOp.emitError("Could not create the '" + ModelSolver::getNumOfVariablesFunctionName + "' function");
      return mlir::failure();
    }

    if (mlir::failed(modelSolver->createGetVariableNameFunction(builder, modelOp))) {
      modelOp.emitError("Could not create the '" + ModelSolver::getVariableNameFunctionName + "' function");
      return mlir::failure();
    }

    if (mlir::failed(modelSolver->createGetVariableRankFunction(builder, modelOp))) {
      modelOp.emitError("Could not create the '" + ModelSolver::getVariableRankFunctionName + "' function");
      return mlir::failure();
    }

    if (mlir::failed(modelSolver->createGetVariableNumOfPrintableRangesFunction(builder, modelOp, derivativesMap))) {
      modelOp.emitError("Could not create the '" + ModelSolver::getVariableNumOfPrintableRangesFunctionName + "' function");
      return mlir::failure();
    }

    if (mlir::failed(modelSolver->createGetVariablePrintableRangeBeginFunction(builder, modelOp, derivativesMap))) {
      modelOp.emitError("Could not create the '" + ModelSolver::getVariablePrintableRangeBeginFunctionName + "' function");
      return mlir::failure();
    }

    if (mlir::failed(modelSolver->createGetVariablePrintableRangeEndFunction(builder, modelOp, derivativesMap))) {
      modelOp.emitError("Could not create the '" + ModelSolver::getVariablePrintableRangeEndFunctionName + "' function");
      return mlir::failure();
    }

    if (mlir::failed(modelSolver->createGetVariableValueFunction(builder, modelOp))) {
      modelOp.emitError("Could not create the '" + ModelSolver::getVariableValueFunctionName + "' function");
      return mlir::failure();
    }

    if (mlir::failed(modelSolver->createGetDerivativeFunction(builder, modelOp, derivativesMap))) {
      modelOp.emitError("Could not create the '" + ModelSolver::getDerivativeFunctionName + "' function");
      return mlir::failure();
    }

    if (mlir::failed(modelSolver->createGetTimeFunction(builder, modelOp))) {
      modelOp.emitError("Could not create the '" + ModelSolver::getTimeFunctionName + "' function");
      return mlir::failure();
    }

    if (mlir::failed(modelSolver->createSetTimeFunction(builder, modelOp))) {
      modelOp.emitError("Could not create the '" + ModelSolver::setTimeFunctionName + "' function");
      return mlir::failure();
    }

    if (mlir::failed(modelSolver->createInitFunction(builder, modelOp))) {
      modelOp.emitError("Could not create the '" + ModelSolver::initFunctionName + "' function");
      return mlir::failure();
    }

    if (mlir::failed(modelSolver->createDeinitFunction(builder, modelOp))) {
      modelOp.emitError("Could not create the '" + ModelSolver::deinitFunctionName + "' function");
      return mlir::failure();
    }

    if (emitSimulationMainFunction) {
      if (mlir::failed(modelSolver->createMainFunction(builder, modelOp))) {
        modelOp.emitError("Could not create the '" + ModelSolver::mainFunctionName + "' function");
        return mlir::failure();
      }
    }

    // Erase the model operation, which has been converted to algorithmic code.
    modelOp.erase();
  }

  return mlir::success();
}

std::unique_ptr<ModelSolver> ModelConversionPass::getSolver(
    mlir::LLVMTypeConverter& typeConverter,
    VariableFilter& variablesFilter)
{
  auto solverKind = solver.getKind();

  if (solverKind == Solver::Kind::forwardEuler) {
    return std::make_unique<EulerForwardSolver>(
        typeConverter, variablesFilter);
  }

  if (solverKind == Solver::Kind::ida) {
    return std::make_unique<IDASolver>(
        typeConverter, variablesFilter, IDACleverDAE);
  }

  llvm_unreachable("Unknown solver");
  return nullptr;
}

mlir::LogicalResult ModelConversionPass::convertFuncOps()
{
  mlir::ConversionTarget target(getContext());

  mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
  llvmLoweringOptions.dataLayout.reset(dataLayout);

  mlir::modelica::LLVMTypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);

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

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createModelConversionPass()
  {
    return std::make_unique<ModelConversionPass>();
  }

  std::unique_ptr<mlir::Pass> createModelConversionPass(const ModelConversionPassOptions& options)
  {
    return std::make_unique<ModelConversionPass>(options);
  }
}