#include "marco/Codegen/Transforms/ScheduledModelToSimulation.h"
#include "marco/Codegen/Transforms/ModelSolving/Equation.h"
#include "marco/Codegen/Transforms/ModelSolving/EquationImpl.h"
#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include "marco/Codegen/Transforms/ModelSolving/ModelConverter.h"
#include "marco/Codegen/Transforms/ModelSolving/Scheduling.h"
#include "marco/Codegen/Transforms/ModelSolving/TypeConverter.h"
#include "marco/Codegen/Transforms/ModelSolving/Utils.h"
#include "marco/Codegen/Utils.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_CONVERTSCHEDULEDMODELTOSIMULATIONPASS
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
  class ConvertScheduledModelToSimulationPass : public mlir::modelica::impl::ConvertScheduledModelToSimulationPassBase<ConvertScheduledModelToSimulationPass>
  {
    public:
      ConvertScheduledModelToSimulationPass() : ConvertScheduledModelToSimulationPassBase()
      {
        variableFilter = new VariableFilter();
        ownVariableFilter = true;
      }

      ConvertScheduledModelToSimulationPass(const ConvertScheduledModelToSimulationPassOptions& options, VariableFilter* variableFilter, Solver solver, IDAOptions idaOptions)
          : ConvertScheduledModelToSimulationPassBase(options),
            variableFilter(variableFilter),
            ownVariableFilter(false),
            solver(solver),
            idaOptions(idaOptions)
      {
      }

      ~ConvertScheduledModelToSimulationPass()
      {
        if (ownVariableFilter) {
          delete variableFilter;
        }
      }

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
          if (mlir::failed(processModelOp(builder, modelOp))) {
            return signalPassFailure();
          }
        }
      }

    private:
      mlir::LogicalResult processModelOp(mlir::OpBuilder& builder, ModelOp modelOp);

      mlir::LogicalResult convertFuncOps();

    private:
      VariableFilter* variableFilter;
      bool ownVariableFilter;
      Solver solver;
      IDAOptions idaOptions;
  };
}

mlir::LogicalResult ConvertScheduledModelToSimulationPass::processModelOp(mlir::OpBuilder& builder, ModelOp modelOp)
{
  Model<Equation> initialConditionsModel(modelOp);
  Model<Equation> mainModel(modelOp);

  // Retrieve the derivatives map computed by the legalization pass
  DerivativesMap derivativesMap = readDerivativesMap(modelOp);
  initialConditionsModel.setDerivativesMap(derivativesMap);
  mainModel.setDerivativesMap(derivativesMap);

  // Discover variables and equations belonging to the 'initial' model
  initialConditionsModel.setVariables(discoverVariables(initialConditionsModel.getOperation()));
  initialConditionsModel.setEquations(discoverInitialEquations(initialConditionsModel.getOperation(), initialConditionsModel.getVariables()));

  // Discover variables and equations belonging to the 'main' model
  mainModel.setVariables(discoverVariables(mainModel.getOperation()));
  mainModel.setEquations(discoverEquations(mainModel.getOperation(), mainModel.getVariables()));

  // Obtain the matched models
  Model<MatchedEquation> matchedInitialConditionsModel(modelOp);
  Model<MatchedEquation> matchedMainModel(modelOp);

  readMatchingAttributes(initialConditionsModel, matchedInitialConditionsModel);
  readMatchingAttributes(mainModel, matchedMainModel);

  // Obtain the scheduled models
  Model<ScheduledEquationsBlock> scheduledInitialConditionsModel(modelOp);
  Model<ScheduledEquationsBlock> scheduledMainModel(modelOp);

  readSchedulingAttributes(matchedInitialConditionsModel, scheduledInitialConditionsModel);
  readSchedulingAttributes(matchedMainModel, scheduledMainModel);

  // Create the simulation functions
  mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
  marco::codegen::TypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);
  ModelConverter modelConverter(typeConverter, variableFilter, solver, startTime, endTime, timeStep, idaOptions);

  if (auto res = modelConverter.createGetModelNameFunction(builder, modelOp); mlir::failed(res)) {
    modelOp.emitError("Could not create the '" + ModelConverter::getModelNameFunctionName + "' function");
    return res;
  }

  if (auto res = modelConverter.createInitFunction(builder, modelOp); mlir::failed(res)) {
    modelOp.emitError("Could not create the '" + ModelConverter::initFunctionName + "' function");
    return res;
  }

  if (auto res = modelConverter.createDeinitFunction(builder, modelOp); mlir::failed(res)) {
    modelOp.emitError("Could not create the '" + ModelConverter::deinitFunctionName + "' function");
    return res;
  }

  if (emitMain) {
    if (auto res = modelConverter.createMainFunction(builder, modelOp); mlir::failed(res)) {
      modelOp.emitError("Could not create the '" + ModelConverter::mainFunctionName + "' function");
      return res;
    }
  }

  if (auto res = modelConverter.convertInitialModel(builder, scheduledInitialConditionsModel); mlir::failed(res)) {
    return res;
  }

  if (auto res = modelConverter.convertMainModel(builder, scheduledMainModel); mlir::failed(res)) {
    return res;
  }

  // Erase the model operation, which has been converted to algorithmic code
  modelOp.erase();

  // Convert the functions having a Modelica type within their signature.
  if (auto res = convertFuncOps(); mlir::failed(res)) {
    return res;
  }

  return mlir::success();
}

mlir::LogicalResult ConvertScheduledModelToSimulationPass::convertFuncOps()
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

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createConvertScheduledModelToSimulationPass()
  {
    return std::make_unique<ConvertScheduledModelToSimulationPass>();
  }

  std::unique_ptr<mlir::Pass> createConvertScheduledModelToSimulationPass(
      const ConvertScheduledModelToSimulationPassOptions& options,
      marco::VariableFilter* variableFilter,
      marco::codegen::Solver solver,
      marco::codegen::IDAOptions idaOptions)
  {
    return std::make_unique<ConvertScheduledModelToSimulationPass>(options, variableFilter, solver, idaOptions);
  }
}
