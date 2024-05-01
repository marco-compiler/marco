#include "marco/Codegen/Transforms/FunctionDefaultValuesConversion.h"
#include "marco/Dialect/BaseModelica/BaseModelicaDialect.h"
#include "marco/Dialect/BaseModelica/DefaultValuesDependencyGraph.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_FUNCTIONDEFAULTVALUESCONVERSIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

namespace
{
  class FunctionDefaultValuesConversionPass
      : public mlir::bmodelica::impl::FunctionDefaultValuesConversionPassBase<
            FunctionDefaultValuesConversionPass>
  {
    public:
      using FunctionDefaultValuesConversionPassBase<
          FunctionDefaultValuesConversionPass>
        ::FunctionDefaultValuesConversionPassBase;

      void runOnOperation() override;

    private:
      mlir::LogicalResult convertInputDefaultValues(
          mlir::SymbolTableCollection& symbolTableCollection,
          llvm::ArrayRef<FunctionOp> functionOps);

      mlir::LogicalResult convertProtectedAndOutputDefaultValues(
          mlir::SymbolTableCollection& symbolTableCollection,
          llvm::ArrayRef<FunctionOp> functionOps);

      mlir::LogicalResult convertProtectedAndOutputDefaultValues(
          mlir::SymbolTableCollection& symbolTableCollection,
          std::mutex& symbolTableMutex,
          FunctionOp functionOp);

      mlir::LogicalResult eraseDefaultOps(
          llvm::ArrayRef<FunctionOp> functionOps);
  };
}

namespace
{
  class DefaultOpComputationOrderings
  {
    public:
      llvm::ArrayRef<VariableOp> get(FunctionOp functionOp) const
      {
        auto it = orderings.find(functionOp);

        // If the assertion doesn't hold, then verification is wrong.
        assert(it != orderings.end());

        return it->getSecond();
      }

      void set(
          FunctionOp functionOp,
          llvm::ArrayRef<VariableOp> variablesOrder)
      {
        for (VariableOp variableOp : variablesOrder) {
          orderings[functionOp].push_back(variableOp);
        }
      }

    private:
      llvm::DenseMap<FunctionOp, llvm::SmallVector<VariableOp, 3>> orderings;
  };

  class CallFiller : public mlir::OpRewritePattern<CallOp>
  {
    public:
      CallFiller(
          mlir::MLIRContext* context,
          mlir::SymbolTableCollection& symbolTableCollection,
          const DefaultOpComputationOrderings& orderings)
          : mlir::OpRewritePattern<CallOp>(context),
            symbolTableCollection(&symbolTableCollection),
            orderings(&orderings)
      {
      }

      mlir::LogicalResult matchAndRewrite(
          CallOp op, mlir::PatternRewriter& rewriter) const override
      {
        auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

        auto functionOp = mlir::cast<FunctionOp>(
            op.getFunction(moduleOp, *symbolTableCollection));

        // Collect the input variables.
        llvm::SmallVector<VariableOp, 3> inputVariables;

        for (VariableOp variableOp : functionOp.getVariables()) {
          if (variableOp.isInput()) {
            inputVariables.push_back(variableOp);
          }
        }

        // Map the default values.
        llvm::DenseMap<mlir::StringAttr, DefaultOp> defaultOps;

        for (DefaultOp defaultOp : functionOp.getDefaultValues()) {
          defaultOps[defaultOp.getVariableAttr()] = defaultOp;
        }

        // Determine the new arguments, ordered according to the declaration
        // of variables inside the function.
        llvm::SmallVector<mlir::Value, 3> newArgs;
        llvm::StringMap<mlir::Value> variables;

        if (auto argNames = op.getArgNames()) {
          for (const auto& [argName, argValue] : llvm::zip(
                   argNames->getAsRange<mlir::FlatSymbolRefAttr>(),
                   op.getArgs())) {
            variables[argName.getValue()] = argValue;
          }

          for (VariableOp variableOp : orderings->get(functionOp)) {
            auto variableName = variableOp.getSymNameAttr();

            if (variables.find(variableName) == variables.end()) {
              DefaultOp defaultOp = defaultOps[variableName];

              mlir::Value defaultValue =
                  cloneDefaultOpBody(rewriter, defaultOp, variables);

              variables[variableName] = defaultValue;
            }
          }
        } else {
          for (auto arg : llvm::enumerate(op.getArgs())) {
            mlir::Value argValue = arg.value();
            variables[inputVariables[arg.index()].getSymNameAttr()] = argValue;
          }

          auto missingVariables = llvm::ArrayRef(inputVariables)
                                      .drop_front(op.getArgs().size());

          llvm::DenseSet<mlir::StringAttr> missingVariableNames;

          for (VariableOp variableOp : missingVariables) {
            missingVariableNames.insert(variableOp.getSymNameAttr());
          }

          for (VariableOp variableOp : orderings->get(functionOp)) {
            auto variableName = variableOp.getSymNameAttr();

            if (missingVariableNames.contains(variableName)) {
              DefaultOp defaultOp = defaultOps[variableName];

              mlir::Value defaultValue =
                  cloneDefaultOpBody(rewriter, defaultOp, variables);

              variables[variableName] = defaultValue;
            }
          }
        }

        for (VariableOp variableOp : inputVariables) {
          newArgs.push_back(variables[variableOp.getSymNameAttr()]);
        }

        // Create the new call operation.
        assert(newArgs.size() == inputVariables.size());

        rewriter.replaceOpWithNewOp<CallOp>(
            op, op.getCallee(), op.getResultTypes(), newArgs);

        return mlir::success();
      }

    private:
      mlir::Value cloneDefaultOpBody(
          mlir::OpBuilder& builder,
          DefaultOp defaultOp,
          const llvm::StringMap<mlir::Value>& variables) const
      {
        mlir::IRMapping mapping;

        for (auto& op : defaultOp.getOps()) {
          if (auto yieldOp = mlir::dyn_cast<YieldOp>(op)) {
            assert(yieldOp.getValues().size() == 1);
            return mapping.lookup(yieldOp.getValues()[0]);
          } else if (auto getOp = mlir::dyn_cast<VariableGetOp>(op)) {
            auto mappedVariableIt = variables.find(getOp.getVariable());
            assert(mappedVariableIt != variables.end());
            mapping.map(getOp.getResult(), mappedVariableIt->getValue());
          } else {
            builder.clone(op, mapping);
          }
        }

        llvm_unreachable("YieldOp not found in DefaultOp");
        return nullptr;
      }

    private:
      mlir::SymbolTableCollection* symbolTableCollection;
      const DefaultOpComputationOrderings* orderings;
  };
}

void FunctionDefaultValuesConversionPass::runOnOperation()
{
  mlir::ModuleOp moduleOp = getOperation();
  mlir::SymbolTableCollection symbolTableCollection;

  // Collect the functions.
  llvm::SmallVector<FunctionOp> functionOps;

  moduleOp.walk([&](FunctionOp functionOp) {
    functionOps.push_back(functionOp);
  });

  // Add the missing arguments to function calls.
  if (mlir::failed(convertInputDefaultValues(
          symbolTableCollection, functionOps))) {
    return signalPassFailure();
  }

  // Copy the default assignments for output and protected variables to the
  // beginning of the function body.
  if (mlir::failed(convertProtectedAndOutputDefaultValues(
          symbolTableCollection, functionOps))) {
    return signalPassFailure();
  }

  // Erase the DefaultOps.
  if (mlir::failed(eraseDefaultOps(functionOps))) {
    return signalPassFailure();
  }
}

mlir::LogicalResult
FunctionDefaultValuesConversionPass::convertInputDefaultValues(
    mlir::SymbolTableCollection& symbolTableCollection,
    llvm::ArrayRef<FunctionOp> functionOps)
{
  mlir::ModuleOp moduleOp = getOperation();

  // Determine the order of computation for the default values of input
  // variables.
  DefaultOpComputationOrderings orderings;

  for (FunctionOp functionOp : functionOps) {
    llvm::StringMap<DefaultOp> defaultOps;

    for (DefaultOp defaultOp : functionOp.getDefaultValues()) {
      defaultOps[defaultOp.getVariable()] = defaultOp;
    }

    llvm::SmallVector<VariableOp, 3> inputVariables;

    for (VariableOp variableOp : functionOp.getVariables()) {
      if (variableOp.isInput()) {
        inputVariables.push_back(variableOp);
      }
    }

    DefaultValuesDependencyGraph defaultValuesGraph(defaultOps);
    defaultValuesGraph.addVariables(inputVariables);
    defaultValuesGraph.discoverDependencies();

    orderings.set(functionOp, defaultValuesGraph.reversePostOrder());
  }

  // Fill the calls with missing arguments.
  mlir::ConversionTarget target(getContext());

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  target.addDynamicallyLegalOp<CallOp>([&](CallOp op) {
    mlir::Operation* callee = op.getFunction(moduleOp, symbolTableCollection);

    if (!mlir::isa<FunctionOp>(callee)) {
      return true;
    }

    auto functionOp = mlir::cast<FunctionOp>(callee);

    size_t numOfInputVariables = llvm::count_if(
        functionOp.getVariables(),
        [](VariableOp variableOp) {
          return variableOp.isInput();
        });

    return op.getArgs().size() == numOfInputVariables;
  });

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<CallFiller>(&getContext(), symbolTableCollection, orderings);

  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

static AlgorithmOp getFirstAlgorithmOp(FunctionOp functionOp)
{
  for (auto& op : functionOp.getOps()) {
    if (auto algorithmOp = mlir::dyn_cast<AlgorithmOp>(op)) {
      return algorithmOp;
    }
  }

  return nullptr;
}

mlir::LogicalResult
FunctionDefaultValuesConversionPass::convertProtectedAndOutputDefaultValues(
    mlir::SymbolTableCollection& symbolTableCollection,
    llvm::ArrayRef<FunctionOp> functionOps)
{
  std::mutex symbolTableCollectionMutex;

  return mlir::failableParallelForEach(
      &getContext(), functionOps,
      [&](FunctionOp functionOp) -> mlir::LogicalResult {
        return convertProtectedAndOutputDefaultValues(
            symbolTableCollection, symbolTableCollectionMutex, functionOp);
      });
}

mlir::LogicalResult
FunctionDefaultValuesConversionPass::convertProtectedAndOutputDefaultValues(
    mlir::SymbolTableCollection& symbolTableCollection,
    std::mutex& symbolTableCollectionMutex,
    FunctionOp functionOp)
{
  mlir::IRRewriter rewriter(&getContext());

  // Collect the operations computing the default values and order them so
  // that dependencies are respected.
  llvm::StringMap<DefaultOp> defaultOps;

  DefaultValuesDependencyGraph defaultValuesGraph(defaultOps);

  for (DefaultOp defaultOp : functionOp.getOps<DefaultOp>()) {
    std::lock_guard<std::mutex> lockGuard(symbolTableCollectionMutex);
    VariableOp variableOp = defaultOp.getVariableOp(symbolTableCollection);

    if (!variableOp.isInput()) {
      defaultValuesGraph.addVariables(variableOp);
      defaultOps[variableOp.getSymName()] = defaultOp;
    }
  }

  defaultValuesGraph.discoverDependencies();

  AlgorithmOp algorithmOp = getFirstAlgorithmOp(functionOp);

  if (!algorithmOp) {
    rewriter.setInsertionPointToEnd(functionOp.getBody());
    algorithmOp = rewriter.create<AlgorithmOp>(functionOp.getLoc());
    rewriter.createBlock(&algorithmOp.getBodyRegion());
  }

  // The assignments are performed at the beginning of the function body.
  rewriter.setInsertionPointToStart(algorithmOp.getBody());

  for (VariableOp variableOp : defaultValuesGraph.reversePostOrder()) {
    auto defaultOpIt = defaultOps.find(variableOp.getSymName());

    if (defaultOpIt != defaultOps.end()) {
      DefaultOp defaultOp = defaultOpIt->getValue();
      mlir::IRMapping mapping;

      for (auto& nestedOp : defaultOp.getOps()) {
        if (auto yieldOp = mlir::dyn_cast<YieldOp>(nestedOp)) {
          assert(yieldOp.getValues().size() == 1);
          mlir::Value yieldedValue = yieldOp.getValues()[0];

          rewriter.create<VariableSetOp>(
              defaultOp.getLoc(), variableOp,
              mapping.lookup(yieldedValue));
        } else {
          rewriter.clone(nestedOp, mapping);
        }
      }
    }
  }

  return mlir::success();
}

namespace
{
  struct DefaultOpRemovePattern : public mlir::OpRewritePattern<DefaultOp>
  {
    using mlir::OpRewritePattern<DefaultOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        DefaultOp op, mlir::PatternRewriter& rewriter) const override
    {
      rewriter.eraseOp(op);
      return mlir::success();
    }
  };
}

mlir::LogicalResult FunctionDefaultValuesConversionPass::eraseDefaultOps(
    llvm::ArrayRef<FunctionOp> functionOps)
{
  return mlir::failableParallelForEach(
      &getContext(), functionOps,
      [&](FunctionOp functionOp) -> mlir::LogicalResult {
        mlir::ConversionTarget target(getContext());
        target.addIllegalOp<DefaultOp>();

        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<DefaultOpRemovePattern>(&getContext());

        return applyPartialConversion(functionOp, target, std::move(patterns));
      });
}

namespace mlir::bmodelica
{
  std::unique_ptr<mlir::Pass> createFunctionDefaultValuesConversionPass()
  {
    return std::make_unique<FunctionDefaultValuesConversionPass>();
  }
}
