#include "marco/Codegen/Transforms/CallDefaultValuesInsertion.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/Modelica/DefaultValuesDependencyGraph.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_CALLDEFAULTVALUESINSERTIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  class CallDefaultValuesInsertionPass
      : public mlir::modelica::impl::CallDefaultValuesInsertionPassBase<
            CallDefaultValuesInsertionPass>
  {
    public:
      using CallDefaultValuesInsertionPassBase<CallDefaultValuesInsertionPass>
          ::CallDefaultValuesInsertionPassBase;

      void runOnOperation() override;
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

void CallDefaultValuesInsertionPass::runOnOperation()
{
  mlir::ModuleOp moduleOp = getOperation();
  mlir::SymbolTableCollection symbolTableCollection;

  // Determine the order of computation for the default values of input
  // variables.
  DefaultOpComputationOrderings orderings;
  llvm::SmallVector<ClassInterface> classes;

  for (ClassInterface cls : moduleOp.getOps<ClassInterface>()) {
    classes.push_back(cls);
  }

  while (!classes.empty()) {
    ClassInterface cls = classes.pop_back_val();

    if (auto functionOp =
            mlir::dyn_cast<FunctionOp>(cls.getOperation())) {
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

    // Search for nested functions.
    for (mlir::Region& region : cls->getRegions()) {
      for (ClassInterface nestedCls : region.getOps<ClassInterface>()) {
        classes.push_back(nestedCls);
      }
    }
  }

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

  if (mlir::failed(applyPartialConversion(
          moduleOp, target, std::move(patterns)))) {
    return signalPassFailure();
  }
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createCallDefaultValuesInsertionPass()
  {
    return std::make_unique<CallDefaultValuesInsertionPass>();
  }
}
