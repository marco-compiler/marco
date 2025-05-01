#include "marco/Dialect/BaseModelica/Transforms/FunctionDefaultValuesConversion.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/IR/DefaultValuesDependencyGraph.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_FUNCTIONDEFAULTVALUESCONVERSIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class FunctionDefaultValuesConversionPass
    : public mlir::bmodelica::impl::FunctionDefaultValuesConversionPassBase<
          FunctionDefaultValuesConversionPass> {
public:
  using FunctionDefaultValuesConversionPassBase<
      FunctionDefaultValuesConversionPass>::
      FunctionDefaultValuesConversionPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult convertInputDefaultValues(
      mlir::LockedSymbolTableCollection &symbolTableCollection,
      llvm::ArrayRef<FunctionOp> functionOps);

  mlir::LogicalResult convertProtectedAndOutputDefaultValues(
      mlir::SymbolTableCollection &symbolTableCollection,
      FunctionOp functionOp);

  void eraseDefaultOps(llvm::ArrayRef<FunctionOp> functionOps);
};

class DefaultOpComputationOrderings {
public:
  llvm::ArrayRef<VariableOp> get(FunctionOp functionOp) const {
    auto it = orderings.find(functionOp);

    // If the assertion doesn't hold, then verification is wrong.
    assert(it != orderings.end());

    return it->getSecond();
  }

  void set(FunctionOp functionOp, llvm::ArrayRef<VariableOp> variablesOrder) {
    for (VariableOp variableOp : variablesOrder) {
      orderings[functionOp].push_back(variableOp);
    }
  }

private:
  llvm::DenseMap<FunctionOp, llvm::SmallVector<VariableOp, 3>> orderings;
};

void collectFunctionOps(mlir::ModuleOp moduleOp,
                        llvm::SmallVectorImpl<FunctionOp> &functionOps) {
  llvm::SmallVector<ClassInterface> classOps;

  for (auto classOp : moduleOp.getOps<ClassInterface>()) {
    if (auto functionOp = mlir::dyn_cast<FunctionOp>(classOp.getOperation())) {
      functionOps.push_back(functionOp);
    }

    classOps.push_back(classOp);
  }

  while (!classOps.empty()) {
    ClassInterface classOp = classOps.pop_back_val();

    for (mlir::Region &region : classOp->getRegions()) {
      for (auto childClassOp : region.getOps<ClassInterface>()) {
        if (auto functionOp =
                mlir::dyn_cast<FunctionOp>(childClassOp.getOperation())) {
          functionOps.push_back(functionOp);
        }

        classOps.push_back(childClassOp);
      }
    }
  }
}

bool hasMissingArgs(mlir::ModuleOp moduleOp,
                    mlir::SymbolTableCollection &symbolTableCollection,
                    CallOp callOp) {
  // Check if the call is legal.
  mlir::Operation *callee = callOp.getFunction(moduleOp, symbolTableCollection);

  if (!mlir::isa<FunctionOp>(callee)) {
    return false;
  }

  auto functionOp = mlir::cast<FunctionOp>(callee);

  size_t numOfInputVariables =
      llvm::count_if(functionOp.getVariables(), [](VariableOp variableOp) {
        return variableOp.isInput();
      });

  return callOp.getArgs().size() != numOfInputVariables;
}

mlir::Value cloneDefaultOpBody(mlir::OpBuilder &builder, DefaultOp defaultOp,
                               const llvm::StringMap<mlir::Value> &variables) {
  mlir::IRMapping mapping;

  for (auto &op : defaultOp.getOps()) {
    if (auto yieldOp = mlir::dyn_cast<YieldOp>(op)) {
      assert(yieldOp.getValues().size() == 1);
      return mapping.lookup(yieldOp.getValues()[0]);
    }

    if (auto getOp = mlir::dyn_cast<VariableGetOp>(op)) {
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

mlir::LogicalResult
fillCallArgs(mlir::ModuleOp moduleOp,
             mlir::SymbolTableCollection &symbolTableCollection, CallOp callOp,
             const DefaultOpComputationOrderings &orderings) {
  mlir::IRRewriter rewriter(callOp);
  rewriter.setInsertionPoint(callOp);

  auto functionOp = mlir::cast<FunctionOp>(
      callOp.getFunction(moduleOp, symbolTableCollection));

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

  if (auto argNames = callOp.getArgNames()) {
    for (const auto &[argName, argValue] :
         llvm::zip(argNames->getAsRange<mlir::FlatSymbolRefAttr>(),
                   callOp.getArgs())) {
      variables[argName.getValue()] = argValue;
    }

    for (VariableOp variableOp : orderings.get(functionOp)) {
      auto variableName = variableOp.getSymNameAttr();

      if (variables.find(variableName) == variables.end()) {
        DefaultOp defaultOp = defaultOps[variableName];

        mlir::Value defaultValue =
            cloneDefaultOpBody(rewriter, defaultOp, variables);

        variables[variableName] = defaultValue;
      }
    }
  } else {
    for (auto arg : llvm::enumerate(callOp.getArgs())) {
      mlir::Value argValue = arg.value();
      variables[inputVariables[arg.index()].getSymNameAttr()] = argValue;
    }

    auto missingVariables =
        llvm::ArrayRef(inputVariables).drop_front(callOp.getArgs().size());

    llvm::DenseSet<mlir::StringAttr> missingVariableNames;

    for (VariableOp variableOp : missingVariables) {
      missingVariableNames.insert(variableOp.getSymNameAttr());
    }

    for (VariableOp variableOp : orderings.get(functionOp)) {
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

  rewriter.replaceOpWithNewOp<CallOp>(callOp, callOp.getCallee(),
                                      callOp.getResultTypes(), newArgs);

  return mlir::success();
}

AlgorithmOp getFirstAlgorithmOp(FunctionOp functionOp) {
  for (auto &op : functionOp.getOps()) {
    if (auto algorithmOp = mlir::dyn_cast<AlgorithmOp>(op)) {
      return algorithmOp;
    }
  }

  return nullptr;
}
} // namespace

void FunctionDefaultValuesConversionPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::SymbolTableCollection symbolTableCollection;
  mlir::LockedSymbolTableCollection lockedSymbolTables(symbolTableCollection);

  // Collect the functions.
  llvm::SmallVector<FunctionOp> functionOps;
  collectFunctionOps(moduleOp, functionOps);

  // Add the missing arguments to function calls.
  if (mlir::failed(
          convertInputDefaultValues(lockedSymbolTables, functionOps))) {
    return signalPassFailure();
  }

  // Copy the default assignments for output and protected variables to the
  // beginning of the function body.

  if (mlir::failed(mlir::failableParallelForEach(
          &getContext(), functionOps,
          [&](FunctionOp functionOp) -> mlir::LogicalResult {
            return convertProtectedAndOutputDefaultValues(lockedSymbolTables,
                                                          functionOp);
          }))) {
    return signalPassFailure();
  }

  // Erase the DefaultOps.
  eraseDefaultOps(functionOps);
}

mlir::LogicalResult
FunctionDefaultValuesConversionPass::convertInputDefaultValues(
    mlir::LockedSymbolTableCollection &symbolTableCollection,
    llvm::ArrayRef<FunctionOp> functionOps) {
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

    orderings.set(functionOp, defaultValuesGraph.postOrder());
  }

  // Fill the calls with missing arguments.
  llvm::MapVector<mlir::Block *, llvm::SmallVector<CallOp>> callOps;

  moduleOp.walk([&](CallOp callOp) {
    if (hasMissingArgs(moduleOp, symbolTableCollection, callOp)) {
      callOps[callOp->getBlock()].push_back(callOp);
    }
  });

  return mlir::failableParallelForEach(
      &getContext(), callOps, [&](const auto &blockEntry) {
        for (CallOp callOp : blockEntry.second) {
          if (mlir::failed(fillCallArgs(moduleOp, symbolTableCollection, callOp,
                                        orderings))) {
            return mlir::failure();
          }
        }

        return mlir::success();
      });
}

mlir::LogicalResult
FunctionDefaultValuesConversionPass::convertProtectedAndOutputDefaultValues(
    mlir::SymbolTableCollection &symbolTableCollection, FunctionOp functionOp) {
  mlir::IRRewriter rewriter(&getContext());

  // Collect the operations computing the default values and order them so
  // that dependencies are respected.
  llvm::StringMap<DefaultOp> defaultOps;

  DefaultValuesDependencyGraph defaultValuesGraph(defaultOps);

  for (DefaultOp defaultOp : functionOp.getOps<DefaultOp>()) {
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

  for (VariableOp variableOp : defaultValuesGraph.postOrder()) {
    auto defaultOpIt = defaultOps.find(variableOp.getSymName());

    if (defaultOpIt != defaultOps.end()) {
      DefaultOp defaultOp = defaultOpIt->getValue();
      mlir::IRMapping mapping;

      for (auto &nestedOp : defaultOp.getOps()) {
        if (auto yieldOp = mlir::dyn_cast<YieldOp>(nestedOp)) {
          assert(yieldOp.getValues().size() == 1);
          mlir::Value yieldedValue = yieldOp.getValues()[0];

          rewriter.create<VariableSetOp>(defaultOp.getLoc(), variableOp,
                                         mapping.lookup(yieldedValue));
        } else {
          rewriter.clone(nestedOp, mapping);
        }
      }
    }
  }

  return mlir::success();
}

void FunctionDefaultValuesConversionPass::eraseDefaultOps(
    llvm::ArrayRef<FunctionOp> functionOps) {
  mlir::parallelForEach(&getContext(), functionOps, [&](FunctionOp functionOp) {
    for (DefaultOp nestedOp :
         llvm::make_early_inc_range(functionOp.getOps<DefaultOp>())) {
      nestedOp.erase();
    }
  });
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createFunctionDefaultValuesConversionPass() {
  return std::make_unique<FunctionDefaultValuesConversionPass>();
}
} // namespace mlir::bmodelica
