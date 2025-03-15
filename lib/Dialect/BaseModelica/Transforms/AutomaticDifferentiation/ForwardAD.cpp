#include "marco/Dialect/BaseModelica/Transforms/AutomaticDifferentiation/ForwardAD.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/AutomaticDifferentiation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"

using namespace ::marco;
using namespace ::mlir::bmodelica;
using namespace ::mlir::bmodelica::ad::forward;

namespace {
template <typename T>
unsigned int numDigits(T number) {
  unsigned int digits = 0;

  while (number != 0) {
    number /= 10;
    ++digits;
  }

  return digits;
}
} // namespace

namespace mlir::bmodelica::ad::forward {
std::string getPartialDerFunctionName(llvm::StringRef baseName) {
  return "pder_" + baseName.str();
}

std::string getPartialDerVariableName(llvm::StringRef baseName) {
  return "pder_" + baseName.str();
}

std::optional<FunctionOp>
createFunctionPartialDerivative(mlir::OpBuilder &builder, State &state,
                                FunctionOp functionOp,
                                llvm::StringRef derivativeName) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(functionOp);

  // Create the derived function.
  auto derivedFunctionOp =
      builder.create<FunctionOp>(functionOp.getLoc(), derivativeName);

  // Add the function to the symbol table.
  mlir::Operation *parentSymbolTable =
      derivedFunctionOp->getParentWithTrait<mlir::OpTrait::SymbolTable>();

  state.getSymbolTableCollection()
      .getSymbolTable(parentSymbolTable)
      .insert(derivedFunctionOp);

  // Create the function body.
  mlir::Block *derivedFunctionBodyBlock =
      builder.createBlock(&derivedFunctionOp.getBodyRegion());

  builder.setInsertionPointToStart(derivedFunctionBodyBlock);
  mlir::IRMapping mapping;

  // Clone the variables.
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> originalVarsMapping;
  size_t variablesCounter = 0;

  for (VariableOp variableOp : functionOp.getVariables()) {
    auto clonedVariableOp =
        mlir::cast<VariableOp>(builder.clone(*variableOp.getOperation()));

    if (clonedVariableOp.isOutput()) {
      // Convert the output variables to protected ones.
      clonedVariableOp.setType(
          clonedVariableOp.getVariableType().withIOProperty(IOProperty::none));
    }

    originalVarsMapping[variableOp] = clonedVariableOp;
    ++variablesCounter;
  }

  // Create the derivatives.
  for (VariableOp variableOp : functionOp.getVariables()) {
    auto clonedVariableOp =
        mlir::cast<VariableOp>(builder.clone(*variableOp.getOperation()));

    std::string variableDerivativeName =
        getPartialDerVariableName(variableOp.getSymName()) + "_" +
        std::to_string(variablesCounter++);

    clonedVariableOp.setSymName(variableDerivativeName);

    state.mapGenericOpDerivative(originalVarsMapping[variableOp],
                                 clonedVariableOp);
  }

  // Clone the rest of the function body.
  for (auto &nestedOp : functionOp.getOps()) {
    if (mlir::isa<VariableOp>(nestedOp)) {
      // Variables have already been handled.
      continue;
    }

    mlir::Operation *clonedOp = builder.clone(nestedOp, mapping);

    if (auto derivableOp = mlir::dyn_cast<DerivableOpInterface>(clonedOp)) {
      if (mlir::failed(derivableOp.createPartialDerivative(builder, state))) {
        return std::nullopt;
      }
    }
  }

  return derivedFunctionOp;
}

std::string getTimeDerFunctionName(llvm::StringRef baseName) {
  return "timeder_" + baseName.str();
}

std::string getTimeDerVariableName(llvm::StringRef baseName, uint64_t order) {
  assert(order > 0);

  if (order == 1) {
    return "der_" + baseName.str();
  }

  return "der_" + std::to_string(order) + "_" + baseName.str();
}

std::string getNextTimeDerVariableName(llvm::StringRef currentName,
                                       uint64_t requestedOrder) {
  if (requestedOrder == 1) {
    return getTimeDerVariableName(currentName, requestedOrder);
  }

  assert(currentName.rfind("der_") == 0);

  if (requestedOrder == 2) {
    return getTimeDerVariableName(currentName.substr(4), requestedOrder);
  }

  return getTimeDerVariableName(
      currentName.substr(5 + numDigits(requestedOrder - 1)), requestedOrder);
}

bool isTimeDerivative(llvm::StringRef name, FunctionOp functionOp,
                      uint64_t maxOrder) {
  if (maxOrder == 0) {
    return false;
  }

  // If the current argument name starts with der, we need to check if the
  // original function to be derived has a variable whose derivative may be the
  // current one. If this is the case, then we don't need to add the n-th
  // derivative as it is already done when encountering that variable. If it is
  // not, then it means the original function had a "strange" variable named
  // "der_something" and the derivative function will contain both
  // "der_something" and "der_der_something"; the original "der_something"
  // could effectively be a derivative, but this is an assumption we can't
  // make.

  if (name.starts_with_insensitive("der_")) {
    for (VariableOp variableOp : functionOp.getVariables()) {
      for (unsigned int i = 1; i <= maxOrder; ++i) {
        if (name == getTimeDerVariableName(variableOp.getSymName(), i)) {
          return true;
        }
      }
    }
  }

  return false;
}

void mapTimeDerivativeFunctionVariables(FunctionOp functionOp,
                                        ad::forward::State &state) {
  mlir::SymbolTable &symbolTable =
      state.getSymbolTableCollection().getSymbolTable(functionOp);

  for (VariableOp variableOp : functionOp.getOps<VariableOp>()) {
    // Given a variable "x", first search for "der_x". If it doesn't exist,
    // then also "der_2_x", "der_3_x", etc. will not exist, and thus we can
    // say that "x" has no derivatives. If it exists, add the first order
    // derivative and then search for the higher order ones.

    std::string candidateFirstOrderDer =
        getTimeDerVariableName(variableOp.getSymName(), 1);

    auto derivativeVariableOp =
        symbolTable.lookup<VariableOp>(candidateFirstOrderDer);

    if (!derivativeVariableOp) {
      continue;
    }

    state.mapGenericOpDerivative(variableOp, derivativeVariableOp);

    uint64_t order = 2;
    bool found;

    do {
      std::string nextName =
          getTimeDerVariableName(variableOp.getSymName(), order);

      auto nextDerivativeVariableOp = symbolTable.lookup<VariableOp>(nextName);

      found = nextDerivativeVariableOp != nullptr;

      if (found) {
        state.mapGenericOpDerivative(derivativeVariableOp,
                                     nextDerivativeVariableOp);

        derivativeVariableOp = nextDerivativeVariableOp;
        ++order;
      }
    } while (found);
  }
}

std::optional<FunctionOp>
createFunctionTimeDerivative(mlir::OpBuilder &builder, State &state,
                             FunctionOp functionOp, uint64_t functionOrder,
                             llvm::StringRef derivativeName,
                             uint64_t derivativeOrder) {
  // TODO support the generation of higher derivatives
  assert(derivativeOrder == functionOrder + 1);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(functionOp);

  // Map the existing derivative relationships among the variables.
  mapTimeDerivativeFunctionVariables(functionOp, state);

  // Create the derived function.
  auto derivedFunctionOp =
      builder.create<FunctionOp>(functionOp.getLoc(), derivativeName);

  derivedFunctionOp.setTimeDerivativeOrder(derivativeOrder);

  mlir::Operation *functionParentSymbolTable =
      derivedFunctionOp->getParentWithTrait<mlir::OpTrait::SymbolTable>();

  state.getSymbolTableCollection()
      .getSymbolTable(functionParentSymbolTable)
      .insert(derivedFunctionOp);

  // Create the function body.
  mlir::Block *derivedFunctionBodyBlock =
      builder.createBlock(&derivedFunctionOp.getBodyRegion());

  builder.setInsertionPointToStart(derivedFunctionBodyBlock);
  mlir::IRMapping mapping;

  // Clone the original variables, with the output ones being converted to
  // protected ones. At the same time, determine the names and the types of
  // the new variables.
  for (VariableOp variableOp : functionOp.getVariables()) {
    auto clonedVariableOp = mlir::cast<VariableOp>(
        builder.clone(*variableOp.getOperation(), mapping));

    state.getSymbolTableCollection()
        .getSymbolTable(derivedFunctionOp)
        .insert(clonedVariableOp);

    VariableType variableType = clonedVariableOp.getVariableType();

    if (variableType.isOutput()) {
      // Convert the output variables to protected ones.
      clonedVariableOp.setType(variableType.withIOProperty(IOProperty::none));
    }
  }

  // Determine the new variables of the derived function.
  llvm::SmallVector<std::string> newInputVariableNames;
  llvm::SmallVector<VariableType> newInputVariableTypes;

  llvm::SmallVector<std::string> newOutputVariableNames;
  llvm::SmallVector<VariableType> newOutputVariableTypes;

  llvm::SmallVector<std::string> newProtectedVariableNames;
  llvm::SmallVector<VariableType> newProtectedVariableTypes;

  llvm::StringMap<llvm::StringRef> inverseDerivativesNamesMap;

  for (VariableOp variableOp : functionOp.getVariables()) {
    llvm::StringRef name = variableOp.getSymName();
    VariableType variableType = variableOp.getVariableType();

    if (variableOp.isInput()) {
      if (mlir::isa<RealType>(variableType.getElementType())) {
        if (isTimeDerivative(name, functionOp, functionOrder)) {
          continue;
        }

        std::string derName = getTimeDerVariableName(name, derivativeOrder);
        newInputVariableNames.push_back(derName);
        newInputVariableTypes.push_back(variableType);
        inverseDerivativesNamesMap[derName] = name;
      }
    } else if (variableOp.isOutput()) {
      if (mlir::isa<RealType>(variableType.getElementType())) {
        auto derName = getNextTimeDerVariableName(name, derivativeOrder);
        newOutputVariableNames.push_back(derName);
        newOutputVariableTypes.push_back(variableType);
        inverseDerivativesNamesMap[derName] = name;
      }
    } else {
      if (state.hasGenericOpDerivative(variableOp)) {
        // Avoid duplicates of original output variables, which have become
        // protected variables in the previous derivative functions.
        continue;
      }

      if (isTimeDerivative(name, functionOp, functionOrder)) {
        continue;
      }

      auto derName = getTimeDerVariableName(name, derivativeOrder);
      newProtectedVariableNames.push_back(derName);
      newProtectedVariableTypes.push_back(variableType);
    }
  }

  // Create the derivative variables.
  auto createDerVarsFn = [&](llvm::ArrayRef<std::string> derNames,
                             llvm::ArrayRef<VariableType> derTypes) {
    for (const auto &[name, type] : llvm::zip(derNames, derTypes)) {
      auto baseVariableName = inverseDerivativesNamesMap[name];

      auto baseVariable =
          state.getSymbolTableCollection().lookupSymbolIn<VariableOp>(
              derivedFunctionOp, builder.getStringAttr(baseVariableName));

      auto clonedOp = mlir::cast<VariableOp>(
          builder.clone(*baseVariable.getOperation(), mapping));

      clonedOp.setSymName(name);
      clonedOp.setType(type);

      state.getSymbolTableCollection()
          .getSymbolTable(derivedFunctionOp)
          .insert(clonedOp.getOperation());

      state.mapGenericOpDerivative(baseVariable, clonedOp);
    }
  };

  createDerVarsFn(newInputVariableNames, newInputVariableTypes);
  createDerVarsFn(newOutputVariableNames, newOutputVariableTypes);
  createDerVarsFn(newProtectedVariableNames, newProtectedVariableTypes);

  mapTimeDerivativeFunctionVariables(derivedFunctionOp, state);

  // Clone the rest of the function body.
  for (auto &nestedOp : functionOp.getOps()) {
    if (mlir::isa<VariableOp>(nestedOp)) {
      // Variables have already been handled.
      continue;
    }

    mlir::Operation *clonedOp = builder.clone(nestedOp, mapping);

    if (auto derivableOp = mlir::dyn_cast<DerivableOpInterface>(clonedOp)) {
      if (mlir::failed(
              derivableOp.createTimeDerivative(builder, state, false))) {
        return std::nullopt;
      }
    }
  }

  return derivedFunctionOp;
}
} // namespace mlir::bmodelica::ad::forward
