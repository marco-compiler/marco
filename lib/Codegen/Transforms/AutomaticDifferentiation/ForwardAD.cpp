#include "marco/Codegen/Transforms/AutomaticDifferentiation/ForwardAD.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation/Common.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation.h"
#include "marco/Dialect/BaseModelica/ModelicaDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"
#include <queue>

using namespace ::marco;
using namespace ::mlir::bmodelica;

static std::string getPartialDerFunctionName(llvm::StringRef baseName)
{
  return "pder_" + baseName.str();
}
  
static std::string getPartialDerVariableName(
    llvm::StringRef baseName, unsigned int order)
{
  return "pder_" + std::to_string(order) + "_" + baseName.str();
}

static bool isFullDerivative(
    llvm::StringRef name,
    FunctionOp functionOp,
    unsigned int maxOrder)
{
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
        if (name == getFullDerVariableName(variableOp.getSymName(), i)) {
          return true;
        }
      }
    }
  }

  return false;
}

namespace mlir::bmodelica
{
  bool ForwardAD::isDerived(mlir::Operation* op) const
  {
    return derivedOps.find(op) != derivedOps.end();
  }

  void ForwardAD::setAsDerived(mlir::Operation* op)
  {
    derivedOps.insert(op);
  }

  mlir::LogicalResult ForwardAD::createFullDerFunction(
      mlir::OpBuilder& builder,
      FunctionOp functionOp,
      mlir::SymbolTableCollection& symbolTable)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto derivativeAttribute =
        functionOp->getAttrOfType<DerivativeAttr>("derivative");

    unsigned int order = derivativeAttribute.getOrder();

    // Check if the derivative is already existing.
    mlir::Operation* symbolTableOp =
        functionOp->getParentWithTrait<mlir::OpTrait::SymbolTable>();

    if (auto derSymbol = symbolTable.lookupSymbolIn(
            symbolTableOp,
            builder.getStringAttr(derivativeAttribute.getName()))) {
      // If the source already provides a symbol with the derived function
      // name, then check that it is a function. If it is, then it means the
      // user manually provided the derivative.
      return mlir::LogicalResult::success(mlir::isa<FunctionOp>(derSymbol));
    }

    // Map the existing derivative relationships among the variables.
    llvm::DenseMap<mlir::StringAttr, mlir::StringAttr> variableDerivatives;
    mapFullDerivatives(functionOp, symbolTable, variableDerivatives);

    // Create the derived function.
    builder.setInsertionPointAfter(functionOp);

    auto derivedFunctionOp = builder.create<FunctionOp>(
        functionOp.getLoc(),
        derivativeAttribute.getName());

    builder.createBlock(&derivedFunctionOp.getBodyRegion());

    symbolTable.getSymbolTable(symbolTableOp).insert(
        derivedFunctionOp.getOperation());

    // Start the body of the function.
    builder.setInsertionPointToStart(derivedFunctionOp.getBody());
    mlir::IRMapping mapping;

    // Clone the original variables, with the output ones being converted to
    // protected ones. At the same time, determine the names and the types of
    // the new variables.
    for (VariableOp variableOp : functionOp.getVariables()) {
      auto clonedVariableOp = mlir::cast<VariableOp>(
          builder.clone(*variableOp.getOperation(), mapping));

      symbolTable.getSymbolTable(derivedFunctionOp.getOperation())
          .insert(clonedVariableOp);

      VariableType variableType = clonedVariableOp.getVariableType();

      if (variableType.isOutput()) {
        // Convert the output variables to protected ones.
        clonedVariableOp.setType(
            variableType.withIOProperty(IOProperty::none));
      }
    }

    // Determine the new variables of the derived function.
    // Notice the usage of std::string instead of llvm::StringRef, in order
    // to guarantee the storage for the names.

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
        if (variableType.getElementType().isa<RealType>()) {
          if (isFullDerivative(name, functionOp, order - 1)) {
            continue;
          }

          std::string derName = getFullDerVariableName(name, order);
          newInputVariableNames.push_back(derName);
          newInputVariableTypes.push_back(variableType);
          inverseDerivativesNamesMap[derName] = name;
        }
      } else if (variableOp.isOutput()) {
        if (variableType.getElementType().isa<RealType>()) {
          auto derName = getNextFullDerVariableName(name, order);
          newOutputVariableNames.push_back(derName);
          newOutputVariableTypes.push_back(variableType);
          inverseDerivativesNamesMap[derName] = name;
        }
      } else {
        if (variableDerivatives.find(variableOp.getSymNameAttr()) !=
            variableDerivatives.end()) {
          // Avoid duplicates of original output variables, which have become
          // protected variables in the previous derivative functions.
          continue;
        }

        if (isFullDerivative(name, functionOp, order - 1)) {
          continue;
        }

        auto derName = getFullDerVariableName(name, order);
        newProtectedVariableNames.push_back(derName);
        newProtectedVariableTypes.push_back(variableType);
      }
    }

    // Create the derivative variables.
    auto createDerVarsFn =
        [&](llvm::ArrayRef<std::string> derNames,
            llvm::ArrayRef<VariableType> derTypes) {
          for (const auto& [name, type] : llvm::zip(derNames, derTypes)) {
            auto baseVariableName = inverseDerivativesNamesMap[name];

            auto baseVariable = symbolTable.lookupSymbolIn<VariableOp>(
                derivedFunctionOp, builder.getStringAttr(baseVariableName));

            auto clonedOp = mlir::cast<VariableOp>(
                builder.clone(*baseVariable.getOperation(), mapping));

            clonedOp.setSymName(name);
            clonedOp.setType(type);

            symbolTable.getSymbolTable(derivedFunctionOp)
                .insert(clonedOp.getOperation());
          }
        };

    createDerVarsFn(newInputVariableNames, newInputVariableTypes);
    createDerVarsFn(newOutputVariableNames, newOutputVariableTypes);
    createDerVarsFn(newProtectedVariableNames, newProtectedVariableTypes);

    variableDerivatives.clear();

    mapFullDerivatives(derivedFunctionOp, symbolTable, variableDerivatives);

    // Clone the other operations. In the meanwhile, collect the algorithms
    // whose operations have to be derived.
    llvm::SmallVector<AlgorithmOp> algorithmOps;

    for (auto& baseOp : functionOp.getOps()) {
      if (mlir::isa<VariableOp>(baseOp)) {
        continue;
      }

      mlir::Operation* clonedOp = builder.clone(baseOp, mapping);

      if (auto algorithmOp = mlir::dyn_cast<AlgorithmOp>(clonedOp)) {
        algorithmOps.push_back(algorithmOp);
      }
    }

    // Compute the derivatives of the operations inside the algorithms.
    mlir::IRMapping ssaDerivatives;

    for (AlgorithmOp algorithmOp : algorithmOps) {
      auto deriveFn =
          [this](llvm::SmallVectorImpl<mlir::Value>& results,
                 mlir::OpBuilder& builder,
                 mlir::Operation* op,
                 mlir::SymbolTableCollection& symbolTable,
                 const llvm::DenseMap<
                     mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
                 mlir::IRMapping& ssaDerivatives)
          -> mlir::LogicalResult {
            return createOpFullDerivative(
                results, builder, op, symbolDerivatives, ssaDerivatives);
          };

      if (mlir::failed(deriveRegion(
              builder, algorithmOp.getBodyRegion(), symbolTable,
              variableDerivatives, ssaDerivatives, deriveFn))) {
        return mlir::failure();
      }
    }

    return mlir::success();
  }

  mlir::LogicalResult ForwardAD::convertPartialDerFunction(
      mlir::OpBuilder& builder,
      DerFunctionOp derFunctionOp,
      mlir::SymbolTableCollection& symbolTable)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto templateFunction = createPartialDerTemplateFunction(
        builder, symbolTable, derFunctionOp);

    if (templateFunction == nullptr) {
      // The template function could not be generated.
      return mlir::failure();
    }

    FunctionOp zeroOrderFunctionOp;
    unsigned int templateOrder;

    std::tie(zeroOrderFunctionOp, templateOrder) =
        getPartialDerBaseFunction(templateFunction);

    mlir::Location loc = derFunctionOp.getLoc();

    mlir::Operation* parentSymbolTable =
        derFunctionOp->getParentWithTrait<mlir::OpTrait::SymbolTable>();

    auto baseFunctionOp = symbolTable.lookupSymbolIn<FunctionOp>(
        parentSymbolTable,
        builder.getStringAttr(derFunctionOp.getDerivedFunction()));

    // Create the derived function.
    builder.setInsertionPointAfter(derFunctionOp);

    auto derivedFunctionOp = builder.create<FunctionOp>(
        derFunctionOp.getLoc(), derFunctionOp.getSymName());

    partialDersTemplateCallers[derivedFunctionOp.getSymName()] =
        templateFunction;

    builder.createBlock(&derivedFunctionOp.getBodyRegion());
    builder.setInsertionPointToStart(derivedFunctionOp.getBody());

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

    mlir::Block* algorithmBody =
        builder.createBlock(&algorithmOp.getBodyRegion());

    builder.setInsertionPointToStart(algorithmBody);

    // Call the template function.
    llvm::SmallVector<mlir::Value> args;

    for (VariableOp variableOp : inputVariables) {
      args.push_back(builder.create<VariableGetOp>(
          loc, variableOp.getVariableType().unwrap(), variableOp.getSymName()));
    }

    size_t zeroOrderArgsNumber = llvm::count_if(
        zeroOrderFunctionOp.getVariables(), [](VariableOp variableOp) {
          return variableOp.isInput();
        });

    std::vector<mlir::Attribute> allIndependentVars;

    if (auto previousTemplateIt =
            partialDerTemplates.find(templateFunction.getSymName());
        previousTemplateIt != partialDerTemplates.end()) {
      auto previousTemplateVarsIt = partialDerTemplatesIndependentVars.find(
          previousTemplateIt->second.getSymName());

      if (previousTemplateVarsIt != partialDerTemplatesIndependentVars.end()) {
        for (const auto& independentVar : previousTemplateVarsIt->second) {
          allIndependentVars.push_back(
              independentVar.cast<mlir::StringAttr>());
        }
      }
    }

    for (const auto& independentVariable :
         derFunctionOp.getIndependentVars()) {
      allIndependentVars.push_back(
          independentVariable.cast<mlir::StringAttr>());
    }

    partialDerTemplatesIndependentVars[templateFunction.getSymName()] =
        builder.getArrayAttr(allIndependentVars);

    assert(templateOrder == allIndependentVars.size());
    unsigned int numberOfSeeds = zeroOrderArgsNumber;

    llvm::SmallVector<llvm::StringRef> inputVariableNames;

    for (VariableOp variableOp : zeroOrderFunctionOp.getVariables()) {
      if (variableOp.isInput()) {
        inputVariableNames.push_back(variableOp.getSymName());
      }
    }

    llvm::SmallVector<mlir::Type> templateFunctionArgTypes =
        templateFunction.getArgumentTypes();

    for (const auto& independentVariable :
         llvm::enumerate(allIndependentVars)) {
      auto independentVarName =
          independentVariable.value().cast<mlir::StringAttr>().getValue();

      unsigned int variableIndex = zeroOrderArgsNumber;

      for (unsigned int i = 0; i < zeroOrderArgsNumber; ++i) {
        if (inputVariableNames[i] == independentVarName) {
          variableIndex = i;
          break;
        }
      }

      assert(variableIndex < zeroOrderArgsNumber);

      for (unsigned int i = 0; i < numberOfSeeds; ++i) {
        float seed = i == variableIndex ? 1 : 0;
        auto argType = templateFunctionArgTypes[i];
        assert(!(seed == 1 && argType.isa<ArrayType>()));

        if (auto arrayType = argType.dyn_cast<ArrayType>()) {
          // TODO dynamic sizes
          assert(arrayType.hasStaticShape());

          mlir::Value array = builder.create<AllocOp>(loc, arrayType, std::nullopt);
          args.push_back(array);
          mlir::Value seedValue = builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), seed));
          builder.create<ArrayFillOp>(loc, array, seedValue);
        } else {
          args.push_back(builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), seed)));
        }
      }

      numberOfSeeds *= 2;
    }

    auto callOp = builder.create<CallOp>(loc, templateFunction, args);
    assert(callOp->getNumResults() == outputVariables.size());

    for (const auto& [variable, result] :
         llvm::zip(outputVariables, callOp->getResults())) {
      builder.create<VariableSetOp>(loc, variable.getSymName(), result);
    }

    symbolTable.getSymbolTable(parentSymbolTable).erase(derFunctionOp);
    symbolTable.getSymbolTable(parentSymbolTable).insert(derivedFunctionOp);

    return mlir::success();
  }

  FunctionOp ForwardAD::createPartialDerTemplateFunction(
      mlir::OpBuilder& builder,
      mlir::SymbolTableCollection& symbolTable,
      DerFunctionOp derFunctionOp)
  {
    mlir::Operation* parentSymbolTable =
        derFunctionOp->getParentWithTrait<mlir::OpTrait::SymbolTable>();

    auto functionOp = symbolTable.lookupSymbolIn<FunctionOp>(
        parentSymbolTable,
        builder.getStringAttr(derFunctionOp.getDerivedFunction()));

    if (auto it = partialDersTemplateCallers.find(functionOp.getSymName());
        it != partialDersTemplateCallers.end()) {
      functionOp = it->second;
    }

    std::string derivedFunctionName =
        getPartialDerFunctionName(derFunctionOp.getSymName());

    for (size_t i = 0; i < derFunctionOp.getIndependentVars().size(); ++i) {
      auto derTemplate = createPartialDerTemplateFunction(
          builder, derFunctionOp.getLoc(), symbolTable,
          functionOp, derivedFunctionName);

      if (derTemplate == nullptr) {
        return nullptr;
      }

      symbolTable.getSymbolTable(parentSymbolTable)
          .insert(derTemplate.getOperation());

      partialDerTemplates[derTemplate.getSymName()] = functionOp;
      functionOp = derTemplate;
      derivedFunctionName = getPartialDerFunctionName(functionOp.getSymName());
    }

    return functionOp;
  }

  FunctionOp ForwardAD::createPartialDerTemplateFunction(
      mlir::OpBuilder& builder,
      mlir::Location loc,
      mlir::SymbolTableCollection& symbolTable,
      FunctionOp functionOp,
      llvm::StringRef derivedFunctionName)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(functionOp);

    // Determine the characteristics of the base function.
    FunctionOp baseFunctionOp;
    unsigned int currentOrder;

    std::tie(baseFunctionOp, currentOrder) =
        getPartialDerBaseFunction(functionOp);

    // Create the derived function.
    auto derivedFunctionOp = builder.create<FunctionOp>(
        functionOp.getLoc(), derivedFunctionName);

    builder.createBlock(&derivedFunctionOp.getBodyRegion());

    mlir::Operation* parentSymbolTable =
        derivedFunctionOp->getParentWithTrait<mlir::OpTrait::SymbolTable>();

    symbolTable.getSymbolTable(parentSymbolTable).insert(derivedFunctionOp);

    // Start the body of the function.
    builder.setInsertionPointToStart(derivedFunctionOp.getBody());
    mlir::IRMapping mapping;

    // Clone the variables.
    size_t variablesCounter = 0;

    for (VariableOp variableOp : functionOp.getVariables()) {
      auto clonedVariableOp = mlir::cast<VariableOp>(
          builder.clone(*variableOp.getOperation()));

      if (clonedVariableOp.isOutput()) {
        // Convert the output variables to protected ones.
        clonedVariableOp.setType(
            clonedVariableOp.getVariableType()
                .withIOProperty(IOProperty::none));
      }

      variablesCounter++;
    }

    // Create the derivatives.
    llvm::DenseMap<mlir::StringAttr, mlir::StringAttr> varDerMap;

    for (VariableOp variableOp : functionOp.getVariables()) {
      auto clonedVariableOp = mlir::cast<VariableOp>(
          builder.clone(*variableOp.getOperation()));

      std::string derivativeName =
          getPartialDerVariableName(variableOp.getSymName(), currentOrder + 1)
          + "_" + std::to_string(variablesCounter++);

      clonedVariableOp.setSymName(derivativeName);

      varDerMap[variableOp.getSymNameAttr()] =
          clonedVariableOp.getSymNameAttr();
    }

    // Clone the rest of the function body.
    mlir::IRMapping ssaDerMap;

    for (auto& op : functionOp.getOps()) {
      if (mlir::isa<VariableOp>(op)) {
        // Variables have already been handled.
        continue;
      }

      mlir::Operation* clonedOp = builder.clone(op, mapping);

      if (auto algorithmOp = mlir::dyn_cast<AlgorithmOp>(clonedOp)) {
        auto deriveFn =
            [this](llvm::SmallVectorImpl<mlir::Value>& results,
                mlir::OpBuilder& nestedBuilder,
                mlir::Operation* op,
                mlir::SymbolTableCollection& symbolTable,
                const llvm::DenseMap<
                    mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
                mlir::IRMapping& ssaDerivatives) {
              return createOpPartialDerivative(
                  results, nestedBuilder, op, symbolTable, symbolDerivatives,
                  ssaDerivatives);
        };

        if (mlir::failed(deriveRegion(
                builder, algorithmOp.getBodyRegion(),
                symbolTable,
                varDerMap, ssaDerMap,
                deriveFn))) {
          return nullptr;
        }
      }
    }

    return derivedFunctionOp;
  }

  std::pair<FunctionOp, unsigned int>
  ForwardAD::getPartialDerBaseFunction(FunctionOp functionOp)
  {
    unsigned int order = 0;
    FunctionOp baseFunction = functionOp;

    while (partialDerTemplates.find(baseFunction.getSymName()) !=
           partialDerTemplates.end()) {
      ++order;
      baseFunction = partialDerTemplates.lookup(baseFunction.getSymName());
    }

    return std::make_pair(baseFunction, order);
  }

  mlir::LogicalResult ForwardAD::deriveRegion(
      mlir::OpBuilder& builder,
      mlir::Region& region,
      mlir::SymbolTableCollection& symbolTable,
      llvm::DenseMap<mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& ssaDerivatives,
      llvm::function_ref<mlir::LogicalResult(
          llvm::SmallVectorImpl<mlir::Value>& results,
          mlir::OpBuilder&,
          mlir::Operation*,
          mlir::SymbolTableCollection&,
          const llvm::DenseMap<mlir::StringAttr, mlir::StringAttr>&,
          mlir::IRMapping&)> deriveFn)
  {
    // Determine the list of the derivable operations. We can't just derive as
    // we find them, as we would invalidate the operation walk's iterator.
    std::queue<mlir::Operation*> ops;

    for (auto& op : region.getOps()) {
      ops.push(&op);
    }

    while (!ops.empty()) {
      auto op = ops.front();
      ops.pop();

      builder.setInsertionPointAfter(op);

      if (isDerivable(op) && !isDerived(op)) {
        llvm::SmallVector<mlir::Value> derivedValues;

        if (mlir::failed(deriveFn(
                derivedValues, builder, op, symbolTable, symbolDerivatives,
                ssaDerivatives))) {
          return mlir::failure();
        }

        assert(op->getNumResults() == derivedValues.size());
        setAsDerived(op);

        if (!derivedValues.empty()) {
          for (const auto& [base, derived] :
               llvm::zip(op->getResults(), derivedValues)) {
            ssaDerivatives.map(base, derived);
          }
        }
      }

      // Derive the regions of the operation.
      if (auto derivableOp = mlir::dyn_cast<DerivableOpInterface>(op)) {
        llvm::SmallVector<mlir::Region*, 2> regions;
        derivableOp.getDerivableRegions(regions);

        for (auto& nestedRegion : regions) {
          for (auto& childOp : nestedRegion->getOps()) {
            ops.push(&childOp);
          }
        }
      }
    }

    return mlir::success();
  }

  bool ForwardAD::isDerivable(mlir::Operation* op) const
  {
    return mlir::isa<
        VariableGetOp,
        VariableSetOp,
        CallOp,
        TimeOp,
        DerivableOpInterface>(op);
  }

  mlir::LogicalResult ForwardAD::createOpFullDerivative(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Operation* op,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& ssaDerivatives)
  {
    if (auto callOp = mlir::dyn_cast<CallOp>(op)) {
      return createCallOpFullDerivative(
          results, builder, callOp, ssaDerivatives);
    }

    if (auto timeOp = mlir::dyn_cast<TimeOp>(op)) {
      return createTimeOpFullDerivative(results, builder, timeOp);
    }

    if (auto derivableOp = mlir::dyn_cast<DerivableOpInterface>(op)) {
      return derivableOp.derive(
          results, builder, symbolDerivatives, ssaDerivatives);
    }

    llvm_unreachable("Can't derive a non-derivable operation");
    return mlir::failure();
  }

  mlir::LogicalResult ForwardAD::createOpPartialDerivative(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Operation* op,
      mlir::SymbolTableCollection& symbolTable,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& ssaDerivatives)
  {
    if (auto callOp = mlir::dyn_cast<CallOp>(op)) {
      return createCallOpPartialDerivative(
          results, builder, callOp, symbolTable, ssaDerivatives);
    }

    if (auto timeOp = mlir::dyn_cast<TimeOp>(op)) {
      return createTimeOpPartialDerivative(results, builder, timeOp);
    }

    if (auto derivableOp = mlir::dyn_cast<DerivableOpInterface>(op)) {
      return derivableOp.derive(
          results, builder, symbolDerivatives, ssaDerivatives);
    }

    llvm_unreachable("Can't derive a non-derivable operation");
    return mlir::failure();
  }

  mlir::LogicalResult ForwardAD::createCallOpFullDerivative(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      CallOp callOp,
      mlir::IRMapping& ssaDerivatives)
  {
    llvm_unreachable("CallOp full derivative is not implemented");
    return mlir::failure();
  }

  mlir::LogicalResult ForwardAD::createCallOpPartialDerivative(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      CallOp callOp,
      mlir::SymbolTableCollection& symbolTable,
      mlir::IRMapping& ssaDerivatives)
  {
    mlir::Location loc = callOp.getLoc();
    auto moduleOp = callOp->getParentOfType<mlir::ModuleOp>();
    auto callee = moduleOp.lookupSymbol<FunctionOp>(callOp.getCallee());

    std::string derivedFunctionName =
        "call_pder_" + callOp.getCallee().getLeafReference().getValue().str();

    llvm::SmallVector<mlir::Value, 3> args;

    for (auto arg : callOp.getArgs()) {
      args.push_back(arg);
    }

    for (auto arg : callOp.getArgs()) {
      args.push_back(ssaDerivatives.lookup(arg));
    }

    if (auto derTemplate =
            moduleOp.lookupSymbol<FunctionOp>(derivedFunctionName)) {
      auto derivedOp = builder.create<CallOp>(loc, derTemplate, args);

      for (mlir::Value result : derivedOp->getResults()) {
        results.push_back(result);
      }

      return mlir::success();
    }

    auto derTemplate = createPartialDerTemplateFunction(
        builder, loc, symbolTable, callee, derivedFunctionName);

    auto derivedOp = builder.create<CallOp>(loc, derTemplate, args);

    for (mlir::Value result : derivedOp->getResults()) {
      results.push_back(result);
    }

    return mlir::success();
  }

  mlir::LogicalResult ForwardAD::createTimeOpFullDerivative(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::bmodelica::TimeOp timeOp)
  {
    auto derivedOp = builder.create<ConstantOp>(
        timeOp.getLoc(), RealAttr::get(timeOp.getContext(), 1));

    results.push_back(derivedOp);
    return mlir::success();
  }

  mlir::LogicalResult ForwardAD::createTimeOpPartialDerivative(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::bmodelica::TimeOp timeOp)
  {
    auto derivedOp = builder.create<ConstantOp>(
        timeOp.getLoc(), RealAttr::get(timeOp.getContext(), 0));

    results.push_back(derivedOp);
    return mlir::success();
  }

  mlir::LogicalResult ForwardAD::deriveTree(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      DerivableOpInterface op,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& ssaDerivatives)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(op.getOperation());

    llvm::SmallVector<mlir::Value, 3> toBeDerived;
    op.getOperandsToBeDerived(toBeDerived);

    for (mlir::Value operand : toBeDerived) {
      if (!ssaDerivatives.contains(operand)) {
        mlir::Operation* definingOp = operand.getDefiningOp();

        if (definingOp == nullptr) {
          return mlir::failure();
        }

        if (!mlir::isa<DerivableOpInterface>(definingOp)) {
          return mlir::failure();
        }
      }
    }

    for (mlir::Value operand : toBeDerived) {
      mlir::Operation* definingOp = operand.getDefiningOp();

      if (definingOp == nullptr) {
        continue;
      }

      if (auto derivableOp =
              mlir::dyn_cast<DerivableOpInterface>(definingOp)) {
        llvm::SmallVector<mlir::Value> derivedValues;

        if (mlir::failed(deriveTree(
                derivedValues, builder, derivableOp, symbolDerivatives,
                ssaDerivatives))) {
          return mlir::failure();
        }

        if (derivedValues.size() != derivableOp->getNumResults()) {
          return mlir::failure();
        }

        for (const auto& [base, derived] : llvm::zip(
                 derivableOp->getResults(), derivedValues)) {
          ssaDerivatives.map(base, derived);
        }
      }
    }

    return op.derive(results, builder, symbolDerivatives, ssaDerivatives);
  }
}
