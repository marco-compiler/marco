/**
 * Il nostro obbiettivo è scrivere un analogo di "lower", ovvero un metodo che porti a "creare" una chiamata.
 * Tuttavia, Call non è esattamente come la nostra external function call ma è molto probabile che la differenza
 * sia minima o quasi assente per quello che ci interessa. La mia idea è dunque ipotizzare che sia uguale e provare
 * a riscrivere i metodi che inserito qui (più altri, se fossero necessari). Verificare almeno che compili e poi
 * provare a vedere se funziona davvero.
 * 
 * In generale sembra che tutto il lowering di call si concluda in "auto callOp = builder().create<CallOp>(loc(call.getLocation()),
                                             getSymbolRefFromRoot(*calleeOp),
                                             resultTypes, argValues);", quindi anche questo è il nostro obbiettivo.

   Teniamo traccia delle differenze rispetto a call e cerchiamo di farla funzionare per i casi semplici inizialmente.
*/

#include "marco/Codegen/Lowering/ExternalFunctionCallLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering {

void CallLowerer::getFunctionResultTypes(
  mlir::Operation *op, llvm::SmallVectorImpl<mlir::Type> &types) {
    /*assert((mlir::isa<FunctionOp, DerFunctionOp>(op)));

    if (auto functionOp = mlir::dyn_cast<FunctionOp>(op)) {
      mlir::FunctionType functionType = functionOp.getFunctionType();
      auto resultTypes = functionType.getResults();
      types.append(resultTypes.begin(), resultTypes.end());
      return;
    }

  if (auto derFunctionOp = mlir::dyn_cast<DerFunctionOp>(op)) {
    auto moduleOp = derFunctionOp->getParentOfType<mlir::ModuleOp>();

    mlir::Operation *derivedFunctionOp = resolveSymbol(
        moduleOp, getSymbolTable(), derFunctionOp.getDerivedFunction());

    while (mlir::isa_and_nonnull<DerFunctionOp>(derivedFunctionOp)) {
      auto baseDerFunctionOp = mlir::cast<DerFunctionOp>(derivedFunctionOp);

      derivedFunctionOp = resolveSymbol(moduleOp, getSymbolTable(),
                                        baseDerFunctionOp.getDerivedFunction());
    }

    assert(derivedFunctionOp && "Derived function not found");
    auto functionOp = mlir::cast<FunctionOp>(derivedFunctionOp);

    mlir::FunctionType functionType = functionOp.getFunctionType();
    auto resultTypes = functionType.getResults();
    types.append(resultTypes.begin(), resultTypes.end());

    return;
  }*/

  return;
}
  bool CallLowerer::lowerCustomFunctionArgs(
    const ast::Call &call, llvm::ArrayRef<VariableOp> calleeInputs,
    llvm::SmallVectorImpl<std::string> &argNames,
    llvm::SmallVectorImpl<mlir::Value> &argValues) {
    /*
    size_t numOfArgs = call.getNumOfArguments();

    if (numOfArgs != 0) {
      if (call.getArgument(0)->dyn_cast<ast::ReductionFunctionArgument>()) {
        assert(call.getNumOfArguments() == 1);
        llvm_unreachable("ReductionOp has not been implemented yet");
        return false;
      }
    }

    bool existsNamedArgument = false;

    for (size_t i = 0; i < numOfArgs && !existsNamedArgument; ++i) {
      if (call.getArgument(i)->isa<ast::NamedFunctionArgument>()) {
        existsNamedArgument = true;
      }
    }

    size_t argIndex = 0;

    // Process the unnamed arguments.
    while (argIndex < numOfArgs &&
           !call.getArgument(argIndex)->isa<ast::NamedFunctionArgument>()) {
      auto arg =
          call.getArgument(argIndex)->cast<ast::ExpressionFunctionArgument>();

      auto argValue = lowerArg(*arg->getExpression());
      if (!argValue) {
        return false;
      }
      argValues.push_back(*argValue);

      if (existsNamedArgument) {
        VariableOp variableOp = calleeInputs[argIndex];
        argNames.push_back(variableOp.getSymName().str());
      }

      ++argIndex;
    }

    // Process the named arguments.
    while (argIndex < numOfArgs) {
      auto arg = call.getArgument(argIndex)->cast<ast::NamedFunctionArgument>();

      auto argValue = lowerArg(*arg->getValue()
                                    ->cast<ast::ExpressionFunctionArgument>()
                                    ->getExpression());
      if (!argValue) {
        return false;
      }
      argValues.push_back(*argValue);

      argNames.push_back(arg->getName().str());
      ++argIndex;
     }*/

  return true;
}

  External_RefLowerer::External_RefLowerer(BridgeInterface *bridge) : Lowerer(bridge) {}

  void CallLowerer::getCustomFunctionInputVariables(
    llvm::SmallVectorImpl<mlir::bmodelica::VariableOp> &inputVariables,
    FunctionOp functionOp) { //dobbiamo trovare una maniera per avere anche noi gli input
  
    /*for (VariableOp variableOp : functionOp.getVariables()) {
      if (variableOp.isInput()) {
        inputVariables.push_back(variableOp);
      }
    }*/
    return (false);
  }

  std::optional<Results> CallLowerer::lower(const ast::Call &call) {
    /*
    const ast::ComponentReference *callee =
        call.getCallee()->cast<ast::ComponentReference>();

    std::optional<mlir::Operation *> calleeOp = resolveCallee(*callee);

    if (!calleeOp) {
      llvm_unreachable("Invalid callee");
      return {};
    }

    if (*calleeOp) {
      if (mlir::isa<FunctionOp, DerFunctionOp>(*calleeOp)) {
        // User-defined function.
        llvm::SmallVector<VariableOp> inputVariables;

        if (auto functionOp = mlir::dyn_cast<FunctionOp>(*calleeOp)) {
          getCustomFunctionInputVariables(inputVariables, functionOp);
        }

        //questa der la possiamo togliere
        if (auto derFunctionOp = mlir::dyn_cast<DerFunctionOp>(*calleeOp)) {
          getCustomFunctionInputVariables(inputVariables, derFunctionOp);
        }

        llvm::SmallVector<std::string, 3> argNames;
        llvm::SmallVector<mlir::Value, 3> argValues;

        if (!lowerCustomFunctionArgs(call, inputVariables, argNames, argValues)) {
          return std::nullopt;
        }
        assert(argNames.empty() && "Named arguments not supported yet");

        llvm::SmallVector<int64_t, 3> expectedArgRanks;
        getFunctionExpectedArgRanks(*calleeOp, expectedArgRanks);

        llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
        getFunctionResultTypes(*calleeOp, scalarizedResultTypes);

        llvm::SmallVector<mlir::Type, 1> resultTypes;

        if (argValues.size() != expectedArgRanks.size()) {
          emitErrorNumArguments(callee->getElement(0)->getName(),
                                callee->getElement(0)->getLocation(),
                                argValues.size(), expectedArgRanks.size());
          return std::nullopt;
        }

        if (!getVectorizedResultTypes(argValues, expectedArgRanks,
                                      scalarizedResultTypes, resultTypes)) {
          assert(false && "Can't vectorize function call");
          return {};
        }

        auto callOp = builder().create<CallOp>(loc(call.getLocation()),
                                               getSymbolRefFromRoot(*calleeOp),
                                               resultTypes, argValues);

        std::vector<Reference> results;

        for (auto result : callOp->getResults()) {
          results.push_back(Reference::ssa(builder(), result));
        }

        return Results(results.begin(), results.end());
      }

      // Check if it's an implicit record constructor.
      if (auto recordConstructor = mlir::dyn_cast<RecordOp>(*calleeOp)) {
        llvm::SmallVector<VariableOp> inputVariables;
        getRecordConstructorInputVariables(inputVariables, recordConstructor);

        llvm::SmallVector<std::string, 3> argNames;
        llvm::SmallVector<mlir::Value, 3> argValues;
        if (!lowerRecordConstructorArgs(call, inputVariables, argNames,
                                        argValues)) {
          return std::nullopt;
        }
        assert(argNames.empty() && "Named args for records not yet supported");

        mlir::SymbolRefAttr symbol = getSymbolRefFromRoot(recordConstructor);

        mlir::Value result = builder().create<RecordCreateOp>(
            loc(call.getLocation()),
            RecordType::get(builder().getContext(), symbol), argValues);

        return Reference::ssa(builder(), result);
      }
    }

    if (isBuiltInFunction(*callee)) {
      // Built-in function.
      return dispatchBuiltInFunctionCall(call);
    }

    // The function doesn't exist.
    std::set<std::string> visibleFunctions;
    getVisibleSymbols(getLookupScope(), visibleFunctions);

    emitIdentifierError(IdentifierError::IdentifierType::FUNCTION,
                        callee->getElement(0)->getName(), visibleFunctions,
                        callee->getElement(0)->getLocation());
                        */
    return std::nullopt;
}

std::optional<mlir::Operation *>
CallLowerer::resolveCallee(const ast::ComponentReference &callee) {
    /*
    size_t pathLength = callee.getPathLength();
    assert(callee.getPathLength() > 0);

    for (size_t i = 0; i < pathLength; ++i) {
      if (callee.getElement(i)->getNumOfSubscripts() != 0) {
        return std::nullopt;
      }
    }

    mlir::Operation *result =
        resolveSymbolName(callee.getElement(0)->getName(), getLookupScope());

    for (size_t i = 1; i < pathLength; ++i) {
      if (result == nullptr) {
        return nullptr;
      }

      result = getSymbolTable().lookupSymbolIn(
          result, builder().getStringAttr(callee.getElement(i)->getName()));
    }*/

    return result;
  }
}

