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

ExternalFunctionCallLowerer::ExternalFunctionCallLowerer(BridgeInterface *bridge) : Lowerer(bridge) {}
  
  void ExternalFunctionCallLowerer::getFunctionResultTypes(
    mlir::Operation *op, llvm::SmallVectorImpl<mlir::Type> &types) {
    assert((mlir::isa<FunctionOp>(op)));

    if (auto functionOp = mlir::dyn_cast<FunctionOp>(op)) {
      mlir::FunctionType functionType = functionOp.getFunctionType();
      auto resultTypes = functionType.getResults();
      types.append(resultTypes.begin(), resultTypes.end());
      return;
    }
  }

  void ExternalFunctionCallLowerer::getFunctionExpectedArgRanks(
      mlir::Operation *op, llvm::SmallVectorImpl<int64_t> &ranks) {
    assert((mlir::isa<FunctionOp, DerFunctionOp>(op)));

    if (auto functionOp = mlir::dyn_cast<FunctionOp>(op)) {
      mlir::FunctionType functionType = functionOp.getFunctionType();

      for (mlir::Type type : functionType.getInputs()) {
        if (auto shapedType = mlir::dyn_cast<mlir::ShapedType>(type)) {
          ranks.push_back(shapedType.getRank());
        } else {
          ranks.push_back(0);
        }
      }

      return;
    }
  }
  bool ExternalFunctionCallLowerer::lowerCustomFunctionArgs(
      const ast::Call &call, llvm::ArrayRef<VariableOp> calleeInputs,
      llvm::SmallVectorImpl<std::string> &argNames,
      llvm::SmallVectorImpl<mlir::Value> &argValues) {
    size_t numOfArgs = call.getNumOfArguments();

    if (numOfArgs != 0) {
      if (call.getArgument(0)->dyn_cast<ast::ReductionFunctionArgument>()) {
        assert(call.getNumOfArguments() == 1);
        llvm_unreachable("ReductionOp has not been implemented in external function calls yet");
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
    }

    return true;
  }
  void ExternalFunctionCallLowerer::getCustomFunctionInputVariables(
      llvm::SmallVectorImpl<mlir::bmodelica::VariableOp> &inputVariables,
      FunctionOp functionOp) {
    for (VariableOp variableOp : functionOp.getVariables()) {
      if (variableOp.isInput()) {
        inputVariables.push_back(variableOp);
      }
    }
  }
  void ExternalFunctionCallLowerer::emitErrorNumArguments(llvm::StringRef function,
                                          const marco::SourceRange &location,
                                          unsigned int actualNum,
                                          unsigned int expectedNum) {
    std::string errorString =
        function.str() + ": expected " + std::to_string(expectedNum) +
        " argument(s) but got " + std::to_string(actualNum) + ".";
    mlir::emitError(loc(location)) << errorString;
  }

  std::optional<Results> ExternalFunctionCallLowerer::lower(const ast::ExternalFunctionCall &call) {
      
      mlir::Operation *result = resolveSymbolName(call.getFatherName(), getLookupScope());

      result = getSymbolTable().lookupSymbolIn(result, builder().getStringAttr(call.getFatherName()));

      llvm::SmallVector<VariableOp> inputVariables;

      if (auto functionOp = mlir::dyn_cast<FunctionOp>(*result)) {
        getCustomFunctionInputVariables(inputVariables, functionOp);
      }

      llvm::SmallVector<std::string, 3> argNames;
      llvm::SmallVector<mlir::Value, 3> argValues;

      if (!lowerCustomFunctionArgs(call, inputVariables, argNames, argValues)) {
        return std::nullopt;
      }

      llvm::SmallVector<int64_t, 3> expectedArgRanks;
      getFunctionExpectedArgRanks(*calleeOp, expectedArgRanks);

      llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
      getFunctionResultTypes(*calleeOp, scalarizedResultTypes);

      llvm::SmallVector<mlir::Type, 1> resultTypes;

      if (argValues.size() != expectedArgRanks.size()) {
        emitErrorNumArguments(call.getName(),
                              callee->getElement(0)->getLocation(),
                              argValues.size(), expectedArgRanks.size());
        return std::nullopt;
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
}

