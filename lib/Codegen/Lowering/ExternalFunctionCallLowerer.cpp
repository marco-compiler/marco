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
    assert((mlir::isa<FunctionOp>(op)));

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
  std::optional<mlir::Value> ExternalFunctionCallLowerer::lowerArg(const ast::Expression &expression) {
    mlir::Location location = loc(expression.getLocation());
    auto loweredExpression = lower(expression);
    if (!loweredExpression) {
      return std::nullopt;
    }
    auto &results = *loweredExpression;
    assert(results.size() == 1);
    return results[0].get(location);
  }

  bool ExternalFunctionCallLowerer::lowerCustomFunctionArgs(
      const ast::ExternalFunctionCall &call, llvm::ArrayRef<VariableOp> calleeInputs,
      llvm::SmallVectorImpl<std::string> &argNames,
      llvm::SmallVectorImpl<mlir::Value> &argValues) {


    auto args = call.getExpressions();

    // Process the unnamed arguments.
    for (size_t i = 0 ; i < args.size() ; i++){
        
      auto argValue = lowerArg(*args[i]->cast<ast::Expression>());

      if (!argValue) {
        return false;
      }

      argValues.push_back(*argValue);
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

  bool ExternalFunctionCallLowerer::lower(const ast::ExternalFunctionCall &call, mlir::Operation * parentOp) {

    llvm::SmallVector<mlir::Type> inputTypes;
    llvm::SmallVector<mlir::Type> outputTypes;
    
    llvm::SmallVector<VariableOp> inputVariables;

    auto parentFuncOp = mlir::cast<FunctionOp>(*parentOp);
    for (VariableOp variable : parentFuncOp.getVariables()) {
      auto varType = mlir::cast<VariableType>(variable.getType()).getElementType();
      if (variable.isInput()) {
        inputTypes.push_back(varType);
        inputVariables.push_back(variable); 
      } else if (variable.isOutput()) {
        outputTypes.push_back(varType);
      }
    }

    auto funcType = mlir::FunctionType::get(builder().getContext(), inputTypes, outputTypes);
    auto funcTypeAttr = mlir::TypeAttr::get(funcType);

    {
      auto module = parentOp->getParentOfType<mlir::ModuleOp>();
      mlir::OpBuilder::InsertionGuard guard(builder());
      builder().setInsertionPointToStart(module.getBody());
      auto externalFunctionOp = builder().create<ExternalFunctionOp>(loc(call.getLocation()), call.getName(), funcTypeAttr);
      externalFunctionOp.setPrivate();
    }

    llvm::SmallVector<std::string> argNames;
    llvm::SmallVector<mlir::Value> argValues;

    if (!lowerCustomFunctionArgs(call, inputVariables, argNames, argValues)) {
      return false;
    }

    auto callOp = builder().create<CallOp>(loc(call.getLocation()),
                                           externalFunctionOp,
                                           argValues);   
    return true;

    /*VariableOp app;

      auto fo = mlir::cast<FunctionOp>(*parentOp);

      for (VariableOp variable : fo.getVariables()) {
        if (variable.isOutput()) {
           app = std::move(variable);
        }
      }

      llvm::SmallVector<VariableOp> inputVariables;

     auto externalFunctionOp = builder().create<ExternalFunctionOp>(loc(call.getLocation()), call.getName(), app.getTypeAttr());
      //auto callOp = builder().create<CallOp>(loc(call.getLocation()),functionOp);


      if (auto functionOp = mlir::dyn_cast<FunctionOp>(parentOp)) {
        getCustomFunctionInputVariables(inputVariables, functionOp);
      }

      llvm::SmallVector<std::string, 3> argNames;
      llvm::SmallVector<mlir::Value, 3> argValues;

      if (!lowerCustomFunctionArgs(call, inputVariables, argNames, argValues)) {
        return false;
      }

      llvm::SmallVector<int64_t, 3> expectedArgRanks;
      getFunctionExpectedArgRanks(parentOp, expectedArgRanks);

      llvm::SmallVector<mlir::Type, 1> scalarizedResultTypes;
      getFunctionResultTypes(parentOp, scalarizedResultTypes);

      if (argValues.size() != expectedArgRanks.size()) {
        emitErrorNumArguments(call.getName(),
                              call.getComponentReference()->cast<ast::ComponentReference>()->getElement(0)->getLocation(),
                              argValues.size(), expectedArgRanks.size());
        return false;
      }

      auto ref = mlir::SymbolRefAttr::get(builder().getContext(), call.getName());

      /*auto clonedFunc = builder().clone(*parentOp);

      clonedFunc->setAttr("llvm.linkage", builder().getStringAttr("external"));

      clonedFunc->setAttr(mlir::SymbolTable::getSymbolAttrName(),builder().getStringAttr(call.getName()));

      if (auto symbolOp = llvm::dyn_cast<mlir::SymbolOpInterface>(clonedFunc)) {
        symbolOp.setVisibility(mlir::SymbolTable::Visibility::Nested);
      }

      auto module = parentOp->getParentOfType<mlir::ModuleOp>();
      mlir::SymbolTable symbolTable(module);
      symbolTable.insert(mlir::cast<mlir::FunctionOpInterface>(clonedFunc));

      auto parentNameAttr = getSymbolRefFromRoot(parentOp);

      auto calleeNameAttr = mlir::StringAttr::get(builder().getContext(), call.getName());

      auto nestedRef = getSymbolRefFromRoot(clonedFunc);*/


    }
}
