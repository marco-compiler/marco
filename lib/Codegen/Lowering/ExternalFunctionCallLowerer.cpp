#include "marco/Codegen/Lowering/ExternalFunctionCallLowerer.h"
#include "llvm/ADT/StringSwitch.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;


namespace marco::codegen::lowering {
ExternalFunctionCallLowerer::ExternalFunctionCallLowerer(BridgeInterface *bridge) : Lowerer(bridge) {}

bool ExternalFunctionCallLowerer::lower(const ast::ExternalFunctionCall &call) {

  mlir::Operation * calleeOpRaw = resolveCallee(call.getName());
  auto calleeOp = mlir::dyn_cast<ExternalFunctionOp>(calleeOpRaw);

  llvm::SmallVector<VariableOp> inputVariables;

  const ast::ASTNode *parentFunctionNode = call.getParent()->getParent();
  auto parentFunctionClass = parentFunctionNode->dyn_cast<const ast::Class>();

  auto parentFunctionOp = mlir::cast<FunctionOp>(getClass(*parentFunctionClass));

  getCustomFunctionInputVariables(inputVariables, parentFunctionOp);

  llvm::SmallVector<std::string, 3> argNames;
  llvm::SmallVector<mlir::Value, 3> argValues;

  if (!lowerCustomFunctionArgs(call, inputVariables, argNames, argValues)) {
    return false;
  }

  mlir::TypeRange resultTypes = calleeOp.getFunctionType().getResults();

  auto callOp = builder().create<CallOp>(loc(call.getLocation()), 
                  getSymbolRefFromRoot(calleeOp),
                  resultTypes, argValues);
  

  mlir::Value returnValue = callOp.getResult(0);

  if (call.hasComponentReference()) {
    const ast::ComponentReference* componentReferenceNode = call.getComponentReference();
    std::string componentReferenceName = componentReferenceNode->getName(); 
    auto outputVarOp = parentFunctionOp.lookupSymbol<VariableOp>(componentReferenceName);
    builder().create<VariableSetOp>(
      loc(call.getLocation()),
      outputVarOp,
      returnValue
    );
  }

  return true; 

}

mlir::Operation * ExternalFunctionCallLowerer::resolveCallee(llvm::StringRef calleeName) {

  mlir::Operation *rootScope = getRoot();
  
  mlir::Operation *foundOp = resolveSymbolName(calleeName, rootScope);

  return foundOp;
}

std::optional<mlir::Value>
ExternalFunctionCallLowerer::lowerArg(const ast::Expression &expression) {
  mlir::Location location = loc(expression.getLocation());
  auto loweredExpression = lower(expression);
  if (!loweredExpression) {
    return std::nullopt;
  }
  auto &results = *loweredExpression;
  assert(results.size() == 1);
  return results[0].get(location);
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


} // namespace marco::codegen::lowering
