#include "marco/Codegen/Lowering/ExternalFunctionCallLowerer.h"
#include "llvm/ADT/StringSwitch.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;


namespace marco::codegen::lowering {
ExternalFunctionCallLowerer::ExternalFunctionCallLowerer(BridgeInterface *bridge) : Lowerer(bridge) {}

bool ExternalFunctionCallLowerer::lower(const ast::ExternalFunctionCall &call) {

  auto calleeOp_ = resolveCallee(call.getName()); 
  auto calleeOp = mlir::dyn_cast<ExternalFunctionOp>(calleeOp_); 

  const ast::ASTNode *parentFunctionNode = call.getParent()->getParent();
  auto parentFunctionClass = parentFunctionNode->dyn_cast<const ast::Class>();
  auto parentFunctionOp = mlir::cast<FunctionOp>(getClass(*parentFunctionClass));

  llvm::SmallVector<mlir::Type> inputTypes;
  llvm::SmallVector<mlir::Value> argValues;
  llvm::SmallVector<mlir::Type> outputTypes;

  VariableOp outputVarOp; 

  if (call.hasComponentReference()) {
    const ast::ComponentReference* componentReferenceNode = call.getComponentReference();
    std::string componentReferenceName = componentReferenceNode->getName(); 
    outputVarOp = parentFunctionOp.lookupSymbol<VariableOp>(componentReferenceName);
    outputTypes.push_back(outputVarOp.getVariableType().unwrap());
  }  

  if (!lowerCustomFunctionArgs(call, argValues, inputTypes)) {
    return false;
  }

  mlir::FunctionType funcType = mlir::FunctionType::get(
    builder().getContext(), 
    inputTypes,             
    outputTypes         
  );

  calleeOp.setFunctionType(funcType);

  mlir::TypeRange resultTypes = funcType.getResults();

  auto callOp = builder().create<CallOp>(loc(call.getLocation()), 
                  getSymbolRefFromRoot(calleeOp),
                  resultTypes, argValues);
  
  mlir::Value returnValue = callOp.getResult(0);

  if (outputVarOp) {
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

bool ExternalFunctionCallLowerer::lowerCustomFunctionArgs(
      const ast::ExternalFunctionCall &call,
      llvm::SmallVectorImpl<mlir::Value> &argValues, 
      llvm::SmallVectorImpl<mlir::Type> &inputTypes) {

  auto args = call.getExpressions();

  for (size_t i = 0 ; i < args.size() ; i++){
    auto argValue = lowerArg(*args[i]->cast<ast::Expression>());
    if (!argValue) {
      return false;
    }
    argValues.push_back(*argValue);
    inputTypes.push_back(argValue->getType()); 
  }

  return true;
}

} // namespace marco::codegen::lowering
