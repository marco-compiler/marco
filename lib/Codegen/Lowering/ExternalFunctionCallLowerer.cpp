#include "marco/Codegen/Lowering/ExternalFunctionCallLowerer.h"
#include "llvm/ADT/StringSwitch.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;


namespace marco::codegen::lowering {
ExternalFunctionCallLowerer::ExternalFunctionCallLowerer(BridgeInterface *bridge) : Lowerer(bridge) {}

std::optional<Results> ExternalFunctionCallLowerer::lower(const ast::ExternalFunctionCall &call) {

  std::optional<mlir::Operation *> calleeOp = resolveCallee(call.getName());

  llvm::SmallVector<VariableOp> inputVariables;

//  auto externalFunctionCallOp = mlir::dyn_cast<ExternalFunctionCallOp>(*calleeOp);
  const ast::ASTNode *grandparent = call.getParent()->getParent();
  auto parentClassNode = grandparent->dyn_cast<const ast::Class>();

  auto parentFunctionOp = mlir::cast<FunctionOp>(getClass(*parentClassNode));

  getCustomFunctionInputVariables(inputVariables, parentFunctionOp);

  llvm::SmallVector<std::string, 3> argNames;
  llvm::SmallVector<mlir::Value, 3> argValues;

  if (!FunctionArgs(call, inputVariables, argNames, argValues)) {
    return std::nullopt;
  }

  auto callOp = builder().create<CallOp>(loc(call.getLocation()),
                                        getSymbolRefFromRoot(*calleeOp),
                                        resultTypes, argValues);

  
  for (auto result : callOp->getResults()) {
    results.push_back(Reference::ssa(builder(), result));
  }
  
  return Results(results.begin(), results.end());

}

std::optional<mlir::Operation *>
ExternalFunctionCallLowerer::resolveCallee(llvm::StringRef calleeName) {

  mlir::Operation *result =
      resolveSymbolName(calleeName, getLookupScope());

  result = getSymbolTable().lookupSymbolIn(
    result, builder().getStringAttr(calleeName));

  return result;
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
    const ast::Call &call, llvm::ArrayRef<VariableOp> calleeInputs,
    llvm::SmallVectorImpl<std::string> &argNames,
    llvm::SmallVectorImpl<mlir::Value> &argValues) {
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
  }

  return true;
}

} // namespace marco::codegen::lowering
