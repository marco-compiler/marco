#define DEBUG_TYPE "scc-solving-with-kinsol"

#include "marco/Dialect/BaseModelica/Transforms/SCCSolvingWithKINSOL.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/AutomaticDifferentiation/ForwardAD.h"
#include "marco/Dialect/BaseModelica/Transforms/Solvers/SUNDIALS.h"
#include "marco/Dialect/KINSOL/IR/KINSOL.h"
#include "marco/Dialect/Runtime/IR/Runtime.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_SCCSOLVINGWITHKINSOLPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class FunctionsBuilder {
  mlir::SymbolTableCollection &symbolTables;
  mlir::ModuleOp moduleOp;
  ModelOp modelOp;

  // Variable getter functions.
  llvm::DenseMap<VariableOp, mlir::sundials::VariableGetterOp> variableGetters;

  // Variable getter functions.
  llvm::DenseMap<VariableOp, mlir::sundials::VariableSetterOp> variableSetters;

  // The access functions.
  llvm::DenseMap<mlir::AffineMap, mlir::sundials::AccessFunctionOp>
      accessFunctions;

  // The residual functions.
  llvm::DenseMap<EquationTemplateOp, mlir::kinsol::ResidualFunctionOp>
      residualFunctionsByTemplate;

  // The zero-valued residual functions used for proxy variables.
  // A function is created for each different rank of the proxy variables.
  llvm::DenseMap<int64_t, mlir::kinsol::ResidualFunctionOp>
      zeroResidualFunctionsByRank;

  // The partial derivative function templates.
  PartialDerivativeTemplatesCollection partialDerivativeTemplateFunctions;

  // The partial derivative functions. The key of the map is composed by the
  // equation and the independent variable.
  llvm::DenseMap<std::pair<EquationInstanceOp, VariableOp>,
                 mlir::kinsol::JacobianFunctionOp>
      partialDerivativeFunctions;

  // The partial derivative functions. The key of the map is composed by the
  // partially written variable and the independent variable.
  llvm::DenseMap<std::pair<VariableOp, VariableOp>,
                 mlir::kinsol::JacobianFunctionOp>
      partialDerivativeProxyFunctions;

public:
  FunctionsBuilder(mlir::SymbolTableCollection &symbolTables,
                   mlir::ModuleOp moduleOp, ModelOp modelOp);

  FunctionsBuilder(const FunctionsBuilder &) = delete;
  FunctionsBuilder(FunctionsBuilder &&) = delete;
  ~FunctionsBuilder() = default;
  FunctionsBuilder &operator=(const FunctionsBuilder &) = delete;
  FunctionsBuilder &operator=(FunctionsBuilder &&) = delete;

  mlir::sundials::VariableGetterOp
  getOrCreateVariableGetterFunction(mlir::OpBuilder &builder,
                                    VariableOp variableOp);

  mlir::sundials::VariableSetterOp
  getOrCreateVariableSetterFunction(mlir::OpBuilder &builder,
                                    VariableOp variableOp);

  mlir::sundials::AccessFunctionOp
  getOrCreateAccessFunction(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::AffineMap map);

  mlir::kinsol::ResidualFunctionOp
  getOrCreateZeroResidualFunction(mlir::OpBuilder &builder, int64_t rank);

  mlir::kinsol::ResidualFunctionOp
  getOrCreateResidualFunction(mlir::RewriterBase &rewriter,
                              EquationTemplateOp equationTemplateOp);

  FunctionOp
  getOrCreatePartialDerivativeFunction(mlir::RewriterBase &rewriter,
                                       EquationTemplateOp equationTemplateOp);

  mlir::kinsol::JacobianFunctionOp
  getOrCreatePartialDerivativeFunction(mlir::RewriterBase &rewriter,
                                       EquationInstanceOp equationOp,
                                       VariableOp independentVariable);

  llvm::SmallVector<int64_t>
  getSeedSizes(EquationTemplateOp equationTemplateOp);

  mlir::kinsol::JacobianFunctionOp
  getOrCreateProxyPartialDerivativeFunction(mlir::OpBuilder &builder,
                                            VariableOp partiallyWrittenVariable,
                                            VariableOp independentVariable);

private:
  mlir::LogicalResult
  replaceVariableGetOps(mlir::RewriterBase &rewriter,
                        llvm::ArrayRef<VariableGetOp> getOps);

  mlir::sundials::AccessFunctionOp
  createAccessFunction(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::AffineMap map);

  mlir::kinsol::ResidualFunctionOp
  createResidualFunction(mlir::RewriterBase &rewriter,
                         EquationTemplateOp equationTemplateOp);

  FunctionOp createPartialDerTemplateFromEquation(
      mlir::RewriterBase &rewriter, EquationTemplateOp templateOp,
      llvm::MapVector<VariableOp, size_t> &variablesPos,
      llvm::StringRef templateName);

  FunctionOp
  createPartialDerivativeFunction(mlir::RewriterBase &rewriter,
                                  EquationTemplateOp equationTemplateOp);

  mlir::kinsol::JacobianFunctionOp
  createPartialDerivativeFunction(mlir::RewriterBase &rewriter,
                                  EquationInstanceOp equationOp,
                                  VariableOp independentVariable);

  mlir::kinsol::JacobianFunctionOp
  createProxyPartialDerivativeFunction(mlir::OpBuilder &builder,
                                       VariableOp partiallyWrittenVariable,
                                       VariableOp independentVariable);
};
} // namespace

FunctionsBuilder::FunctionsBuilder(mlir::SymbolTableCollection &symbolTables,
                                   mlir::ModuleOp moduleOp, ModelOp modelOp)
    : symbolTables(symbolTables), moduleOp(moduleOp), modelOp(modelOp) {}

mlir::sundials::VariableGetterOp
FunctionsBuilder::getOrCreateVariableGetterFunction(mlir::OpBuilder &builder,
                                                    VariableOp variableOp) {
  if (auto it = variableGetters.find(variableOp); it != variableGetters.end()) {
    return it->second;
  }

  auto result = createGetterFunction(
      builder, symbolTables, variableOp.getLoc(), moduleOp, variableOp,
      "kinsol_var_getter_" + variableOp.getSymName().str());

  variableGetters[variableOp] = result;
  return result;
}

mlir::sundials::VariableSetterOp
FunctionsBuilder::getOrCreateVariableSetterFunction(mlir::OpBuilder &builder,
                                                    VariableOp variableOp) {
  if (auto it = variableSetters.find(variableOp); it != variableSetters.end()) {
    return it->second;
  }

  auto result = createSetterFunction(
      builder, symbolTables, variableOp.getLoc(), moduleOp, variableOp,
      "kinsol_var_setter_" + variableOp.getSymName().str());

  if (result) {
    variableSetters[variableOp] = result;
  }

  return result;
}

mlir::kinsol::ResidualFunctionOp FunctionsBuilder::getOrCreateResidualFunction(
    mlir::RewriterBase &rewriter, EquationTemplateOp equationTemplateOp) {
  if (auto it = residualFunctionsByTemplate.find(equationTemplateOp);
      it != residualFunctionsByTemplate.end()) {
    return it->second;
  }

  auto result = createResidualFunction(rewriter, equationTemplateOp);

  if (result) {
    residualFunctionsByTemplate[equationTemplateOp] = result;
  }

  return result;
}

mlir::sundials::AccessFunctionOp FunctionsBuilder::getOrCreateAccessFunction(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::AffineMap map) {
  if (auto it = accessFunctions.find(map); it != accessFunctions.end()) {
    return it->second;
  }

  auto result = createAccessFunction(builder, loc, map);

  if (result) {
    accessFunctions[map] = result;
  }

  return result;
}

mlir::kinsol::ResidualFunctionOp
FunctionsBuilder::getOrCreateZeroResidualFunction(mlir::OpBuilder &builder,
                                                  int64_t rank) {
  mlir::OpBuilder::InsertionGuard residualGuard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  if (auto it = zeroResidualFunctionsByRank.find(rank);
      it != zeroResidualFunctionsByRank.end()) {
    return it->second;
  }

  auto residualFunction = builder.create<mlir::kinsol::ResidualFunctionOp>(
      moduleOp.getLoc(), "kinsol_zero_residual", rank);

  symbolTables.getSymbolTable(moduleOp).insert(residualFunction,
                                               moduleOp.getBody()->end());

  mlir::Block *bodyBlock = residualFunction.addEntryBlock();
  builder.setInsertionPointToStart(bodyBlock);

  mlir::Value result = builder.create<mlir::arith::ConstantOp>(
      residualFunction.getLoc(), builder.getF64FloatAttr(0));

  builder.create<mlir::kinsol::ReturnOp>(residualFunction.getLoc(), result);

  if (residualFunction) {
    zeroResidualFunctionsByRank[rank] = residualFunction;
  }

  return residualFunction;
}

FunctionOp FunctionsBuilder::getOrCreatePartialDerivativeFunction(
    mlir::RewriterBase &rewriter, EquationTemplateOp equationTemplateOp) {
  if (auto templateFunc =
          partialDerivativeTemplateFunctions.getDerivativeTemplate(
              equationTemplateOp)) {
    return *templateFunc;
  }

  return createPartialDerivativeFunction(rewriter, equationTemplateOp);
}

mlir::kinsol::JacobianFunctionOp
FunctionsBuilder::getOrCreateProxyPartialDerivativeFunction(
    mlir::OpBuilder &builder, VariableOp partiallyWrittenVariable,
    VariableOp independentVariable) {
  if (auto it = partialDerivativeProxyFunctions.find(
          {partiallyWrittenVariable, independentVariable});
      it != partialDerivativeProxyFunctions.end()) {
    return it->second;
  }

  auto result = createProxyPartialDerivativeFunction(
      builder, partiallyWrittenVariable, independentVariable);

  if (result) {
    partialDerivativeProxyFunctions[{partiallyWrittenVariable,
                                     independentVariable}] = result;
  }

  return result;
}

mlir::kinsol::JacobianFunctionOp
FunctionsBuilder::getOrCreatePartialDerivativeFunction(
    mlir::RewriterBase &rewriter, EquationInstanceOp equationOp,
    VariableOp independentVariable) {
  if (auto it =
          partialDerivativeFunctions.find({equationOp, independentVariable});
      it != partialDerivativeFunctions.end()) {
    return it->second;
  }

  auto result = createPartialDerivativeFunction(rewriter, equationOp,
                                                independentVariable);

  if (result) {
    partialDerivativeFunctions[{equationOp, independentVariable}] = result;
  }

  return result;
}

mlir::LogicalResult
FunctionsBuilder::replaceVariableGetOps(mlir::RewriterBase &rewriter,
                                        llvm::ArrayRef<VariableGetOp> getOps) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  for (VariableGetOp variableGetOp : getOps) {
    rewriter.setInsertionPoint(variableGetOp);

    auto variableOp = symbolTables.lookupSymbolIn<VariableOp>(
        modelOp, variableGetOp.getVariableAttr());

    // Use the original variable.
    auto qualifiedGetOp = rewriter.create<QualifiedVariableGetOp>(
        variableGetOp.getLoc(), variableOp);

    rewriter.replaceOp(variableGetOp, qualifiedGetOp);
  }

  return mlir::success();
}

mlir::sundials::AccessFunctionOp FunctionsBuilder::createAccessFunction(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::AffineMap map) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  // Normalize the access so that it has at least one dimension and one result.
  llvm::SmallVector<mlir::AffineExpr> expressions;

  for (mlir::AffineExpr expression : map.getResults()) {
    expressions.push_back(expression);
  }

  if (expressions.empty()) {
    expressions.push_back(mlir::getAffineConstantExpr(0, builder.getContext()));
  }

  auto extendedAccess = mlir::AffineMap::get(
      std::max(static_cast<unsigned int>(1), map.getNumDims()),
      map.getNumSymbols(), expressions, builder.getContext());

  // Create the operation for the access function.
  auto accessFunctionOp = builder.create<mlir::sundials::AccessFunctionOp>(
      loc, "kinsol_access_function", extendedAccess.getNumDims(),
      extendedAccess.getNumResults());

  symbolTables.getSymbolTable(moduleOp).insert(accessFunctionOp);

  mlir::Block *bodyBlock = accessFunctionOp.addEntryBlock();
  builder.setInsertionPointToStart(bodyBlock);

  // Materialize the access.
  llvm::SmallVector<mlir::Value, 3> results;

  if (mlir::failed(materializeAffineMap(builder, loc, extendedAccess,
                                        accessFunctionOp.getEquationIndices(),
                                        results))) {
    return nullptr;
  }

  builder.create<mlir::sundials::ReturnOp>(loc, results);
  return accessFunctionOp;
}

mlir::kinsol::ResidualFunctionOp FunctionsBuilder::createResidualFunction(
    mlir::RewriterBase &rewriter, EquationTemplateOp equationTemplateOp) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  mlir::Location loc = equationTemplateOp.getLoc();
  size_t numOfInductionVariables =
      equationTemplateOp.getInductionVariables().size();

  auto residualFunction = rewriter.create<mlir::kinsol::ResidualFunctionOp>(
      loc, "kinsol_residual", numOfInductionVariables);

  symbolTables.getSymbolTable(moduleOp).insert(residualFunction);

  mlir::Block *bodyBlock = residualFunction.addEntryBlock();
  rewriter.setInsertionPointToStart(bodyBlock);

  // Map for the SSA values.
  mlir::IRMapping mapping;

  // Map the iteration variables.
  auto originalInductions = equationTemplateOp.getInductionVariables();
  auto mappedInductions = residualFunction.getEquationIndices();
  assert(originalInductions.size() == mappedInductions.size());

  for (size_t i = 0, e = originalInductions.size(); i < e; ++i) {
    mapping.map(originalInductions[i], mappedInductions[i]);
  }

  for (auto &op : equationTemplateOp.getOps()) {
    if (mlir::isa<EquationSideOp>(op)) {
      continue;
    }

    if (auto equationSidesOp = mlir::dyn_cast<EquationSidesOp>(op)) {
      // Compute the difference between the right-hand side and the left-hand
      // side of the equation.
      mlir::Value lhs = mapping.lookup(equationSidesOp.getLhsValues()[0]);
      mlir::Value rhs = mapping.lookup(equationSidesOp.getRhsValues()[0]);
      assert(!mlir::isa<mlir::ShapedType>(lhs.getType()));
      assert(!mlir::isa<mlir::ShapedType>(rhs.getType()));

      mlir::Value difference =
          rewriter.create<SubOp>(loc, rewriter.getF64Type(), rhs, lhs);

      rewriter.create<mlir::kinsol::ReturnOp>(difference.getLoc(), difference);
    } else {
      rewriter.clone(op, mapping);
    }
  }

  // Replace the original variable accesses.
  llvm::SmallVector<VariableGetOp> getOps;

  residualFunction.walk([&](VariableGetOp getOp) { getOps.push_back(getOp); });

  if (mlir::failed(replaceVariableGetOps(rewriter, getOps))) {
    return nullptr;
  }

  return residualFunction;
}

mlir::bmodelica::FunctionOp
FunctionsBuilder::createPartialDerTemplateFromEquation(
    mlir::RewriterBase &rewriter, EquationTemplateOp templateOp,
    llvm::MapVector<VariableOp, size_t> &variablesPos,
    llvm::StringRef templateName) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  mlir::Location loc = templateOp.getLoc();

  // Create the function.
  std::string functionOpName = templateName.str() + "_base";

  // Create the function to be derived.
  auto functionOp = rewriter.create<FunctionOp>(loc, functionOpName);
  symbolTables.getSymbolTable(moduleOp).insert(functionOp);
  rewriter.createBlock(&functionOp.getBodyRegion());

  // Start the body of the function.
  rewriter.setInsertionPointToStart(functionOp.getBody());

  // Determine the variables needed by the equation.
  llvm::SmallVector<VariableAccess> accesses;

  if (mlir::failed(templateOp.getAccesses(accesses, symbolTables))) {
    return nullptr;
  }

  llvm::DenseSet<VariableOp> accessedVariables;

  for (const VariableAccess &access : accesses) {
    auto variableOp =
        symbolTables.lookupSymbolIn<VariableOp>(modelOp, access.getVariable());

    if (!variableOp) {
      return nullptr;
    }

    accessedVariables.insert(variableOp);
  }

  // Replicate the variables inside the function.
  llvm::StringMap<VariableOp> localVariableOps;
  size_t variableIndex = 0;

  for (VariableOp variableOp : accessedVariables) {
    VariableType variableType =
        variableOp.getVariableType().withIOProperty(IOProperty::input);

    auto clonedVariableOp = rewriter.create<VariableOp>(
        variableOp.getLoc(), variableOp.getSymName(), variableType);

    symbolTables.getSymbolTable(functionOp).insert(clonedVariableOp);
    localVariableOps[variableOp.getSymName()] = clonedVariableOp;
    variablesPos[variableOp] = variableIndex++;
  }

  // Create the induction variables.
  llvm::SmallVector<VariableOp, 3> inductionVariablesOps;

  size_t numOfInductions = templateOp.getInductionVariables().size();

  for (size_t i = 0; i < numOfInductions; ++i) {
    std::string variableName = "ind" + std::to_string(i);

    auto variableType = VariableType::wrap(
        rewriter.getIndexType(), VariabilityProperty::none, IOProperty::input);

    auto variableOp =
        rewriter.create<VariableOp>(loc, variableName, variableType);

    symbolTables.getSymbolTable(functionOp).insert(variableOp);
    inductionVariablesOps.push_back(variableOp);
  }

  // Create the output variable, that is the difference between its equation
  // right-hand side value and its left-hand side value.
  auto outputVariableOp = rewriter.create<VariableOp>(
      loc, "out",
      VariableType::wrap(RealType::get(rewriter.getContext()),
                         VariabilityProperty::none, IOProperty::output));

  symbolTables.getSymbolTable(functionOp).insert(outputVariableOp);

  // Create the body of the function.
  auto algorithmOp = rewriter.create<AlgorithmOp>(loc);

  rewriter.setInsertionPointToStart(
      rewriter.createBlock(&algorithmOp.getBodyRegion()));

  mlir::IRMapping mapping;

  // Get the values of the induction variables.
  auto originalInductions = templateOp.getInductionVariables();
  assert(originalInductions.size() <= inductionVariablesOps.size());

  for (size_t i = 0, e = originalInductions.size(); i < e; ++i) {
    mlir::Value mappedInduction = rewriter.create<VariableGetOp>(
        inductionVariablesOps[i].getLoc(),
        inductionVariablesOps[i].getVariableType().unwrap(),
        inductionVariablesOps[i].getSymName());

    mapping.map(originalInductions[i], mappedInduction);
  }

  // Determine the operations to be cloned by starting from the terminator and
  // walking through the dependencies.
  llvm::DenseSet<mlir::Operation *> toBeCloned;
  llvm::SmallVector<mlir::Operation *> toBeClonedVisitStack;

  auto equationSidesOp =
      mlir::cast<EquationSidesOp>(templateOp.getBody()->getTerminator());

  mlir::Value lhs = equationSidesOp.getLhsValues()[0];
  mlir::Value rhs = equationSidesOp.getRhsValues()[0];

  if (mlir::Operation *lhsOp = lhs.getDefiningOp()) {
    toBeClonedVisitStack.push_back(lhsOp);
  }

  if (mlir::Operation *rhsOp = rhs.getDefiningOp()) {
    toBeClonedVisitStack.push_back(rhsOp);
  }

  while (!toBeClonedVisitStack.empty()) {
    mlir::Operation *op = toBeClonedVisitStack.pop_back_val();
    toBeCloned.insert(op);

    for (mlir::Value operand : op->getOperands()) {
      if (auto operandOp = operand.getDefiningOp()) {
        toBeClonedVisitStack.push_back(operandOp);
      }
    }
  }

  // Clone the original operations and compute the residual value.
  for (auto &op : templateOp.getOps()) {
    if (!toBeCloned.contains(&op)) {
      continue;
    }

    if (mlir::isa<EquationSideOp, EquationSidesOp>(op)) {
      continue;
    }

    rewriter.clone(op, mapping);
  }

  mlir::Value mappedLhs = mapping.lookup(lhs);
  mlir::Value mappedRhs = mapping.lookup(rhs);

  auto result = rewriter.create<SubOp>(
      loc, RealType::get(rewriter.getContext()), mappedRhs, mappedLhs);

  rewriter.create<VariableSetOp>(loc, outputVariableOp, result);

  // Create the derivative template function.
  ad::forward::State state;

  auto derTemplate = ad::forward::createFunctionPartialDerivative(
      rewriter, symbolTables, state, functionOp, templateName);

  LLVM_DEBUG({
    llvm::dbgs() << "Function being derived:\n" << functionOp << "\n";
  });

  if (!derTemplate) {
    return nullptr;
  }

  symbolTables.getSymbolTable(moduleOp).remove(functionOp);
  symbolTables.invalidateSymbolTable(functionOp);
  rewriter.eraseOp(functionOp);

  // Replace the local variables with the global ones.
  llvm::DenseSet<VariableOp> variablesToBeReplaced;

  for (VariableOp variableOp : derTemplate->getVariables()) {
    if (localVariableOps.count(variableOp.getSymName()) != 0) {
      variablesToBeReplaced.insert(variableOp);
    }
  }

  llvm::SmallVector<VariableGetOp> variableGetOps;

  derTemplate->walk([&](VariableGetOp getOp) {
    if (localVariableOps.count(getOp.getVariable()) != 0) {
      variableGetOps.push_back(getOp);
    }
  });

  if (mlir::failed(replaceVariableGetOps(rewriter, variableGetOps))) {
    return nullptr;
  }

  for (VariableOp variableOp : variablesToBeReplaced) {
    rewriter.eraseOp(variableOp);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Derivative template:\n" << *derTemplate << "\n";
  });

  return *derTemplate;
}

FunctionOp FunctionsBuilder::createPartialDerivativeFunction(
    mlir::RewriterBase &rewriter, EquationTemplateOp equationTemplateOp) {
  llvm::MapVector<VariableOp, size_t> variablesPos;

  auto partialDerTemplate = createPartialDerTemplateFromEquation(
      rewriter, equationTemplateOp, variablesPos, "kinsol_pder_template");

  if (!partialDerTemplate) {
    return nullptr;
  }

  // Cache the function to avoid duplicates.
  partialDerivativeTemplateFunctions.setDerivativeTemplate(equationTemplateOp,
                                                           partialDerTemplate);

  for (auto &entry : variablesPos) {
    partialDerivativeTemplateFunctions.setVariablePos(
        equationTemplateOp, entry.first, entry.second);
  }

  partialDerivativeTemplateFunctions.setDerivativeTemplate(equationTemplateOp,
                                                           partialDerTemplate);

  return partialDerTemplate;
}

mlir::kinsol::JacobianFunctionOp
FunctionsBuilder::createPartialDerivativeFunction(
    mlir::RewriterBase &rewriter, EquationInstanceOp equationOp,
    VariableOp independentVariable) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  mlir::Location loc = equationOp.getLoc();

  // Get the partial derivative template function.
  FunctionOp pderTemplate =
      getOrCreatePartialDerivativeFunction(rewriter, equationOp.getTemplate());

  if (!pderTemplate) {
    return nullptr;
  }

  size_t numOfVars = partialDerivativeTemplateFunctions.getVariablesCount(
      equationOp.getTemplate());

  size_t numOfInductions = equationOp.getInductionVariables().size();

  // Create the function.
  auto jacobianFunction = rewriter.create<mlir::kinsol::JacobianFunctionOp>(
      loc, "kinsol_pder", numOfInductions,
      independentVariable.getVariableType().getRank(), numOfVars);

  symbolTables.getSymbolTable(moduleOp).insert(jacobianFunction);

  mlir::Block *bodyBlock = jacobianFunction.addEntryBlock();
  rewriter.setInsertionPointToStart(bodyBlock);

  // Get the seeds for the variables.
  llvm::SmallVector<mlir::Value> varSeeds(numOfVars, nullptr);

  for (VariableOp variableOp : partialDerivativeTemplateFunctions.getVariables(
           equationOp.getTemplate())) {
    auto pos = partialDerivativeTemplateFunctions.getVariablePos(
        equationOp.getTemplate(), variableOp);

    if (!pos) {
      return nullptr;
    }

    mlir::Value seed = rewriter.create<PoolVariableGetOp>(
        loc, variableOp.getVariableType().toArrayType(),
        jacobianFunction.getMemoryPool(), jacobianFunction.getADSeeds()[*pos]);

    varSeeds[*pos] = seed;
  }

  // Zero and one constants to be used to update the array seeds or for the
  // scalar seeds.
  mlir::Value zero =
      rewriter.create<ConstantOp>(loc, RealAttr::get(rewriter.getContext(), 0));

  mlir::Value one =
      rewriter.create<ConstantOp>(loc, RealAttr::get(rewriter.getContext(), 1));

  // Function to collect the arguments to be passed to the derivative template
  // function.
  auto collectArgsFn = [&](llvm::SmallVectorImpl<mlir::Value> &args) {
    // Equation indices.
    for (mlir::Value equationIndex : jacobianFunction.getEquationIndices()) {
      args.push_back(equationIndex);
    }

    // Seeds of the variables.
    for (mlir::Value seed : varSeeds) {
      auto seedArrayType = mlir::cast<ArrayType>(seed.getType());

      if (seedArrayType.isScalar()) {
        seed = rewriter.create<LoadOp>(loc, seed);
      } else {
        auto tensorType = mlir::RankedTensorType::get(
            seedArrayType.getShape(), seedArrayType.getElementType());

        seed = rewriter.create<ArrayToTensorOp>(loc, tensorType, seed);
      }

      args.push_back(seed);
    }

    // Seeds of the equation indices. They are all equal to zero.
    for (size_t i = 0; i < jacobianFunction.getEquationIndices().size(); ++i) {
      args.push_back(zero);
    }
  };

  llvm::SmallVector<mlir::Value> args;

  // Set the seed of the variable to one.
  auto oneSeedPosition = partialDerivativeTemplateFunctions.getVariablePos(
      equationOp.getTemplate(), independentVariable);

  assert(oneSeedPosition.has_value());

  if (!oneSeedPosition) {
    return nullptr;
  }

  rewriter.create<StoreOp>(loc, one, varSeeds[*oneSeedPosition],
                           jacobianFunction.getVariableIndices());

  // Call the template function.
  args.clear();
  collectArgsFn(args);

  auto firstTemplateCall = rewriter.create<CallOp>(loc, pderTemplate, args);
  mlir::Value result = firstTemplateCall.getResult(0);

  // Reset the seed of the variable.
  rewriter.create<StoreOp>(loc, zero, varSeeds[*oneSeedPosition],
                           jacobianFunction.getVariableIndices());

  // Return the result.
  result = rewriter.create<CastOp>(loc, rewriter.getF64Type(), result);
  rewriter.create<mlir::kinsol::ReturnOp>(loc, result);

  return jacobianFunction;
}

llvm::SmallVector<int64_t>
FunctionsBuilder::getSeedSizes(EquationTemplateOp equationTemplateOp) {
  assert(partialDerivativeTemplateFunctions.hasEquationTemplate(
             equationTemplateOp) &&
         "Equation template not yet registered");

  size_t numOfVars =
      partialDerivativeTemplateFunctions.getVariablesCount(equationTemplateOp);

  llvm::SmallVector<int64_t> seedSizes(numOfVars, 0);

  for (VariableOp variableOp :
       partialDerivativeTemplateFunctions.getVariables(equationTemplateOp)) {
    auto pos = partialDerivativeTemplateFunctions.getVariablePos(
        equationTemplateOp, variableOp);

    assert(pos && "Variable position not found");

    seedSizes[*pos] =
        variableOp.getVariableType().toArrayType().getNumElements();
  }

  assert(llvm::none_of(seedSizes, [](int64_t size) { return size == 0; }) &&
         "Some seed sizes are zero");

  return seedSizes;
}

mlir::kinsol::JacobianFunctionOp
FunctionsBuilder::createProxyPartialDerivativeFunction(
    mlir::OpBuilder &builder, VariableOp partiallyWrittenVariable,
    VariableOp independentVariable) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  mlir::Location loc = partiallyWrittenVariable.getLoc();

  int64_t partiallyWrittenVariableRank =
      partiallyWrittenVariable.getVariableType().getRank();

  int64_t independentVariableRank =
      independentVariable.getVariableType().getRank();

  // Create the function.
  auto jacobianFunction = builder.create<mlir::kinsol::JacobianFunctionOp>(
      loc, "kinsol_proxy_pder_", partiallyWrittenVariableRank,
      independentVariableRank, 0);

  symbolTables.getSymbolTable(moduleOp).insert(jacobianFunction);

  mlir::Block *bodyBlock = jacobianFunction.addEntryBlock();
  builder.setInsertionPointToStart(bodyBlock);

  mlir::Value zero =
      builder.create<mlir::arith::ConstantOp>(loc, builder.getF64FloatAttr(0));

  mlir::Value result = zero;

  if (partiallyWrittenVariable == independentVariable) {
    assert(partiallyWrittenVariableRank == independentVariableRank);

    mlir::Value one = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getF64FloatAttr(1));

    mlir::Value condition =
        builder.create<mlir::arith::ConstantOp>(loc, builder.getBoolAttr(true));

    for (int64_t i = 0; i < partiallyWrittenVariableRank; ++i) {
      mlir::Value equal = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq,
          jacobianFunction.getEquationIndices()[i],
          jacobianFunction.getVariableIndices()[i]);

      condition = builder.create<mlir::arith::AndIOp>(loc, condition, equal);
    }

    result = builder.create<mlir::arith::SelectOp>(loc, condition, one, zero);
  }

  // Return the result.
  builder.create<mlir::kinsol::ReturnOp>(loc, result);

  return jacobianFunction;
}

namespace {
class KINSOLInstance {
public:
  KINSOLInstance(llvm::StringRef identifier,
                 mlir::SymbolTableCollection &symbolTables,
                 bool reducedDerivatives, bool debugInformation,
                 FunctionsBuilder &functionsBuilder);

  [[nodiscard]] bool hasVariable(VariableOp variable) const;

  void addVariable(VariableOp variable, const IndexSet &writtenIndices);

  void addEquation(EquationInstanceOp equation);

  mlir::LogicalResult initialize(mlir::RewriterBase &rewriter,
                                 mlir::Location loc, mlir::ModuleOp moduleOp,
                                 ModelOp modelOp,
                                 llvm::ArrayRef<VariableOp> variableOps);

  mlir::func::FuncOp createInitializationFunction(mlir::OpBuilder &builder,
                                                  mlir::Location loc,
                                                  mlir::ModuleOp moduleOp);

  mlir::LogicalResult performSolve(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::ModuleOp moduleOp);

  RawFunctionOp createSolveFunction(mlir::OpBuilder &builder,
                                    mlir::Location loc,
                                    mlir::ModuleOp moduleOp);

  mlir::LogicalResult deinitialize(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::ModuleOp moduleOp);

  mlir::func::FuncOp createDeinitializationFunction(mlir::OpBuilder &builder,
                                                    mlir::Location loc,
                                                    mlir::ModuleOp moduleOp);

private:
  mlir::LogicalResult
  addVariablesToKINSOL(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::ModuleOp moduleOp, ModelOp modelOp,
                       llvm::ArrayRef<VariableOp> variableOps);

  mlir::LogicalResult
  addEquationsToKINSOL(mlir::RewriterBase &rewriter, mlir::Location loc,
                       mlir::ModuleOp moduleOp, ModelOp modelOp,
                       llvm::ArrayRef<VariableOp> variableOps);

  mlir::LogicalResult addVariableAccessesInfoToKINSOL(
      mlir::OpBuilder &builder, mlir::Location loc, ModelOp modelOp,
      EquationInstanceOp equationOp, mlir::Value kinsolEquation);

  [[nodiscard]] std::string getKINSOLFunctionName(llvm::StringRef name) const;

private:
  /// Instance identifier.
  /// It is used to create unique symbols.
  std::string identifier;

  mlir::SymbolTableCollection &symbolTables;

  bool reducedDerivatives;
  bool debugInformation;

  /// The functions builder.
  FunctionsBuilder &functionsBuilder;

  /// The variables of the model that are managed by KINSOL.
  llvm::SmallVector<VariableOp> variables;

  /// The written indices of the variables.
  llvm::DenseMap<VariableOp, IndexSet> writtenVariableIndices;

  /// The SSA values of the KINSOL variables.
  llvm::SmallVector<mlir::Value> kinsolVariables;

  /// Map used for a faster lookup of the variable position.
  llvm::DenseMap<VariableOp, size_t> variablesLookup;

  /// The equations managed by KINSOL.
  llvm::SetVector<EquationInstanceOp> equations;
};
} // namespace

KINSOLInstance::KINSOLInstance(llvm::StringRef identifier,
                               mlir::SymbolTableCollection &symbolTables,
                               bool reducedDerivatives, bool debugInformation,
                               FunctionsBuilder &functionsBuilder)
    : identifier(identifier.str()), symbolTables(symbolTables),
      reducedDerivatives(reducedDerivatives),
      debugInformation(debugInformation), functionsBuilder(functionsBuilder) {}

bool KINSOLInstance::hasVariable(VariableOp variable) const {
  assert(variable != nullptr);
  return variablesLookup.contains(variable);
}

void KINSOLInstance::addVariable(VariableOp variable,
                                 const IndexSet &writtenIndices) {
  assert(variable != nullptr);

  if (!hasVariable(variable)) {
    variables.push_back(variable);
    variablesLookup[variable] = variables.size() - 1;
  }

  writtenVariableIndices[variable] += writtenIndices;
}

void KINSOLInstance::addEquation(EquationInstanceOp equation) {
  assert(equation != nullptr);
  equations.insert(equation);
}

mlir::LogicalResult
KINSOLInstance::initialize(mlir::RewriterBase &rewriter, mlir::Location loc,
                           mlir::ModuleOp moduleOp, ModelOp modelOp,
                           llvm::ArrayRef<VariableOp> variableOps) {
  mlir::func::FuncOp configureFuncOp =
      createInitializationFunction(rewriter, loc, moduleOp);

  rewriter.create<mlir::func::CallOp>(loc, configureFuncOp);

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&configureFuncOp.getBody().front());

  // Initialize the instance.
  rewriter.create<mlir::kinsol::InitOp>(loc, identifier);

  // Add the variables to KINSOL.
  if (mlir::failed(addVariablesToKINSOL(rewriter, loc, moduleOp, modelOp,
                                        variableOps))) {
    return mlir::failure();
  }

  // Add the equations to KINSOL.
  if (mlir::failed(addEquationsToKINSOL(rewriter, loc, moduleOp, modelOp,
                                        variableOps))) {
    return mlir::failure();
  }

  rewriter.create<mlir::func::ReturnOp>(loc);
  return mlir::success();
}

mlir::func::FuncOp KINSOLInstance::createInitializationFunction(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::ModuleOp moduleOp) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto funcOp = builder.create<mlir::func::FuncOp>(
      loc, getKINSOLFunctionName("init"), builder.getFunctionType({}, {}));

  symbolTables.getSymbolTable(moduleOp).insert(funcOp,
                                               moduleOp.getBody()->end());

  funcOp.addEntryBlock();
  return funcOp;
}

mlir::LogicalResult KINSOLInstance::performSolve(mlir::OpBuilder &builder,
                                                 mlir::Location loc,
                                                 mlir::ModuleOp moduleOp) {
  // Create a dedicated function to invoke the solver.
  RawFunctionOp funcOp = createSolveFunction(builder, loc, moduleOp);
  builder.create<CallOp>(loc, funcOp);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&funcOp.getBody().front());

  builder.create<mlir::kinsol::SolveOp>(loc, identifier);
  builder.create<RawReturnOp>(loc);
  return mlir::success();
}

RawFunctionOp KINSOLInstance::createSolveFunction(mlir::OpBuilder &builder,
                                                  mlir::Location loc,
                                                  mlir::ModuleOp moduleOp) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  // The call operation is inside the model, and the semantics of the
  // func::FuncOp operation prohibit referencing functions outside of the
  // closest symbol table. Therefore, we use a RawFunctionOp.
  auto funcOp = builder.create<RawFunctionOp>(
      loc, getKINSOLFunctionName("solve"), builder.getFunctionType({}, {}));

  symbolTables.getSymbolTable(moduleOp).insert(funcOp,
                                               moduleOp.getBody()->end());

  funcOp.addEntryBlock();
  return funcOp;
}

mlir::LogicalResult KINSOLInstance::deinitialize(mlir::OpBuilder &builder,
                                                 mlir::Location loc,
                                                 mlir::ModuleOp moduleOp) {
  mlir::func::FuncOp funcOp =
      createDeinitializationFunction(builder, loc, moduleOp);

  builder.create<mlir::func::CallOp>(loc, funcOp);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&funcOp.getBody().front());

  llvm::DenseMap<mlir::AffineMap, mlir::sundials::AccessFunctionOp>
      accessFunctionsMap;

  // Destroy the instance.
  builder.create<mlir::kinsol::FreeOp>(loc, identifier);

  builder.create<mlir::func::ReturnOp>(loc);
  return mlir::success();
}

mlir::func::FuncOp KINSOLInstance::createDeinitializationFunction(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::ModuleOp moduleOp) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto funcOp = builder.create<mlir::func::FuncOp>(
      loc, getKINSOLFunctionName("deinit"), builder.getFunctionType({}, {}));

  symbolTables.getSymbolTable(moduleOp).insert(funcOp,
                                               moduleOp.getBody()->end());

  funcOp.addEntryBlock();
  return funcOp;
}

mlir::LogicalResult KINSOLInstance::addVariablesToKINSOL(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::ModuleOp moduleOp,
    ModelOp modelOp, llvm::ArrayRef<VariableOp> variableOps) {
  // Function to get the dimensions of a variable.
  auto getDimensionsFn = [](ArrayType arrayType) -> std::vector<int64_t> {
    assert(arrayType.hasStaticShape());

    std::vector<int64_t> dimensions;

    if (arrayType.isScalar()) {
      // In case of scalar variables, the shape of the array would be empty
      // but KINSOL needs to see a single dimension of value 1.
      dimensions.push_back(1);
    } else {
      auto shape = arrayType.getShape();
      dimensions.insert(dimensions.end(), shape.begin(), shape.end());
    }

    return dimensions;
  };

  // Algebraic variables.
  for (VariableOp variableOp : variables) {
    // Declare the variable inside the KINSOL instance.
    auto arrayType = variableOp.getVariableType().toArrayType();
    std::vector<int64_t> dimensions = getDimensionsFn(arrayType);

    mlir::sundials::VariableGetterOp variableGetter =
        functionsBuilder.getOrCreateVariableGetterFunction(builder, variableOp);

    mlir::sundials::VariableSetterOp variableSetter =
        functionsBuilder.getOrCreateVariableSetterFunction(builder, variableOp);

    auto addVariableOp = builder.create<mlir::kinsol::AddVariableOp>(
        loc, identifier, builder.getI64ArrayAttr(dimensions),
        variableGetter.getSymName(), variableSetter.getSymName());

    if (debugInformation) {
      addVariableOp.setNameAttr(variableOp.getSymNameAttr());
    }

    kinsolVariables.push_back(addVariableOp);

    LLVM_DEBUG({
      llvm::dbgs() << "Variable '" << variableOp.getSymName()
                   << "' added to KINSOL instance '" << identifier << "'\n";
    });
  }

  return mlir::success();
}

mlir::LogicalResult KINSOLInstance::addEquationsToKINSOL(
    mlir::RewriterBase &rewriter, mlir::Location loc, mlir::ModuleOp moduleOp,
    ModelOp modelOp, llvm::ArrayRef<VariableOp> variableOps) {
  llvm::DenseMap<VariableOp, mlir::Value> variablesMapping;

  for (const auto &[variable, kinsolVariable] :
       llvm::zip(variables, kinsolVariables)) {
    variablesMapping[variable] = kinsolVariable;
  }

  for (EquationInstanceOp equationOp : equations) {
    // Keep track of the accessed variables in order to reduce the amount of
    // generated partial derivatives.
    llvm::SmallVector<VariableAccess> accesses;
    llvm::DenseSet<VariableOp> accessedVariables;

    if (mlir::failed(equationOp.getAccesses(accesses, symbolTables))) {
      return mlir::failure();
    }

    for (const VariableAccess &access : accesses) {
      auto variableOp = symbolTables.lookupSymbolIn<VariableOp>(
          modelOp, access.getVariable());

      accessedVariables.insert(variableOp);
    }

    // Get the indices of the equation.
    IndexSet equationIndices = equationOp.getIterationSpace();

    if (equationIndices.empty()) {
      equationIndices = IndexSet(Point(0));
    }

    // Get the write access.
    llvm::SmallVector<VariableAccess> writeAccesses;

    if (mlir::failed(equationOp.getWriteAccesses(writeAccesses, symbolTables,
                                                 accesses))) {
      return mlir::failure();
    }

    llvm::sort(writeAccesses,
               [](const VariableAccess &first, const VariableAccess &second) {
                 if (first.getAccessFunction().isInvertible() &&
                     !second.getAccessFunction().isInvertible()) {
                   return true;
                 }

                 if (!first.getAccessFunction().isInvertible() &&
                     second.getAccessFunction().isInvertible()) {
                   return false;
                 }

                 return first < second;
               });

    // Create the partial derivative template.
    for (const MultidimensionalRange &range : llvm::make_range(
             equationIndices.rangesBegin(), equationIndices.rangesEnd())) {
      // Add the equation to the KINSOL instance.
      mlir::sundials::AccessFunctionOp accessFunctionOp =
          functionsBuilder.getOrCreateAccessFunction(
              rewriter, equationOp.getLoc(),
              writeAccesses[0].getAccessFunction().getAffineMap());

      if (!accessFunctionOp) {
        return mlir::failure();
      }

      auto kinsolEquation = rewriter.create<mlir::kinsol::AddEquationOp>(
          equationOp.getLoc(), identifier,
          mlir::kinsol::MultidimensionalRangeAttr::get(rewriter.getContext(),
                                                       range));

      if (debugInformation) {
        std::string stringRepresentation;
        llvm::raw_string_ostream stringOstream(stringRepresentation);
        equationOp.printInline(stringOstream);

        kinsolEquation.setStringRepresentationAttr(
            rewriter.getStringAttr(stringRepresentation));
      }

      if (reducedDerivatives) {
        // Inform KINSOL about the accesses performed by the equation.
        if (mlir::failed(addVariableAccessesInfoToKINSOL(
                rewriter, loc, modelOp, equationOp, kinsolEquation))) {
          return mlir::failure();
        }
      }

      // Create the residual function.
      mlir::kinsol::ResidualFunctionOp residualFunction =
          functionsBuilder.getOrCreateResidualFunction(
              rewriter, equationOp.getTemplate());

      if (!residualFunction) {
        return mlir::failure();
      }

      rewriter.create<mlir::kinsol::SetResidualOp>(
          loc, identifier, kinsolEquation, residualFunction.getSymName());

      // Create the Jacobian functions.
      // Notice that Jacobian functions are not created for derivative
      // variables. Those are already handled when encountering the state
      // variable through the 'alpha' parameter set into the derivative seed.

      assert(variables.size() == kinsolVariables.size());

      for (auto [variable, kinsolVariable] :
           llvm::zip(variables, kinsolVariables)) {
        if (reducedDerivatives && !accessedVariables.contains(variable)) {
          // The partial derivative is always zero.
          continue;
        }

        mlir::kinsol::JacobianFunctionOp jacobianFunction =
            functionsBuilder.getOrCreatePartialDerivativeFunction(
                rewriter, equationOp, variable);

        if (!jacobianFunction) {
          return mlir::failure();
        }

        llvm::SmallVector<int64_t> seedSizes =
            functionsBuilder.getSeedSizes(equationOp.getTemplate());

        rewriter.create<mlir::kinsol::AddJacobianOp>(
            loc, identifier, kinsolEquation, kinsolVariable,
            jacobianFunction.getSymName(), rewriter.getI64ArrayAttr(seedSizes));
      }
    }
  }

  // Add dummy equations for the unwritten variable indices.
  // If this task is not performed, then the Jacobian matrix would become
  // singular.

  for (VariableOp variableOp : variables) {
    IndexSet unwrittenIndices = variableOp.getIndices();

    if (auto it = writtenVariableIndices.find(variableOp);
        it != writtenVariableIndices.end()) {
      unwrittenIndices -= it->getSecond();
    }

    if (unwrittenIndices.empty()) {
      continue;
    }

    for (const MultidimensionalRange &range : llvm::make_range(
             unwrittenIndices.rangesBegin(), unwrittenIndices.rangesEnd())) {
      // Create an access function for the identity access.
      auto accessFunctionOp = functionsBuilder.getOrCreateAccessFunction(
          rewriter, variableOp.getLoc(),
          mlir::AffineMap::getMultiDimIdentityMap(range.rank(),
                                                  rewriter.getContext()));

      if (!accessFunctionOp) {
        return mlir::failure();
      }

      // Add the equation to the KINSOL instance.
      auto kinsolEquation = rewriter.create<mlir::kinsol::AddEquationOp>(
          variableOp.getLoc(), identifier,
          mlir::kinsol::MultidimensionalRangeAttr::get(rewriter.getContext(),
                                                       range));

      if (debugInformation) {
        kinsolEquation.setStringRepresentationAttr(
            rewriter.getStringAttr("proxy equation"));
      }

      if (reducedDerivatives) {
        rewriter.create<mlir::kinsol::AddVariableAccessOp>(
            loc, identifier, kinsolEquation,
            kinsolVariables[variablesLookup[variableOp]],
            accessFunctionOp.getSymName());
      }

      // Create the residual function.
      mlir::kinsol::ResidualFunctionOp residualFunction =
          functionsBuilder.getOrCreateZeroResidualFunction(
              rewriter, variableOp.getVariableType().getRank());

      rewriter.create<mlir::kinsol::SetResidualOp>(
          loc, identifier, kinsolEquation, residualFunction.getSymName());

      // Create the partial Jacobian functions.
      assert(variables.size() == kinsolVariables.size());

      for (auto [variable, kinsolVariable] :
           llvm::zip(variables, kinsolVariables)) {
        if (reducedDerivatives && variable != variableOp) {
          // The partial derivative is always zero.
          continue;
        }

        llvm::SmallVector<int64_t> seedSizes;

        mlir::kinsol::JacobianFunctionOp jacobianFunction =
            functionsBuilder.getOrCreateProxyPartialDerivativeFunction(
                rewriter, variableOp, variable);

        if (!jacobianFunction) {
          return mlir::failure();
        }

        rewriter.create<mlir::kinsol::AddJacobianOp>(
            loc, identifier, kinsolEquation, kinsolVariable,
            jacobianFunction.getSymName(), rewriter.getI64ArrayAttr(seedSizes));
      }
    }
  }

  return mlir::success();
}

mlir::LogicalResult KINSOLInstance::addVariableAccessesInfoToKINSOL(
    mlir::OpBuilder &builder, mlir::Location loc, ModelOp modelOp,
    EquationInstanceOp equationOp, mlir::Value kinsolEquation) {
  assert(mlir::isa<mlir::kinsol::EquationType>(kinsolEquation.getType()));

  // Keep track of the discovered accesses in order to avoid adding the same
  // access map multiple times for the same variable.
  llvm::DenseMap<mlir::Value, llvm::DenseSet<mlir::AffineMap>> maps;

  llvm::SmallVector<VariableAccess> accesses;

  if (mlir::failed(equationOp.getAccesses(accesses, symbolTables))) {
    return mlir::failure();
  }

  for (const VariableAccess &access : accesses) {
    auto variableOp =
        symbolTables.lookupSymbolIn<VariableOp>(modelOp, access.getVariable());

    if (!hasVariable(variableOp)) {
      continue;
    }

    mlir::Value kinsolVariable = kinsolVariables[variablesLookup[variableOp]];
    assert(kinsolVariable != nullptr);
    maps[kinsolVariable].insert(access.getAccessFunction().getAffineMap());
  }

  // Inform KINSOL about the discovered accesses.
  for (const auto &entry : maps) {
    mlir::Value kinsolVariable = entry.getFirst();

    for (mlir::AffineMap map : entry.getSecond()) {
      mlir::sundials::AccessFunctionOp accessFunctionOp =
          functionsBuilder.getOrCreateAccessFunction(builder, loc, map);

      if (!accessFunctionOp) {
        return mlir::failure();
      }

      builder.create<mlir::kinsol::AddVariableAccessOp>(
          loc, identifier, kinsolEquation, kinsolVariable,
          accessFunctionOp.getSymName());
    }
  }

  return mlir::success();
}

std::string KINSOLInstance::getKINSOLFunctionName(llvm::StringRef name) const {
  return identifier + "_" + name.str();
}

namespace {
class SCCSolvingWithKINSOLPass
    : public mlir::bmodelica::impl::SCCSolvingWithKINSOLPassBase<
          SCCSolvingWithKINSOLPass> {
public:
  using SCCSolvingWithKINSOLPassBase::SCCSolvingWithKINSOLPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult processModelOp(mlir::RewriterBase &rewriter,
                                     mlir::SymbolTableCollection &symbolTables,
                                     mlir::ModuleOp moduleOp, ModelOp modelOp);

  mlir::LogicalResult
  processInitialOp(mlir::RewriterBase &rewriter,
                   mlir::SymbolTableCollection &symbolTables,
                   mlir::ModuleOp moduleOp, ModelOp modelOp,
                   InitialOp initialOp, FunctionsBuilder &functionsBuilder);

  mlir::LogicalResult
  processDynamicOp(mlir::RewriterBase &rewriter,
                   mlir::SymbolTableCollection &symbolTables,
                   mlir::ModuleOp moduleOp, ModelOp modelOp,
                   DynamicOp dynamicOp, FunctionsBuilder &functionsBuilder);

  mlir::LogicalResult processSCC(
      mlir::RewriterBase &rewriter, mlir::SymbolTableCollection &symbolTables,
      mlir::ModuleOp moduleOp, ModelOp modelOp, SCCOp scc,
      llvm::function_ref<mlir::Block *(mlir::OpBuilder &, mlir::Location)>
          createBeginFn,
      llvm::function_ref<mlir::Block *(mlir::OpBuilder &, mlir::Location)>
          createEndFn,
      FunctionsBuilder &functionsBuilder);

  mlir::kinsol::InstanceOp
  declareInstance(mlir::OpBuilder &builder,
                  mlir::SymbolTableCollection &symbolTables, mlir::Location loc,
                  mlir::ModuleOp moduleOp);

  mlir::LogicalResult
  getAccessAttrs(llvm::SmallVectorImpl<Variable> &writtenVariables,
                 llvm::SmallVectorImpl<Variable> &readVariables,
                 mlir::SymbolTableCollection &symbolTables, SCCOp scc);

  mlir::LogicalResult createBeginFunction(
      mlir::RewriterBase &rewriter, mlir::ModuleOp moduleOp, ModelOp modelOp,
      mlir::Location loc, KINSOLInstance *kinsolInstance,
      llvm::ArrayRef<VariableOp> variables,
      llvm::function_ref<mlir::Block *(mlir::OpBuilder &, mlir::Location)>
          createBeginFn) const;

  mlir::LogicalResult createEndFunction(
      mlir::RewriterBase &rewriter, mlir::ModuleOp moduleOp, mlir::Location loc,
      KINSOLInstance *kinsolInstance,
      llvm::function_ref<mlir::Block *(mlir::OpBuilder &, mlir::Location)>
          createEndFn) const;
};
} // namespace

void SCCSolvingWithKINSOLPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();

  mlir::IRRewriter rewriter(&getContext());
  mlir::SymbolTableCollection symbolTables;
  llvm::SmallVector<ModelOp, 1> modelOps;

  walkClasses(getOperation(), [&](mlir::Operation *op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  for (ModelOp modelOp : modelOps) {
    if (mlir::failed(
            processModelOp(rewriter, symbolTables, moduleOp, modelOp))) {
      return signalPassFailure();
    }
  }
}

mlir::LogicalResult SCCSolvingWithKINSOLPass::processModelOp(
    mlir::RewriterBase &rewriter, mlir::SymbolTableCollection &symbolTables,
    mlir::ModuleOp moduleOp, ModelOp modelOp) {
  FunctionsBuilder functionsBuilder(symbolTables, moduleOp, modelOp);

  for (ScheduleOp scheduleOp : modelOp.getOps<ScheduleOp>()) {
    for (InitialOp initialOp : scheduleOp.getOps<InitialOp>()) {
      if (mlir::failed(processInitialOp(rewriter, symbolTables, moduleOp,
                                        modelOp, initialOp,
                                        functionsBuilder))) {
        return mlir::failure();
      }
    }

    for (DynamicOp dynamicOp : scheduleOp.getOps<DynamicOp>()) {
      if (mlir::failed(processDynamicOp(rewriter, symbolTables, moduleOp,
                                        modelOp, dynamicOp,
                                        functionsBuilder))) {
        return mlir::failure();
      }
    }
  }

  return mlir::success();
}

mlir::LogicalResult SCCSolvingWithKINSOLPass::processInitialOp(
    mlir::RewriterBase &rewriter, mlir::SymbolTableCollection &symbolTables,
    mlir::ModuleOp moduleOp, ModelOp modelOp, InitialOp initialOp,
    FunctionsBuilder &functionsBuilder) {
  auto createBeginFn = [&](mlir::OpBuilder &builder,
                           mlir::Location loc) -> mlir::Block * {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(moduleOp.getBody());
    auto beginFn = builder.create<mlir::runtime::ICModelBeginOp>(loc);
    return builder.createBlock(&beginFn.getBodyRegion());
  };

  auto createEndFn = [&](mlir::OpBuilder &builder,
                         mlir::Location loc) -> mlir::Block * {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(moduleOp.getBody());
    auto beginFn = builder.create<mlir::runtime::ICModelEndOp>(loc);
    return builder.createBlock(&beginFn.getBodyRegion());
  };

  llvm::SmallVector<SCCOp> SCCs;

  for (SCCOp scc : initialOp.getOps<SCCOp>()) {
    SCCs.push_back(scc);
  }

  for (SCCOp scc : SCCs) {
    if (mlir::succeeded(processSCC(rewriter, symbolTables, moduleOp, modelOp,
                                   scc, createBeginFn, createEndFn,
                                   functionsBuilder))) {
      rewriter.eraseOp(scc);
    }
  }

  return mlir::success();
}

mlir::LogicalResult SCCSolvingWithKINSOLPass::processDynamicOp(
    mlir::RewriterBase &rewriter, mlir::SymbolTableCollection &symbolTables,
    mlir::ModuleOp moduleOp, ModelOp modelOp, DynamicOp dynamicOp,
    FunctionsBuilder &functionsBuilder) {
  auto createBeginFn = [&](mlir::OpBuilder &builder,
                           mlir::Location loc) -> mlir::Block * {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(moduleOp.getBody());
    auto beginFn = builder.create<mlir::runtime::DynamicModelBeginOp>(loc);
    return builder.createBlock(&beginFn.getBodyRegion());
  };

  auto createEndFn = [&](mlir::OpBuilder &builder,
                         mlir::Location loc) -> mlir::Block * {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(moduleOp.getBody());
    auto beginFn = builder.create<mlir::runtime::DynamicModelEndOp>(loc);
    return builder.createBlock(&beginFn.getBodyRegion());
  };

  // Process the SCCs.
  llvm::SmallVector<SCCOp> SCCs;
  llvm::append_range(SCCs, dynamicOp.getOps<SCCOp>());

  for (SCCOp scc : SCCs) {
    if (mlir::succeeded(processSCC(rewriter, symbolTables, moduleOp, modelOp,
                                   scc, createBeginFn, createEndFn,
                                   functionsBuilder))) {
      rewriter.eraseOp(scc);
    }
  }

  return mlir::success();
}

mlir::LogicalResult SCCSolvingWithKINSOLPass::processSCC(
    mlir::RewriterBase &rewriter, mlir::SymbolTableCollection &symbolTables,
    mlir::ModuleOp moduleOp, ModelOp modelOp, SCCOp scc,
    llvm::function_ref<mlir::Block *(mlir::OpBuilder &, mlir::Location)>
        createBeginFn,
    llvm::function_ref<mlir::Block *(mlir::OpBuilder &, mlir::Location)>
        createEndFn,
    FunctionsBuilder &functionsBuilder) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  LLVM_DEBUG({
    llvm::dbgs() << "Processing SCC composed by:\n";

    for (EquationInstanceOp equation : scc.getOps<EquationInstanceOp>()) {
      llvm::dbgs() << "  ";
      equation.printInline(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
  });

  rewriter.setInsertionPoint(scc);

  VariablesList writtenVariables;
  VariablesList readVariables;

  if (mlir::failed(
          getAccessAttrs(writtenVariables, readVariables, symbolTables, scc))) {
    return mlir::failure();
  }

  auto scheduleBlockOp = rewriter.create<ScheduleBlockOp>(
      scc.getLoc(), false, writtenVariables, readVariables);

  llvm::SmallVector<VariableOp> variables;

  // Determine the unwritten indices among the externalized variables.
  llvm::DenseMap<VariableOp, IndexSet> writtenVariableIndices;

  for (const Variable &variable :
       scheduleBlockOp.getProperties().writtenVariables) {
    auto writtenVariableName = variable.name;

    auto writtenVariableOp =
        symbolTables.lookupSymbolIn<VariableOp>(modelOp, writtenVariableName);

    variables.push_back(writtenVariableOp);
    writtenVariableIndices[writtenVariableOp] += variable.indices;
  }

  // Create the instance of the solver.
  auto instanceOp =
      declareInstance(rewriter, symbolTables, modelOp.getLoc(), moduleOp);

  auto kinsolInstance = std::make_unique<KINSOLInstance>(
      instanceOp.getSymName(), symbolTables, reducedDerivatives,
      debugInformation, functionsBuilder);

  // Add the variables to KINSOL.
  for (VariableOp variable : variables) {
    LLVM_DEBUG(llvm::dbgs()
               << "Add variable: " << variable.getSymName() << "\n");

    kinsolInstance->addVariable(variable, writtenVariableIndices[variable]);
  }

  for (EquationInstanceOp equation : scc.getOps<EquationInstanceOp>()) {
    LLVM_DEBUG({
      llvm::dbgs() << "Add equation\n";
      equation.printInline(llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    kinsolInstance->addEquation(equation);
  }

  if (mlir::failed(createBeginFunction(rewriter, moduleOp, modelOp,
                                       scc.getLoc(), kinsolInstance.get(),
                                       variables, createBeginFn))) {
    return mlir::failure();
  }

  if (mlir::failed(createEndFunction(rewriter, moduleOp, modelOp.getLoc(),
                                     kinsolInstance.get(), createEndFn))) {
    return mlir::failure();
  }

  rewriter.createBlock(&scheduleBlockOp.getBodyRegion());
  rewriter.setInsertionPointToStart(scheduleBlockOp.getBody());

  if (mlir::failed(
          kinsolInstance->performSolve(rewriter, scc.getLoc(), moduleOp))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::kinsol::InstanceOp SCCSolvingWithKINSOLPass::declareInstance(
    mlir::OpBuilder &builder, mlir::SymbolTableCollection &symbolTables,
    mlir::Location loc, mlir::ModuleOp moduleOp) {
  // Create the instance.
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());

  auto instanceOp = builder.create<mlir::kinsol::InstanceOp>(loc, "kinsol");

  // Update the symbol table.
  symbolTables.getSymbolTable(moduleOp).insert(instanceOp);

  return instanceOp;
}

mlir::LogicalResult SCCSolvingWithKINSOLPass::getAccessAttrs(
    llvm::SmallVectorImpl<Variable> &writtenVariables,
    llvm::SmallVectorImpl<Variable> &readVariables,
    mlir::SymbolTableCollection &symbolTables, SCCOp scc) {
  llvm::DenseMap<mlir::SymbolRefAttr, IndexSet> writtenVariablesIndices;
  llvm::DenseMap<mlir::SymbolRefAttr, IndexSet> readVariablesIndices;

  for (EquationInstanceOp equationOp : scc.getOps<EquationInstanceOp>()) {
    IndexSet equationIndices = equationOp.getIterationSpace();

    writtenVariablesIndices[equationOp.getProperties().match.name] +=
        equationOp.getProperties().match.indices;

    llvm::SmallVector<VariableAccess> accesses;
    llvm::SmallVector<VariableAccess> readAccesses;

    if (mlir::failed(equationOp.getAccesses(accesses, symbolTables))) {
      return mlir::failure();
    }

    if (mlir::failed(
            equationOp.getReadAccesses(readAccesses, symbolTables, accesses))) {
      return mlir::failure();
    }

    for (const VariableAccess &readAccess : readAccesses) {
      const AccessFunction &accessFunction = readAccess.getAccessFunction();
      IndexSet readIndices = accessFunction.map(equationIndices);
      readVariablesIndices[readAccess.getVariable()] += readIndices;
    }
  }

  for (const auto &entry : writtenVariablesIndices) {
    writtenVariables.emplace_back(entry.getFirst(), entry.getSecond());
  }

  for (const auto &entry : readVariablesIndices) {
    readVariables.emplace_back(entry.getFirst(), entry.getSecond());
  }

  return mlir::success();
}

mlir::LogicalResult SCCSolvingWithKINSOLPass::createBeginFunction(
    mlir::RewriterBase &rewriter, mlir::ModuleOp moduleOp, ModelOp modelOp,
    mlir::Location loc, KINSOLInstance *kinsolInstance,
    llvm::ArrayRef<VariableOp> variables,
    llvm::function_ref<mlir::Block *(mlir::OpBuilder &, mlir::Location)>
        createBeginFn) const {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  mlir::Block *bodyBlock = createBeginFn(rewriter, loc);
  rewriter.setInsertionPointToStart(bodyBlock);

  if (mlir::failed(kinsolInstance->initialize(rewriter, loc, moduleOp, modelOp,
                                              variables))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult SCCSolvingWithKINSOLPass::createEndFunction(
    mlir::RewriterBase &rewriter, mlir::ModuleOp moduleOp, mlir::Location loc,
    KINSOLInstance *kinsolInstance,
    llvm::function_ref<mlir::Block *(mlir::OpBuilder &, mlir::Location)>
        createEndFn) const {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  mlir::Block *bodyBlock = createEndFn(rewriter, loc);
  rewriter.setInsertionPointToStart(bodyBlock);

  if (mlir::failed(kinsolInstance->deinitialize(rewriter, loc, moduleOp))) {
    return mlir::failure();
  }

  return mlir::success();
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createSCCSolvingWithKINSOLPass() {
  return std::make_unique<SCCSolvingWithKINSOLPass>();
}

std::unique_ptr<mlir::Pass>
createSCCSolvingWithKINSOLPass(const SCCSolvingWithKINSOLPassOptions &options) {
  return std::make_unique<SCCSolvingWithKINSOLPass>(options);
}
} // namespace mlir::bmodelica
