#include "marco/Codegen/Transforms/SCCSolvingWithKINSOL.h"
#include "marco/Codegen/Analysis/DerivativesMap.h"
#include "marco/Dialect/BaseModelica/ModelicaDialect.h"
#include "marco/Dialect/KINSOL/KINSOLDialect.h"
#include "marco/Dialect/Runtime/RuntimeDialect.h"
#include "marco/Codegen/Analysis/VariableAccessAnalysis.h"
#include "marco/Codegen/Transforms/Solvers/SUNDIALS.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation/ForwardAD.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "scc-solving-with-kinsol"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_SCCSOLVINGWITHKINSOLPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

namespace
{
  class KINSOLInstance
  {
    public:
      KINSOLInstance(
          llvm::StringRef identifier,
          mlir::SymbolTableCollection& symbolTableCollection,
          bool reducedDerivatives,
          bool debugInformation);

      [[nodiscard]] bool hasVariable(VariableOp variable) const;

      void addVariable(VariableOp variable);

      void addEquation(ScheduledEquationInstanceOp equation);

      mlir::LogicalResult initialize(
          mlir::OpBuilder& builder,
          mlir::Location loc);

      mlir::LogicalResult configure(
          mlir::RewriterBase& rewriter,
          mlir::Location loc,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variableOps);

      mlir::LogicalResult performSolve(
          mlir::OpBuilder& builder,
          mlir::Location loc);

      mlir::LogicalResult deleteInstance(
          mlir::OpBuilder& builder,
          mlir::Location loc);

    private:
      mlir::LogicalResult addVariablesToKINSOL(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variableOps);

      mlir::sundials::VariableGetterOp createGetterFunction(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::ModuleOp moduleOp,
          VariableOp variable,
          llvm::StringRef functionName);

      mlir::sundials::VariableSetterOp createSetterFunction(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::ModuleOp moduleOp,
          VariableOp variable,
          llvm::StringRef functionName);

      mlir::LogicalResult addEquationsToKINSOL(
          mlir::RewriterBase& rewriter,
          mlir::Location loc,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variableOps,
          llvm::DenseMap<
              mlir::AffineMap,
              mlir::sundials::AccessFunctionOp>& accessFunctionsMap);

      mlir::LogicalResult addVariableAccessesInfoToKINSOL(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          ModelOp modelOp,
          ScheduledEquationInstanceOp equationOp,
          mlir::Value kinsolEquation,
          llvm::DenseMap<
              mlir::AffineMap,
              mlir::sundials::AccessFunctionOp>& accessFunctionsMap,
          size_t& accessFunctionsCounter);

      mlir::sundials::AccessFunctionOp getOrCreateAccessFunction(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::ModuleOp moduleOp,
          mlir::AffineMap access,
          llvm::StringRef functionNamePrefix,
          llvm::DenseMap<
              mlir::AffineMap,
              mlir::sundials::AccessFunctionOp>& accessFunctionsMap,
          size_t& accessFunctionsCounter);

      mlir::sundials::AccessFunctionOp createAccessFunction(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::ModuleOp moduleOp,
          mlir::AffineMap access,
          llvm::StringRef functionName);

      mlir::LogicalResult createResidualFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          ScheduledEquationInstanceOp equationOp,
          mlir::Value kinsolEquation,
          llvm::StringRef residualFunctionName);

      mlir::LogicalResult getIndependentVariablesForAD(
          llvm::DenseSet<VariableOp>& result,
          ModelOp modelOp,
          ScheduledEquationInstanceOp equationOp);

      mlir::LogicalResult createPartialDerTemplateFunction(
          mlir::RewriterBase& rewriter,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          ScheduledEquationInstanceOp equationOp,
          llvm::DenseMap<VariableOp, size_t>& variablesPos,
          llvm::StringRef templateName);

      mlir::bmodelica::FunctionOp createPartialDerTemplateFromEquation(
          mlir::RewriterBase& rewriter,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          ScheduledEquationInstanceOp equationOp,
          llvm::DenseMap<VariableOp, size_t>& variablesPos,
          llvm::StringRef templateName);

      mlir::LogicalResult createJacobianFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          ScheduledEquationInstanceOp equationOp,
          llvm::StringRef jacobianFunctionName,
          const llvm::DenseMap<VariableOp, size_t>& variablesPos,
          VariableOp independentVariable,
          llvm::StringRef partialDerTemplateName);

      [[nodiscard]] std::string getKINSOLFunctionName(
          llvm::StringRef name) const;

    private:
      /// Instance identifier.
      /// It is used to create unique symbols.
      std::string identifier;

      mlir::SymbolTableCollection* symbolTableCollection;

      bool reducedDerivatives;
      bool debugInformation;

      /// The variables of the model that are managed by KINSOL.
      llvm::SmallVector<VariableOp> variables;

      /// The SSA values of the KINSOL variables.
      llvm::SmallVector<mlir::Value> kinsolVariables;

      /// Map used for a faster lookup of the variable position.
      llvm::DenseMap<VariableOp, size_t> variablesLookup;

      /// The equations managed by KINSOL.
      llvm::DenseSet<ScheduledEquationInstanceOp> equations;
  };
}

KINSOLInstance::KINSOLInstance(
    llvm::StringRef identifier,
    mlir::SymbolTableCollection& symbolTableCollection,
    bool reducedDerivatives,
    bool debugInformation)
    : identifier(identifier.str()),
      symbolTableCollection(&symbolTableCollection),
      reducedDerivatives(reducedDerivatives),
      debugInformation(debugInformation)
{
}

bool KINSOLInstance::hasVariable(VariableOp variable) const
{
  assert(variable != nullptr);
  return variablesLookup.contains(variable);
}

void KINSOLInstance::addVariable(VariableOp variable)
{
  assert(variable != nullptr);

  if (!hasVariable(variable)) {
    variables.push_back(variable);
    variablesLookup[variable] = variables.size() - 1;
  }
}

void KINSOLInstance::addEquation(ScheduledEquationInstanceOp equation)
{
  assert(equation != nullptr);
  equations.insert(equation);
}

mlir::LogicalResult KINSOLInstance::initialize(
    mlir::OpBuilder& builder,
    mlir::Location loc)
{
  // Initialize the instance.
  builder.create<mlir::kinsol::InitOp>(loc, identifier);

  return mlir::success();
}

mlir::LogicalResult KINSOLInstance::configure(
    mlir::RewriterBase& rewriter,
    mlir::Location loc,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variableOps)
{
  llvm::DenseMap<
      mlir::AffineMap,
      mlir::sundials::AccessFunctionOp> accessFunctionsMap;

  // Add the variables to KINSOL.
  if (mlir::failed(addVariablesToKINSOL(
          rewriter, loc, moduleOp, modelOp, variableOps))) {
    return mlir::failure();
  }

  // Add the equations to KINSOL.
  if (mlir::failed(addEquationsToKINSOL(
          rewriter, loc, moduleOp, modelOp, variableOps,
          accessFunctionsMap))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult KINSOLInstance::performSolve(
    mlir::OpBuilder& builder,
    mlir::Location loc)
{
  builder.create<mlir::kinsol::SolveOp>(loc, identifier);
  return mlir::success();
}

mlir::LogicalResult KINSOLInstance::deleteInstance(
    mlir::OpBuilder& builder,
    mlir::Location loc)
{
  builder.create<mlir::kinsol::FreeOp>(loc, identifier);
  return mlir::success();
}

mlir::LogicalResult KINSOLInstance::addVariablesToKINSOL(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variableOps)
{
  // Counters used to generate unique names for the getter and setter
  // functions.
  unsigned int getterFunctionCounter = 0;
  unsigned int setterFunctionCounter = 0;

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

  auto createGetterFn =
      [&](VariableOp variableOp) -> mlir::sundials::VariableGetterOp {
    std::string getterName = getKINSOLFunctionName(
        "getter_" + std::to_string(getterFunctionCounter++));

    return createGetterFunction(
        builder, loc, moduleOp, variableOp, getterName);
  };

  auto createSetterFn =
      [&](VariableOp variableOp) -> mlir::sundials::VariableSetterOp {
    std::string setterName = getKINSOLFunctionName(
        "setter_" + std::to_string(setterFunctionCounter++));

    return createSetterFunction(
        builder, loc, moduleOp, variableOp, setterName);
  };

  // Algebraic variables.
  for (VariableOp variableOp : variables) {
    auto arrayType = variableOp.getVariableType().toArrayType();

    std::vector<int64_t> dimensions = getDimensionsFn(arrayType);
    auto getter = createGetterFn(variableOp);
    auto setter = createSetterFn(variableOp);

    auto addVariableOp =
        builder.create<mlir::kinsol::AddVariableOp>(
            loc,
            identifier,
            builder.getI64ArrayAttr(dimensions),
            getter.getSymName(),
            setter.getSymName());

    if (debugInformation) {
      addVariableOp.setNameAttr(variableOp.getSymNameAttr());
    }

    kinsolVariables.push_back(addVariableOp);
  }

  return mlir::success();
}

mlir::sundials::VariableGetterOp KINSOLInstance::createGetterFunction(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    mlir::ModuleOp moduleOp,
    VariableOp variable,
    llvm::StringRef functionName)
{
  return ::mlir::bmodelica::createGetterFunction(
      builder, *symbolTableCollection, loc, moduleOp, variable, functionName);
}

mlir::sundials::VariableSetterOp KINSOLInstance::createSetterFunction(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    mlir::ModuleOp moduleOp,
    VariableOp variable,
    llvm::StringRef functionName)
{
  return ::mlir::bmodelica::createSetterFunction(
      builder, *symbolTableCollection, loc, moduleOp, variable, functionName);
}

mlir::LogicalResult KINSOLInstance::addEquationsToKINSOL(
    mlir::RewriterBase& rewriter,
    mlir::Location loc,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variableOps,
    llvm::DenseMap<
        mlir::AffineMap,
        mlir::sundials::AccessFunctionOp>& accessFunctionsMap)
{
  // Counters used to obtain unique names for the functions.
  size_t accessFunctionsCounter = 0;
  size_t residualFunctionsCounter = 0;
  size_t jacobianFunctionsCounter = 0;
  size_t partialDerTemplatesCounter = 0;

  llvm::DenseMap<VariableOp, mlir::Value> variablesMapping;

  for (const auto& [variable, kinsolVariable] :
       llvm::zip(variables, kinsolVariables)) {
    variablesMapping[variable] = kinsolVariable;
  }

  for (ScheduledEquationInstanceOp equationOp : equations) {
    // Keep track of the accessed variables in order to reduce the amount of
    // generated partial derivatives.
    llvm::SmallVector<VariableAccess> accesses;
    llvm::DenseSet<VariableOp> accessedVariables;

    if (mlir::failed(equationOp.getAccesses(
            accesses, *symbolTableCollection))) {
      return mlir::failure();
    }

    for (const VariableAccess& access : accesses) {
      auto variableOp =
          symbolTableCollection->lookupSymbolIn<VariableOp>(
              modelOp, access.getVariable());

      accessedVariables.insert(variableOp);
    }

    // Get the indices of the equation.
    IndexSet equationIndices = equationOp.getIterationSpace();

    if (equationIndices.empty()) {
      equationIndices = IndexSet(Point(0));
    }

    // Get the write access.
    auto writeAccess = equationOp.getMatchedAccess(*symbolTableCollection);

    if (!writeAccess) {
      return mlir::failure();
    }

    auto writtenVar = symbolTableCollection->lookupSymbolIn<VariableOp>(
        modelOp, writeAccess->getVariable());

    // Collect the independent variables for automatic differentiation.
    llvm::DenseSet<VariableOp> independentVariables;

    if (mlir::failed(getIndependentVariablesForAD(
            independentVariables, modelOp, equationOp))) {
      return mlir::failure();
    }

    // Create the partial derivative template.
    std::string partialDerTemplateName = getKINSOLFunctionName(
        "pder_" + std::to_string(partialDerTemplatesCounter++));

    llvm::DenseMap<VariableOp, size_t> variablesPos;

    if (mlir::failed(createPartialDerTemplateFunction(
            rewriter, moduleOp, modelOp, equationOp, variablesPos,
            partialDerTemplateName))) {
      return mlir::failure();
    }

    for (const MultidimensionalRange& range : llvm::make_range(
             equationIndices.rangesBegin(), equationIndices.rangesEnd())) {
      // Add the equation to the KINSOL instance.
      auto accessFunctionOp = getOrCreateAccessFunction(
          rewriter, equationOp.getLoc(), moduleOp,
          writeAccess->getAccessFunction().getAffineMap(),
          getKINSOLFunctionName("access"),
          accessFunctionsMap, accessFunctionsCounter);

      if (!accessFunctionOp) {
        return mlir::failure();
      }

      auto kinsolEquation = rewriter.create<mlir::kinsol::AddEquationOp>(
          equationOp.getLoc(),
          identifier,
          mlir::kinsol::MultidimensionalRangeAttr::get(
              rewriter.getContext(), range),
          variablesMapping[writtenVar],
          accessFunctionOp.getSymName());

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
                rewriter, loc, modelOp, equationOp, kinsolEquation,
                accessFunctionsMap, accessFunctionsCounter))) {
          return mlir::failure();
        }
      }

      // Create the residual function.
      std::string residualFunctionName = getKINSOLFunctionName(
          "residualFunction_" + std::to_string(residualFunctionsCounter++));

      if (mlir::failed(createResidualFunction(
              rewriter, moduleOp, modelOp, equationOp, kinsolEquation,
              residualFunctionName))) {
        return mlir::failure();
      }

      rewriter.create<mlir::kinsol::SetResidualOp>(
          loc, identifier, kinsolEquation, residualFunctionName);

      // Create the Jacobian functions.
      // Notice that Jacobian functions are not created for derivative
      // variables. Those are already handled when encountering the state
      // variable through the 'alpha' parameter set into the derivative seed.

      assert(variables.size() == kinsolVariables.size());

      for (auto [variable, kinsolVariable] :
           llvm::zip(variables, kinsolVariables)) {
        if (reducedDerivatives &&
            !accessedVariables.contains(variable)) {
          // The partial derivative is always zero.
          continue;
        }

        std::string jacobianFunctionName = getKINSOLFunctionName(
            "jacobianFunction_" + std::to_string(jacobianFunctionsCounter++));

        if (mlir::failed(createJacobianFunction(
                rewriter, moduleOp, modelOp, equationOp, jacobianFunctionName,
                variablesPos, variable, partialDerTemplateName))) {
          return mlir::failure();
        }

        rewriter.create<mlir::kinsol::AddJacobianOp>(
            loc, identifier, kinsolEquation, kinsolVariable,
            jacobianFunctionName);
      }
    }
  }

  return mlir::success();
}

mlir::LogicalResult KINSOLInstance::addVariableAccessesInfoToKINSOL(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    ModelOp modelOp,
    ScheduledEquationInstanceOp equationOp,
    mlir::Value kinsolEquation,
    llvm::DenseMap<
        mlir::AffineMap,
        mlir::sundials::AccessFunctionOp>& accessFunctionsMap,
    size_t& accessFunctionsCounter)
{
  auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();
  assert(kinsolEquation.getType().isa<mlir::kinsol::EquationType>());

  // Keep track of the discovered accesses in order to avoid adding the same
  // access map multiple times for the same variable.
  llvm::DenseMap<mlir::Value, llvm::DenseSet<mlir::AffineMap>> maps;

  llvm::SmallVector<VariableAccess> accesses;

  if (mlir::failed(equationOp.getAccesses(accesses, *symbolTableCollection))) {
    return mlir::failure();
  }

  for (const VariableAccess& access : accesses) {
    auto variableOp = symbolTableCollection->lookupSymbolIn<VariableOp>(
        modelOp, access.getVariable());

    if (!hasVariable(variableOp)) {
      continue;
    }

    mlir::Value kinsolVariable = kinsolVariables[variablesLookup[variableOp]];
    assert(kinsolVariable != nullptr);
    maps[kinsolVariable].insert(access.getAccessFunction().getAffineMap());
  }

  // Inform KINSOL about the discovered accesses.
  for (const auto& entry : maps) {
    mlir::Value kinsolVariable = entry.getFirst();

    for (mlir::AffineMap map : entry.getSecond()) {
      auto accessFunctionOp = getOrCreateAccessFunction(
          builder, loc, moduleOp, map,
          getKINSOLFunctionName("access"),
          accessFunctionsMap, accessFunctionsCounter);

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

mlir::sundials::AccessFunctionOp KINSOLInstance::getOrCreateAccessFunction(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    mlir::ModuleOp moduleOp,
    mlir::AffineMap access,
    llvm::StringRef functionNamePrefix,
    llvm::DenseMap<
        mlir::AffineMap,
        mlir::sundials::AccessFunctionOp>& accessFunctionsMap,
    size_t& accessFunctionsCounter)
{
  auto it = accessFunctionsMap.find(access);

  if (it == accessFunctionsMap.end()) {
    std::string functionName =
        functionNamePrefix.str() + "_" +
        std::to_string(accessFunctionsCounter++);

    auto accessFunctionOp = createAccessFunction(
        builder, loc, moduleOp, access, functionName);

    if (!accessFunctionOp) {
      return nullptr;
    }

    accessFunctionsMap[access] = accessFunctionOp;
    return accessFunctionOp;
  }

  return it->getSecond();
}

mlir::sundials::AccessFunctionOp KINSOLInstance::createAccessFunction(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    mlir::ModuleOp moduleOp,
    mlir::AffineMap access,
    llvm::StringRef functionName)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  // Normalize the access so that it has at least one dimension and one result.
  llvm::SmallVector<mlir::AffineExpr> expressions;

  for (mlir::AffineExpr expression : access.getResults()) {
    expressions.push_back(expression);
  }

  if (expressions.empty()) {
    expressions.push_back(
        mlir::getAffineConstantExpr(0, builder.getContext()));
  }

  auto extendedAccess = mlir::AffineMap::get(
      std::max(static_cast<unsigned int>(1), access.getNumDims()),
      access.getNumSymbols(),
      expressions, builder.getContext());

  // Create the operation for the access function.
  auto accessFunctionOp = builder.create<mlir::sundials::AccessFunctionOp>(
      loc,
      functionName,
      extendedAccess.getNumDims(),
      extendedAccess.getNumResults());

  symbolTableCollection->getSymbolTable(moduleOp).insert(accessFunctionOp);

  mlir::Block* bodyBlock = accessFunctionOp.addEntryBlock();
  builder.setInsertionPointToStart(bodyBlock);

  // Materialize the access.
  llvm::SmallVector<mlir::Value, 3> results;

  if (mlir::failed(materializeAffineMap(
          builder, loc, extendedAccess,
          accessFunctionOp.getEquationIndices(), results))) {
    return nullptr;
  }

  builder.create<mlir::sundials::ReturnOp>(loc, results);
  return accessFunctionOp;
}

mlir::LogicalResult KINSOLInstance::createResidualFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    ScheduledEquationInstanceOp equationOp,
    mlir::Value kinsolEquation,
    llvm::StringRef residualFunctionName)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  mlir::Location loc = equationOp.getLoc();

  size_t numOfInductionVariables = equationOp.getInductionVariables().size();

  auto residualFunction = builder.create<mlir::kinsol::ResidualFunctionOp>(
      loc,
      residualFunctionName,
      numOfInductionVariables);

  symbolTableCollection->getSymbolTable(moduleOp).insert(residualFunction);

  mlir::Block* bodyBlock = residualFunction.addEntryBlock();
  builder.setInsertionPointToStart(bodyBlock);

  // Map for the SSA values.
  mlir::IRMapping mapping;

  // Map the iteration variables.
  auto originalInductions = equationOp.getInductionVariables();
  auto mappedInductions = residualFunction.getEquationIndices();
  assert(originalInductions.size() <= mappedInductions.size());

  llvm::SmallVector<mlir::Value, 3> implicitInductions(
      std::next(mappedInductions.begin(), originalInductions.size()),
      mappedInductions.end());

  for (size_t i = 0, e = originalInductions.size(); i < e; ++i) {
    mapping.map(originalInductions[i], mappedInductions[i]);
  }

  for (auto& op : equationOp.getTemplate().getOps()) {
    if (mlir::isa<EquationSideOp>(op)) {
      continue;
    } else if (auto equationSidesOp = mlir::dyn_cast<EquationSidesOp>(op)) {
      // Compute the difference between the right-hand side and the left-hand
      // side of the equation.
      mlir::Value lhs = mapping.lookup(equationSidesOp.getLhsValues()[0]);
      mlir::Value rhs = mapping.lookup(equationSidesOp.getRhsValues()[0]);

      if (lhs.getType().isa<ArrayType>()) {
        assert(lhs.getType().cast<ArrayType>().getRank() ==
               static_cast<int64_t>(implicitInductions.size()));

        lhs = builder.create<LoadOp>(lhs.getLoc(), lhs, implicitInductions);
      }

      if (rhs.getType().isa<ArrayType>()) {
        assert(rhs.getType().cast<ArrayType>().getRank() ==
               static_cast<int64_t>(implicitInductions.size()));

        rhs = builder.create<LoadOp>(rhs.getLoc(), rhs, implicitInductions);
      }

      mlir::Value difference = builder.create<SubOp>(
          loc, builder.getF64Type(), rhs, lhs);

      builder.create<mlir::kinsol::ReturnOp>(difference.getLoc(), difference);
    } else if (auto variableGetOp = mlir::dyn_cast<VariableGetOp>(op)) {
      // Replace the local variables with the global ones.
      auto variableOp = symbolTableCollection->lookupSymbolIn<VariableOp>(
          modelOp, variableGetOp.getVariableAttr());

      auto getOp = builder.create<QualifiedVariableGetOp>(
          variableGetOp.getLoc(), variableOp);

      mapping.map(variableGetOp.getResult(), getOp.getResult());
    } else {
      builder.clone(op, mapping);
    }
  }

  return mlir::success();
}

mlir::LogicalResult KINSOLInstance::getIndependentVariablesForAD(
    llvm::DenseSet<VariableOp>& result,
    ModelOp modelOp,
    ScheduledEquationInstanceOp equationOp)
{
  llvm::SmallVector<VariableAccess> accesses;

  if (mlir::failed(equationOp.getAccesses(accesses, *symbolTableCollection))) {
    return mlir::failure();
  }

  for (const VariableAccess& access : accesses) {
    auto variableOp =
        symbolTableCollection->lookupSymbolIn<VariableOp>(
            modelOp, access.getVariable());

    result.insert(variableOp);
  }

  return mlir::success();
}

mlir::LogicalResult KINSOLInstance::createPartialDerTemplateFunction(
    mlir::RewriterBase& rewriter,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    ScheduledEquationInstanceOp equationOp,
    llvm::DenseMap<VariableOp, size_t>& variablesPos,
    llvm::StringRef templateName)
{
  auto partialDerTemplate = createPartialDerTemplateFromEquation(
      rewriter, moduleOp, modelOp, equationOp, variablesPos, templateName);

  if (!partialDerTemplate) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::bmodelica::FunctionOp
KINSOLInstance::createPartialDerTemplateFromEquation(
    mlir::RewriterBase& rewriter,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    ScheduledEquationInstanceOp equationOp,
    llvm::DenseMap<VariableOp, size_t>& variablesPos,
    llvm::StringRef templateName)
{
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  mlir::Location loc = equationOp.getLoc();

  // Create the function.
  std::string functionOpName = templateName.str() + "_base";

  // Create the function to be derived.
  auto functionOp = rewriter.create<FunctionOp>(loc, functionOpName);
  rewriter.createBlock(&functionOp.getBodyRegion());

  // Start the body of the function.
  rewriter.setInsertionPointToStart(functionOp.getBody());

  // Determine the variables needed by the equation.
  llvm::SmallVector<VariableAccess> accesses;

  if (mlir::failed(equationOp.getAccesses(accesses, *symbolTableCollection))) {
    return nullptr;
  }

  llvm::DenseSet<VariableOp> accessedVariables;

  for (const VariableAccess& access : accesses) {
    auto variableOp = symbolTableCollection->lookupSymbolIn<VariableOp>(
        modelOp, access.getVariable());

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

    localVariableOps[variableOp.getSymName()] = clonedVariableOp;
    variablesPos[variableOp] = variableIndex++;
  }

  // Create the induction variables.
  llvm::SmallVector<VariableOp, 3> inductionVariablesOps;

  size_t numOfInductions = equationOp.getInductionVariables().size();

  for (size_t i = 0; i < numOfInductions; ++i) {
    std::string variableName = "ind" + std::to_string(i);

    auto variableType = VariableType::wrap(
        rewriter.getIndexType(),
        VariabilityProperty::none,
        IOProperty::input);

    auto variableOp = rewriter.create<VariableOp>(
        loc, variableName, variableType);

    inductionVariablesOps.push_back(variableOp);
  }

  // Create the output variable, that is the difference between its equation
  // right-hand side value and its left-hand side value.
  std::string outVariableName = "out";
  size_t outVariableNameCounter = 0;

  while (symbolTableCollection->lookupSymbolIn(
      functionOp, rewriter.getStringAttr(outVariableName))) {
    outVariableName = "out_" + std::to_string(outVariableNameCounter++);
  }

  auto outputVariableOp = rewriter.create<VariableOp>(
      loc, outVariableName,
      VariableType::wrap(
          RealType::get(rewriter.getContext()),
          VariabilityProperty::none,
          IOProperty::output));

  // Create the body of the function.
  auto algorithmOp = rewriter.create<AlgorithmOp>(loc);

  rewriter.setInsertionPointToStart(
      rewriter.createBlock(&algorithmOp.getBodyRegion()));

  mlir::IRMapping mapping;

  // Get the values of the induction variables.
  auto originalInductions = equationOp.getInductionVariables();
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
  llvm::DenseSet<mlir::Operation*> toBeCloned;
  llvm::SmallVector<mlir::Operation*> toBeClonedVisitStack;

  auto equationSidesOp = mlir::cast<EquationSidesOp>(
      equationOp.getTemplate().getBody()->getTerminator());

  mlir::Value lhs = equationSidesOp.getLhsValues()[0];
  mlir::Value rhs = equationSidesOp.getRhsValues()[0];

  if (mlir::Operation* lhsOp = lhs.getDefiningOp()) {
    toBeClonedVisitStack.push_back(lhsOp);
  }

  if (mlir::Operation* rhsOp = rhs.getDefiningOp()) {
    toBeClonedVisitStack.push_back(rhsOp);
  }

  while (!toBeClonedVisitStack.empty()) {
    mlir::Operation* op = toBeClonedVisitStack.pop_back_val();
    toBeCloned.insert(op);

    for (mlir::Value operand : op->getOperands()) {
      if (auto operandOp = operand.getDefiningOp()) {
        toBeClonedVisitStack.push_back(operandOp);
      }
    }
  }

  // Clone the original operations and compute the residual value.
  for (auto& op : equationOp.getTemplate().getOps()) {
    if (!toBeCloned.contains(&op)) {
      continue;
    }

    if (auto globalGetOp = mlir::dyn_cast<GlobalVariableGetOp>(op)) {
      VariableOp variableOp = localVariableOps[globalGetOp.getVariable()];

      auto getOp = rewriter.create<VariableGetOp>(
          globalGetOp.getLoc(), variableOp);

      mapping.map(globalGetOp.getResult(), getOp.getResult());
    } else if (mlir::isa<EquationSideOp, EquationSidesOp>(op)) {
      continue;
    } else {
      rewriter.clone(op, mapping);
    }
  }

  mlir::Value mappedLhs = mapping.lookup(lhs);
  mlir::Value mappedRhs = mapping.lookup(rhs);

  auto result = rewriter.create<SubOp>(
      loc, RealType::get(rewriter.getContext()), mappedRhs, mappedLhs);

  rewriter.create<VariableSetOp>(
      loc, outputVariableOp.getSymName(), result);

  // Create the derivative template function.
  ForwardAD forwardAD;

  auto derTemplate = forwardAD.createPartialDerTemplateFunction(
      rewriter, loc, *symbolTableCollection, functionOp, templateName);

  rewriter.eraseOp(functionOp);

  // Replace the local variables with the global ones.
  llvm::DenseSet<VariableOp> variablesToBeReplaced;

  for (VariableOp variableOp : derTemplate.getVariables()) {
    if (localVariableOps.count(variableOp.getSymName()) != 0) {
      variablesToBeReplaced.insert(variableOp);
    }
  }

  llvm::SmallVector<VariableGetOp> variableGetOps;

  derTemplate.walk([&](VariableGetOp getOp) {
    if (localVariableOps.count(getOp.getVariable()) != 0) {
      variableGetOps.push_back(getOp);
    }
  });

  for (VariableGetOp getOp : variableGetOps) {
    rewriter.setInsertionPoint(getOp);

    auto variableOp = symbolTableCollection->lookupSymbolIn<VariableOp>(
        modelOp, getOp.getVariableAttr());

    mlir::Value globalVariable = rewriter.create<QualifiedVariableGetOp>(
        getOp.getLoc(), variableOp);

    rewriter.replaceOp(getOp, globalVariable);
  }

  for (VariableOp variableOp : variablesToBeReplaced) {
    rewriter.eraseOp(variableOp);
  }

  return derTemplate;
}

mlir::LogicalResult KINSOLInstance::createJacobianFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    ScheduledEquationInstanceOp equationOp,
    llvm::StringRef jacobianFunctionName,
    const llvm::DenseMap<VariableOp, size_t>& variablesPos,
    VariableOp independentVariable,
    llvm::StringRef partialDerTemplateName)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  mlir::Location loc = equationOp.getLoc();

  size_t numOfVars = variablesPos.size();
  size_t numOfInductions = equationOp.getInductionVariables().size();

  // Create the function.
  auto jacobianFunction = builder.create<mlir::kinsol::JacobianFunctionOp>(
      loc, jacobianFunctionName, numOfInductions,
      independentVariable.getVariableType().getRank());

  symbolTableCollection->getSymbolTable(moduleOp).insert(jacobianFunction);

  mlir::Block* bodyBlock = jacobianFunction.addEntryBlock();
  builder.setInsertionPointToStart(bodyBlock);

  // Create the global seeds for the variables.
  llvm::SmallVector<GlobalVariableOp> varSeeds(numOfVars, nullptr);
  size_t seedsCounter = 0;

  for (const auto& entry : variablesPos) {
    VariableOp variableOp = entry.getFirst();
    size_t pos = entry.getSecond();

    std::string seedName = jacobianFunctionName.str() + "_seed_" +
        std::to_string(seedsCounter++);

    assert(varSeeds[pos] == nullptr && "Seed already created");

    auto seed = createGlobalADSeed(
        builder, moduleOp, loc, seedName,
        variableOp.getVariableType().toArrayType());

    if (!seed) {
      return mlir::failure();
    }

    varSeeds[pos] = seed;
  }

  assert(llvm::none_of(varSeeds, [](GlobalVariableOp seed) {
           return seed == nullptr;
         }) && "Some seeds have not been created");

  // Zero and one constants to be used to update the array seeds or for the
  // scalar seeds.
  mlir::Value zero = builder.create<ConstantOp>(
      loc, RealAttr::get(builder.getContext(), 0));

  mlir::Value one = builder.create<ConstantOp>(
      loc, RealAttr::get(builder.getContext(), 1));

  // Function to collect the arguments to be passed to the derivative template
  // function.
  auto collectArgsFn = [&](llvm::SmallVectorImpl<mlir::Value>& args) {
    // Equation indices.
    for (mlir::Value equationIndex : jacobianFunction.getEquationIndices()) {
      args.push_back(equationIndex);
    }

    // Seeds of the variables.
    for (GlobalVariableOp globalSeed : varSeeds) {
      mlir::Value seed = builder.create<GlobalVariableGetOp>(loc, globalSeed);

      if (seed.getType().cast<ArrayType>().isScalar()) {
        seed = builder.create<LoadOp>(loc, seed, std::nullopt);
      }

      args.push_back(seed);
    }

    // Seeds of the equation indices. They are all equal to zero.
    for (size_t i = 0; i < jacobianFunction.getEquationIndices().size(); ++i) {
      args.push_back(zero);
    }
  };

  llvm::SmallVector<mlir::Value> args;

  // Perform the first call to the template function.
  assert(variablesPos.count(independentVariable) != 0);

  // Set the seed of the variable to one.
  size_t oneSeedPosition = variablesPos.lookup(independentVariable);

  setGlobalADSeed(builder, loc, varSeeds[oneSeedPosition],
                  jacobianFunction.getVariableIndices(), one);

  // Call the template function.
  args.clear();
  collectArgsFn(args);

  auto firstTemplateCall = builder.create<CallOp>(
      loc,
      mlir::SymbolRefAttr::get(builder.getContext(), partialDerTemplateName),
      RealType::get(builder.getContext()),
      args);

  mlir::Value result = firstTemplateCall.getResult(0);

  // Reset the seed of the variable.
  setGlobalADSeed(builder, loc, varSeeds[oneSeedPosition],
                  jacobianFunction.getVariableIndices(), zero);

  // Return the result.
  result = builder.create<CastOp>(loc, builder.getF64Type(), result);
  builder.create<mlir::kinsol::ReturnOp>(loc, result);

  return mlir::success();
}

std::string KINSOLInstance::getKINSOLFunctionName(llvm::StringRef name) const
{
  return identifier + "_" + name.str();
}

namespace
{
  class SCCSolvingWithKINSOLPass
      : public mlir::bmodelica::impl::SCCSolvingWithKINSOLPassBase<
            SCCSolvingWithKINSOLPass>
  {
    public:
      using SCCSolvingWithKINSOLPassBase<SCCSolvingWithKINSOLPass>
          ::SCCSolvingWithKINSOLPassBase;

      void runOnOperation() override;

    private:
      mlir::LogicalResult processModelOp(
          mlir::RewriterBase& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp);

      mlir::LogicalResult processInitialModelOp(
          mlir::RewriterBase& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          InitialModelOp initialModelOp);

      mlir::LogicalResult processMainModelOp(
          mlir::RewriterBase& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          MainModelOp mainModelOp);

      mlir::LogicalResult processSCC(
          mlir::RewriterBase& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          SCCOp scc,
          llvm::function_ref<mlir::Block*(
              mlir::OpBuilder&, mlir::Location)> createBeginFn,
          llvm::function_ref<mlir::Block*(
              mlir::OpBuilder&, mlir::Location)> createEndFn);

      mlir::kinsol::InstanceOp declareInstance(
          mlir::OpBuilder& builder,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::Location loc,
          mlir::ModuleOp moduleOp);

      mlir::LogicalResult getAccessAttrs(
          llvm::SmallVectorImpl<mlir::Attribute>& writtenVariables,
          llvm::SmallVectorImpl<mlir::Attribute>& readVariables,
          mlir::SymbolTableCollection& symbolTableCollection,
          SCCOp scc);

      mlir::LogicalResult createBeginFunction(
          mlir::RewriterBase& rewriter,
          mlir::ModuleOp moduleOp,
          ModelOp modelOp,
          mlir::Location loc,
          KINSOLInstance* kinsolInstance,
          llvm::ArrayRef<VariableOp> variables,
          llvm::function_ref<mlir::Block*(
              mlir::OpBuilder&, mlir::Location)> createBeginFn) const;

      mlir::LogicalResult createEndFunction(
          mlir::RewriterBase& rewriter,
          mlir::ModuleOp moduleOp,
          mlir::Location loc,
          KINSOLInstance* kinsolInstance,
          llvm::function_ref<mlir::Block*(
              mlir::OpBuilder&, mlir::Location)> createEndFn) const;

      mlir::LogicalResult cleanModelOp(ModelOp modelOp);
  };
}

void SCCSolvingWithKINSOLPass::runOnOperation()
{
  mlir::ModuleOp moduleOp = getOperation();

  mlir::IRRewriter rewriter(&getContext());
  mlir::SymbolTableCollection symbolTableCollection;
  llvm::SmallVector<ModelOp> modelOps;

  for (ModelOp modelOp : moduleOp.getOps<ModelOp>()) {
    modelOps.push_back(modelOp);
  }

  for (ModelOp modelOp : modelOps) {
    if (mlir::failed(processModelOp(
            rewriter, symbolTableCollection, moduleOp, modelOp))) {
      return signalPassFailure();
    }

    if (mlir::failed(cleanModelOp(modelOp))) {
      return signalPassFailure();
    }
  }

  // Determine the analyses to be preserved.
  markAnalysesPreserved<DerivativesMap>();
}

mlir::LogicalResult SCCSolvingWithKINSOLPass::processModelOp(
    mlir::RewriterBase& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp)
{
  for (ScheduleOp scheduleOp : modelOp.getOps<ScheduleOp>()) {
    for (InitialModelOp initialModelOp : scheduleOp.getOps<InitialModelOp>()) {
      if (mlir::failed(processInitialModelOp(
              rewriter, symbolTableCollection, moduleOp, modelOp,
              initialModelOp))) {
        return mlir::failure();
      }
    }

    for (MainModelOp mainModelOp : scheduleOp.getOps<MainModelOp>()) {
      if (mlir::failed(processMainModelOp(
              rewriter, symbolTableCollection, moduleOp, modelOp,
              mainModelOp))) {
        return mlir::failure();
      }
    }
  }

  return mlir::success();
}

mlir::LogicalResult SCCSolvingWithKINSOLPass::processInitialModelOp(
    mlir::RewriterBase& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    InitialModelOp initialModelOp)
{
  auto createBeginFn =
      [&](mlir::OpBuilder& builder, mlir::Location loc) -> mlir::Block* {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(moduleOp.getBody());
        auto beginFn = builder.create<mlir::runtime::ICModelBeginOp>(loc);
        return builder.createBlock(&beginFn.getBodyRegion());
      };

  auto createEndFn =
      [&](mlir::OpBuilder& builder, mlir::Location loc) -> mlir::Block* {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(moduleOp.getBody());
        auto beginFn = builder.create<mlir::runtime::ICModelEndOp>(loc);
        return builder.createBlock(&beginFn.getBodyRegion());
  };

  llvm::SmallVector<SCCOp> SCCs;

  for (SCCOp scc : initialModelOp.getOps<SCCOp>()) {
    SCCs.push_back(scc);
  }

  for (SCCOp scc : SCCs) {
    if (mlir::succeeded(processSCC(
            rewriter, symbolTableCollection, moduleOp, modelOp, scc,
            createBeginFn, createEndFn))) {
      rewriter.eraseOp(scc);
    }
  }

  return mlir::success();
}

mlir::LogicalResult SCCSolvingWithKINSOLPass::processMainModelOp(
    mlir::RewriterBase& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    MainModelOp mainModelOp)
{
  auto createBeginFn =
      [&](mlir::OpBuilder& builder, mlir::Location loc) -> mlir::Block* {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(moduleOp.getBody());
        auto beginFn = builder.create<mlir::runtime::ICModelBeginOp>(loc);
        return builder.createBlock(&beginFn.getBodyRegion());
      };

  auto createEndFn =
      [&](mlir::OpBuilder& builder, mlir::Location loc) -> mlir::Block* {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(moduleOp.getBody());
        auto beginFn = builder.create<mlir::runtime::ICModelEndOp>(loc);
        return builder.createBlock(&beginFn.getBodyRegion());
      };

  llvm::SmallVector<SCCOp> SCCs;

  for (SCCOp scc : mainModelOp.getOps<SCCOp>()) {
    SCCs.push_back(scc);
  }

  for (SCCOp scc : SCCs) {
    if (mlir::succeeded(processSCC(
            rewriter, symbolTableCollection, moduleOp, modelOp, scc,
            createBeginFn, createEndFn))) {
      rewriter.eraseOp(scc);
    }
  }

  return mlir::success();
}

mlir::LogicalResult SCCSolvingWithKINSOLPass::processSCC(
    mlir::RewriterBase& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    SCCOp scc,
    llvm::function_ref<mlir::Block*(
        mlir::OpBuilder&, mlir::Location)> createBeginFn,
    llvm::function_ref<mlir::Block*(
        mlir::OpBuilder&, mlir::Location)> createEndFn)
{
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  LLVM_DEBUG({
     llvm::dbgs() << "Processing SCC composed by:\n";

     for (ScheduledEquationInstanceOp equation :
          scc.getOps<ScheduledEquationInstanceOp>()) {
       llvm::dbgs() << "  ";
       equation.printInline(llvm::dbgs());
       llvm::dbgs() << "\n";
     }
  });

  llvm::SmallVector<mlir::Attribute> writtenVariables;
  llvm::SmallVector<mlir::Attribute> readVariables;

  if (mlir::failed(getAccessAttrs(
          writtenVariables, readVariables, symbolTableCollection, scc))) {
    return mlir::failure();
  }

  llvm::SmallVector<VariableOp> variables;

  // Check if there are unwritten indices among the externalized variables.
  for (mlir::Attribute attr : writtenVariables) {
    auto variableAttr = attr.cast<VariableAttr>();
    auto writtenVariableName = variableAttr.cast<VariableAttr>().getName();

    auto writtenVariableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
        modelOp, writtenVariableName);

    if (variableAttr.getIndices().getValue() != writtenVariableOp.getIndices()) {
      return mlir::failure();
    }

    variables.push_back(writtenVariableOp);
  }

  auto instanceOp = declareInstance(
      rewriter, symbolTableCollection, modelOp.getLoc(), moduleOp);

  auto kinsolInstance = std::make_unique<KINSOLInstance>(
      instanceOp.getSymName(), symbolTableCollection, reducedDerivatives,
      debugInformation);

  // Add the variables to KINSOL.
  for (VariableOp variable : variables) {
    LLVM_DEBUG(llvm::dbgs() << "Add variable: " << variable.getSymName()
                            << "\n");

    kinsolInstance->addVariable(variable);
  }

  for (ScheduledEquationInstanceOp equation :
       scc.getOps<ScheduledEquationInstanceOp>()) {
    LLVM_DEBUG({
      llvm::dbgs() << "Add equation\n";
      equation.printInline(llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    kinsolInstance->addEquation(equation);

    std::optional<VariableAccess> writeAccess =
        equation.getMatchedAccess(symbolTableCollection);

    if (!writeAccess) {
      return mlir::failure();
    }

    auto writtenVariableOp =
        symbolTableCollection.lookupSymbolIn<VariableOp>(
            modelOp, writeAccess->getVariable());

    LLVM_DEBUG(llvm::dbgs() << "Add algebraic variable: "
                            << writtenVariableOp.getSymName() << "\n");

    kinsolInstance->addVariable(writtenVariableOp);
  }

  if (mlir::failed(createBeginFunction(
          rewriter, moduleOp, modelOp, scc.getLoc(),
          kinsolInstance.get(), variables, createBeginFn))) {
    return mlir::failure();
  }

  if (mlir::failed(createEndFunction(
          rewriter, moduleOp, modelOp.getLoc(), kinsolInstance.get(),
          createEndFn))) {
    return mlir::failure();
  }

  rewriter.setInsertionPoint(scc);

  auto scheduleBlockOp = rewriter.create<ScheduleBlockOp>(
      scc.getLoc(), false,
      rewriter.getArrayAttr(writtenVariables),
      rewriter.getArrayAttr(readVariables));

  rewriter.createBlock(&scheduleBlockOp.getBodyRegion());
  rewriter.setInsertionPointToStart(scheduleBlockOp.getBody());

  if (mlir::failed(kinsolInstance->performSolve(rewriter, scc.getLoc()))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::kinsol::InstanceOp SCCSolvingWithKINSOLPass::declareInstance(
    mlir::OpBuilder& builder,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::Location loc,
    mlir::ModuleOp moduleOp)
{
  // Create the instance.
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());

  auto instanceOp = builder.create<mlir::kinsol::InstanceOp>(loc, "kinsol");

  // Update the symbol table.
  symbolTableCollection.getSymbolTable(moduleOp).insert(instanceOp);

  return instanceOp;
}

mlir::LogicalResult SCCSolvingWithKINSOLPass::getAccessAttrs(
    llvm::SmallVectorImpl<mlir::Attribute>& writtenVariables,
    llvm::SmallVectorImpl<mlir::Attribute>& readVariables,
    mlir::SymbolTableCollection& symbolTableCollection,
    SCCOp scc)
{
  llvm::DenseMap<mlir::SymbolRefAttr, IndexSet> writtenVariablesIndices;
  llvm::DenseMap<mlir::SymbolRefAttr, IndexSet> readVariablesIndices;

  for (ScheduledEquationInstanceOp equationOp :
       scc.getOps<ScheduledEquationInstanceOp>()) {
    IndexSet equationIndices = equationOp.getIterationSpace();
    auto matchedAccess = equationOp.getMatchedAccess(symbolTableCollection);

    if (!matchedAccess) {
      return mlir::failure();
    }

    IndexSet matchedVariableIndices =
        matchedAccess->getAccessFunction().map(equationIndices);

    writtenVariablesIndices[matchedAccess->getVariable()] +=
        matchedVariableIndices;

    llvm::SmallVector<VariableAccess> accesses;
    llvm::SmallVector<VariableAccess> readAccesses;

    if (mlir::failed(equationOp.getAccesses(
            accesses, symbolTableCollection))) {
      return mlir::failure();
    }

    if (mlir::failed(equationOp.getReadAccesses(
            readAccesses, symbolTableCollection, accesses))) {
      return mlir::failure();
    }

    for (const VariableAccess& readAccess : readAccesses) {
      const AccessFunction& accessFunction = readAccess.getAccessFunction();
      IndexSet readIndices = accessFunction.map(equationIndices);
      readVariablesIndices[readAccess.getVariable()] += readIndices;
    }
  }

  for (const auto& entry : writtenVariablesIndices) {
    writtenVariables.push_back(VariableAttr::get(
        &getContext(),
        entry.getFirst(),
        IndexSetAttr::get(&getContext(), entry.getSecond())));
  }

  for (const auto& entry : readVariablesIndices) {
    readVariables.push_back(VariableAttr::get(
        &getContext(),
        entry.getFirst(),
        IndexSetAttr::get(&getContext(), entry.getSecond())));
  }

  return mlir::success();
}

mlir::LogicalResult SCCSolvingWithKINSOLPass::createBeginFunction(
    mlir::RewriterBase& rewriter,
    mlir::ModuleOp moduleOp,
    ModelOp modelOp,
    mlir::Location loc,
    KINSOLInstance* kinsolInstance,
    llvm::ArrayRef<VariableOp> variables,
    llvm::function_ref<mlir::Block*(
        mlir::OpBuilder&, mlir::Location)> createBeginFn) const
{
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  mlir::Block* bodyBlock = createBeginFn(rewriter, loc);
  rewriter.setInsertionPointToStart(bodyBlock);

  if (mlir::failed(kinsolInstance->initialize(rewriter, loc))) {
    return mlir::failure();
  }

  if (mlir::failed(kinsolInstance->configure(
          rewriter, loc, moduleOp, modelOp, variables))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult SCCSolvingWithKINSOLPass::createEndFunction(
    mlir::RewriterBase& rewriter,
    mlir::ModuleOp moduleOp,
    mlir::Location loc,
    KINSOLInstance* kinsolInstance,
    llvm::function_ref<mlir::Block*(
        mlir::OpBuilder&, mlir::Location)> createEndFn) const
{
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  mlir::Block* bodyBlock = createEndFn(rewriter, loc);
  rewriter.setInsertionPointToStart(bodyBlock);

  if (mlir::failed(kinsolInstance->deleteInstance(rewriter, loc))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult SCCSolvingWithKINSOLPass::cleanModelOp(ModelOp modelOp)
{
  mlir::RewritePatternSet patterns(&getContext());
  ModelOp::getCleaningPatterns(patterns, &getContext());
  return mlir::applyPatternsAndFoldGreedily(modelOp, std::move(patterns));
}

namespace mlir::bmodelica
{
  std::unique_ptr<mlir::Pass> createSCCSolvingWithKINSOLPass()
  {
    return std::make_unique<SCCSolvingWithKINSOLPass>();
  }

  std::unique_ptr<mlir::Pass> createSCCSolvingWithKINSOLPass(
      const SCCSolvingWithKINSOLPassOptions& options)
  {
    return std::make_unique<SCCSolvingWithKINSOLPass>(options);
  }
}
