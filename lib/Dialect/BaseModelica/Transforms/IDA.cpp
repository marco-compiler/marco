#include "marco/Dialect/BaseModelica/Transforms/IDA.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/AutomaticDifferentiation/ForwardAD.h"
#include "marco/Dialect/BaseModelica/Transforms/Solvers/SUNDIALS.h"
#include "marco/Dialect/IDA/IR/IDA.h"
#include "marco/Dialect/Runtime/IR/Runtime.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ida"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_IDAPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class IDAInstance {
public:
  IDAInstance(llvm::StringRef identifier,
              mlir::SymbolTableCollection &symbolTableCollection,
              const DerivativesMap *derivativesMap, bool reducedSystem,
              bool reducedDerivatives, bool jacobianOneSweep,
              bool debugInformation);

  bool hasVariable(VariableOp variable) const;

  void addStateVariable(VariableOp variable);

  void addDerivativeVariable(VariableOp variable);

  void addAlgebraicVariable(VariableOp variable);

  bool hasEquation(MatchedEquationInstanceOp equation) const;

  void addEquation(MatchedEquationInstanceOp equation);

  mlir::LogicalResult declareInstance(mlir::OpBuilder &builder,
                                      mlir::Location loc,
                                      mlir::ModuleOp moduleOp);

  mlir::LogicalResult initialize(mlir::OpBuilder &builder, mlir::Location loc);

  mlir::LogicalResult configure(mlir::IRRewriter &rewriter, mlir::Location loc,
                                mlir::ModuleOp moduleOp, ModelOp modelOp,
                                llvm::ArrayRef<VariableOp> variableOps,
                                llvm::ArrayRef<SCCOp> allSCCs);

  mlir::LogicalResult performCalcIC(mlir::OpBuilder &builder,
                                    mlir::Location loc);

  mlir::LogicalResult performStep(mlir::OpBuilder &builder, mlir::Location loc);

  mlir::Value getCurrentTime(mlir::OpBuilder &builder, mlir::Location loc);

  mlir::LogicalResult deleteInstance(mlir::OpBuilder &builder,
                                     mlir::Location loc);

private:
  bool hasAlgebraicVariable(VariableOp variable) const;

  bool hasStateVariable(VariableOp variable) const;

  bool hasDerivativeVariable(VariableOp variable) const;

  mlir::LogicalResult addVariablesToIDA(mlir::OpBuilder &builder,
                                        mlir::Location loc,
                                        mlir::ModuleOp moduleOp,
                                        ModelOp modelOp,
                                        llvm::ArrayRef<VariableOp> variableOps);

  mlir::sundials::VariableGetterOp
  createGetterFunction(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::ModuleOp moduleOp, VariableOp variable,
                       llvm::StringRef functionName);

  mlir::sundials::VariableSetterOp
  createSetterFunction(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::ModuleOp moduleOp, VariableOp variable,
                       llvm::StringRef functionName);

  mlir::LogicalResult addEquationsToIDA(
      mlir::IRRewriter &rewriter, mlir::Location loc, mlir::ModuleOp moduleOp,
      ModelOp modelOp, llvm::ArrayRef<VariableOp> variableOps,
      llvm::ArrayRef<SCCOp> SCCs,
      llvm::DenseMap<mlir::AffineMap, mlir::sundials::AccessFunctionOp>
          &accessFunctionsMap);

  mlir::LogicalResult addVariableAccessesInfoToIDA(
      mlir::OpBuilder &builder, mlir::Location loc, ModelOp modelOp,
      MatchedEquationInstanceOp equationOp, mlir::Value idaEquation,
      llvm::DenseMap<mlir::AffineMap, mlir::sundials::AccessFunctionOp>
          &accessFunctionsMap,
      size_t &accessFunctionsCounter);

  mlir::sundials::AccessFunctionOp getOrCreateAccessFunction(
      mlir::OpBuilder &builder, mlir::Location loc, mlir::ModuleOp moduleOp,
      mlir::AffineMap access, llvm::StringRef functionNamePrefix,
      llvm::DenseMap<mlir::AffineMap, mlir::sundials::AccessFunctionOp>
          &accessFunctionsMap,
      size_t &accessFunctionsCounter);

  mlir::sundials::AccessFunctionOp
  createAccessFunction(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::ModuleOp moduleOp, mlir::AffineMap access,
                       llvm::StringRef functionName);

  mlir::LogicalResult
  createResidualFunction(mlir::RewriterBase &rewriter, mlir::ModuleOp moduleOp,
                         ModelOp modelOp, MatchedEquationInstanceOp equationOp,
                         mlir::Value idaEquation,
                         llvm::StringRef residualFunctionName);

  mlir::LogicalResult
  getIndependentVariablesForAD(llvm::DenseSet<VariableOp> &result,
                               ModelOp modelOp,
                               MatchedEquationInstanceOp equationOp);

  mlir::LogicalResult createPartialDerTemplateFunction(
      mlir::IRRewriter &rewriter, mlir::ModuleOp moduleOp, ModelOp modelOp,
      llvm::ArrayRef<VariableOp> variableOps,
      MatchedEquationInstanceOp equationOp,
      const llvm::DenseSet<VariableOp> &independentVariables,
      llvm::DenseMap<VariableOp, size_t> &independentVariablesPos,
      llvm::StringRef templateName);

  mlir::bmodelica::FunctionOp createPartialDerTemplateFromEquation(
      mlir::IRRewriter &rewriter, mlir::ModuleOp moduleOp, ModelOp modelOp,
      llvm::ArrayRef<VariableOp> variableOps,
      MatchedEquationInstanceOp equationOp,
      const llvm::DenseSet<VariableOp> &independentVariables,
      llvm::DenseMap<VariableOp, size_t> &independentVariablesPos,
      llvm::StringRef templateName);

  mlir::LogicalResult createJacobianFunction(
      mlir::OpBuilder &builder, mlir::ModuleOp moduleOp, ModelOp modelOp,
      MatchedEquationInstanceOp equationOp,
      llvm::StringRef jacobianFunctionName,
      const llvm::DenseSet<VariableOp> &independentVariables,
      const llvm::DenseMap<VariableOp, size_t> &independentVariablesPos,
      VariableOp independentVariable, llvm::StringRef partialDerTemplateName,
      llvm::SmallVectorImpl<int64_t> &seedSizes);

  std::string getIDAFunctionName(llvm::StringRef name) const;

  mlir::LogicalResult
  replaceVariableGetOps(mlir::RewriterBase &rewriter, ModelOp modelOp,
                        llvm::ArrayRef<VariableGetOp> getOps);

  std::optional<mlir::SymbolRefAttr>
  getDerivative(mlir::SymbolRefAttr variable) const;

  std::optional<mlir::SymbolRefAttr>
  getDerivedVariable(mlir::SymbolRefAttr derivative) const;

private:
  /// Instance identifier.
  /// It is used to create unique symbols.
  std::string identifier;

  mlir::SymbolTableCollection *symbolTableCollection;

  const DerivativesMap *derivativesMap;

  bool reducedSystem;
  bool reducedDerivatives;
  bool jacobianOneSweep;
  bool debugInformation;

  std::optional<double> startTime;
  std::optional<double> endTime;

  /// The algebraic variables of the model that are managed by IDA.
  /// An algebraic variable is a variable that is not a parameter, state or
  /// derivative.
  llvm::SmallVector<VariableOp> algebraicVariables;

  /// The state variables of the model that are managed by IDA.
  /// A state variable is a variable for which there exists a derivative
  /// variable.
  llvm::SmallVector<VariableOp> stateVariables;

  /// The derivative variables of the model that are managed by IDA.
  /// A derivative variable is a variable that is the derivative of another
  /// variable.
  llvm::SmallVector<VariableOp> derivativeVariables;

  /// The SSA values of the IDA variables representing the algebraic ones.
  llvm::SmallVector<mlir::Value> idaAlgebraicVariables;

  /// The SSA values of the IDA variables representing the state ones.
  llvm::SmallVector<mlir::Value> idaStateVariables;

  /// Map used for a faster lookup of the algebraic variable position.
  llvm::DenseMap<VariableOp, size_t> algebraicVariablesLookup;

  /// Map used for a faster lookup of the state variable position.
  llvm::DenseMap<VariableOp, size_t> stateVariablesLookup;

  /// Map used for a faster lookup of the derivative variable position.
  llvm::DenseMap<VariableOp, size_t> derivativeVariablesLookup;

  /// The equations managed by IDA.
  llvm::DenseSet<MatchedEquationInstanceOp> equations;
};
} // namespace

IDAInstance::IDAInstance(llvm::StringRef identifier,
                         mlir::SymbolTableCollection &symbolTableCollection,
                         const DerivativesMap *derivativesMap,
                         bool reducedSystem, bool reducedDerivatives,
                         bool jacobianOneSweep, bool debugInformation)
    : identifier(identifier.str()),
      symbolTableCollection(&symbolTableCollection),
      derivativesMap(derivativesMap), reducedSystem(reducedSystem),
      reducedDerivatives(reducedDerivatives),
      jacobianOneSweep(jacobianOneSweep), debugInformation(debugInformation),
      startTime(std::nullopt), endTime(std::nullopt) {}

bool IDAInstance::hasVariable(VariableOp variable) const {
  assert(variable != nullptr);

  return hasAlgebraicVariable(variable) || hasStateVariable(variable) ||
         hasDerivativeVariable(variable);
}

void IDAInstance::addAlgebraicVariable(VariableOp variable) {
  assert(variable != nullptr);

  if (!hasVariable(variable)) {
    algebraicVariables.push_back(variable);
    algebraicVariablesLookup[variable] = algebraicVariables.size() - 1;
  }
}

void IDAInstance::addStateVariable(VariableOp variable) {
  assert(variable != nullptr);

  if (!hasVariable(variable)) {
    stateVariables.push_back(variable);
    stateVariablesLookup[variable] = stateVariables.size() - 1;
  }
}

void IDAInstance::addDerivativeVariable(VariableOp variable) {
  assert(variable != nullptr);

  if (!hasVariable(variable)) {
    derivativeVariables.push_back(variable);
    derivativeVariablesLookup[variable] = derivativeVariables.size() - 1;
  }
}

bool IDAInstance::hasAlgebraicVariable(VariableOp variable) const {
  assert(variable != nullptr);
  return algebraicVariablesLookup.contains(variable);
}

bool IDAInstance::hasStateVariable(VariableOp variable) const {
  assert(variable != nullptr);
  return stateVariablesLookup.contains(variable);
}

bool IDAInstance::hasDerivativeVariable(VariableOp variable) const {
  assert(variable != nullptr);
  return derivativeVariablesLookup.contains(variable);
}

bool IDAInstance::hasEquation(MatchedEquationInstanceOp equation) const {
  assert(equation != nullptr);
  return llvm::find(equations, equation) != equations.end();
}

void IDAInstance::addEquation(MatchedEquationInstanceOp equation) {
  assert(equation != nullptr);
  equations.insert(equation);
}

mlir::LogicalResult IDAInstance::declareInstance(mlir::OpBuilder &builder,
                                                 mlir::Location loc,
                                                 mlir::ModuleOp moduleOp) {
  // Create the instance.
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());

  auto instanceOp = builder.create<mlir::ida::InstanceOp>(loc, identifier);

  // Update the symbol table.
  symbolTableCollection->getSymbolTable(moduleOp).insert(instanceOp);

  return mlir::success();
}

mlir::LogicalResult IDAInstance::initialize(mlir::OpBuilder &builder,
                                            mlir::Location loc) {
  // Initialize the instance.
  builder.create<mlir::ida::InitOp>(loc, identifier);

  return mlir::success();
}

mlir::LogicalResult IDAInstance::deleteInstance(mlir::OpBuilder &builder,
                                                mlir::Location loc) {
  builder.create<mlir::ida::FreeOp>(loc, identifier);
  return mlir::success();
}

mlir::LogicalResult
IDAInstance::configure(mlir::IRRewriter &rewriter, mlir::Location loc,
                       mlir::ModuleOp moduleOp, ModelOp modelOp,
                       llvm::ArrayRef<VariableOp> variableOps,
                       llvm::ArrayRef<SCCOp> allSCCs) {
  llvm::DenseMap<mlir::AffineMap, mlir::sundials::AccessFunctionOp>
      accessFunctionsMap;

  if (startTime.has_value()) {
    rewriter.create<mlir::ida::SetStartTimeOp>(
        loc, rewriter.getStringAttr(identifier),
        rewriter.getF64FloatAttr(*startTime));
  }

  if (endTime.has_value()) {
    rewriter.create<mlir::ida::SetEndTimeOp>(
        loc, rewriter.getStringAttr(identifier),
        rewriter.getF64FloatAttr(*endTime));
  }

  // Add the variables to IDA.
  if (mlir::failed(
          addVariablesToIDA(rewriter, loc, moduleOp, modelOp, variableOps))) {
    return mlir::failure();
  }

  // Add the equations to IDA.
  if (mlir::failed(addEquationsToIDA(rewriter, loc, moduleOp, modelOp,
                                     variableOps, allSCCs,
                                     accessFunctionsMap))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult
IDAInstance::addVariablesToIDA(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::ModuleOp moduleOp, ModelOp modelOp,
                               llvm::ArrayRef<VariableOp> variableOps) {
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
      // but IDA needs to see a single dimension of value 1.
      dimensions.push_back(1);
    } else {
      auto shape = arrayType.getShape();
      dimensions.insert(dimensions.end(), shape.begin(), shape.end());
    }

    return dimensions;
  };

  auto createGetterFn =
      [&](VariableOp variableOp) -> mlir::sundials::VariableGetterOp {
    std::string getterName =
        getIDAFunctionName("getter_" + std::to_string(getterFunctionCounter++));

    return createGetterFunction(builder, loc, moduleOp, variableOp, getterName);
  };

  auto createSetterFn =
      [&](VariableOp variableOp) -> mlir::sundials::VariableSetterOp {
    std::string setterName =
        getIDAFunctionName("setter_" + std::to_string(setterFunctionCounter++));

    return createSetterFunction(builder, loc, moduleOp, variableOp, setterName);
  };

  // Algebraic variables.
  for (VariableOp variableOp : algebraicVariables) {
    auto arrayType = variableOp.getVariableType().toArrayType();

    std::vector<int64_t> dimensions = getDimensionsFn(arrayType);
    auto getter = createGetterFn(variableOp);
    auto setter = createSetterFn(variableOp);

    auto addVariableOp = builder.create<mlir::ida::AddAlgebraicVariableOp>(
        loc, identifier, builder.getI64ArrayAttr(dimensions),
        getter.getSymName(), setter.getSymName());

    if (debugInformation) {
      addVariableOp.setNameAttr(variableOp.getSymNameAttr());
    }

    idaAlgebraicVariables.push_back(addVariableOp);
  }

  // State variables.
  for (VariableOp variableOp : stateVariables) {
    auto arrayType = variableOp.getVariableType().toArrayType();

    std::optional<mlir::SymbolRefAttr> derivativeName =
        getDerivative(mlir::SymbolRefAttr::get(variableOp.getSymNameAttr()));

    if (!derivativeName) {
      return mlir::failure();
    }

    assert(derivativeName->getNestedReferences().empty());

    auto derivativeVariableOp =
        symbolTableCollection->lookupSymbolIn<VariableOp>(modelOp,
                                                          *derivativeName);

    std::vector<int64_t> dimensions = getDimensionsFn(arrayType);
    auto stateGetter = createGetterFn(variableOp);
    auto stateSetter = createSetterFn(variableOp);
    auto derivativeGetter = createGetterFn(derivativeVariableOp);
    auto derivativeSetter = createSetterFn(derivativeVariableOp);

    auto addVariableOp = builder.create<mlir::ida::AddStateVariableOp>(
        loc, identifier, builder.getI64ArrayAttr(dimensions),
        stateGetter.getSymName(), stateSetter.getSymName(),
        derivativeGetter.getSymName(), derivativeSetter.getSymName());

    if (debugInformation) {
      addVariableOp.setNameAttr(variableOp.getSymNameAttr());
    }

    idaStateVariables.push_back(addVariableOp);
  }

  return mlir::success();
}

mlir::sundials::VariableGetterOp
IDAInstance::createGetterFunction(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::ModuleOp moduleOp, VariableOp variable,
                                  llvm::StringRef functionName) {
  return ::mlir::bmodelica::createGetterFunction(
      builder, *symbolTableCollection, loc, moduleOp, variable, functionName);
}

mlir::sundials::VariableSetterOp
IDAInstance::createSetterFunction(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::ModuleOp moduleOp, VariableOp variable,
                                  llvm::StringRef functionName) {
  return ::mlir::bmodelica::createSetterFunction(
      builder, *symbolTableCollection, loc, moduleOp, variable, functionName);
}

mlir::LogicalResult IDAInstance::addEquationsToIDA(
    mlir::IRRewriter &rewriter, mlir::Location loc, mlir::ModuleOp moduleOp,
    ModelOp modelOp, llvm::ArrayRef<VariableOp> variableOps,
    llvm::ArrayRef<SCCOp> allSCCs,
    llvm::DenseMap<mlir::AffineMap, mlir::sundials::AccessFunctionOp>
        &accessFunctionsMap) {
  // Substitute the accesses to non-IDA variables with the equations writing
  // in such variables.
  llvm::SmallVector<MatchedEquationInstanceOp> independentEquations;

  // First create the writes map, that is the knowledge of which equation
  // writes into a variable and in which indices.
  WritesMap<VariableOp, MatchedEquationInstanceOp> writesMap;

  if (mlir::failed(
          getWritesMap(writesMap, modelOp, allSCCs, *symbolTableCollection))) {
    return mlir::failure();
  }

  // The equations we are operating on.
  std::queue<MatchedEquationInstanceOp> processedEquations;

  for (MatchedEquationInstanceOp equation : equations) {
    processedEquations.push(equation);
  }

  LLVM_DEBUG(llvm::dbgs() << "Replacing the non-IDA variables\n");
  llvm::DenseSet<MatchedEquationInstanceOp> toBeErased;

  while (!processedEquations.empty()) {
    MatchedEquationInstanceOp equationOp = processedEquations.front();

    LLVM_DEBUG({
      llvm::dbgs() << "Current equation\n";
      equationOp.printInline(llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    IndexSet equationIndices = equationOp.getIterationSpace();

    LLVM_DEBUG(
        { llvm::dbgs() << "Equation indices: " << equationIndices << "\n"; });

    // Get the accesses of the equation.
    llvm::SmallVector<VariableAccess> accesses;

    if (mlir::failed(
            equationOp.getAccesses(accesses, *symbolTableCollection))) {
      return mlir::failure();
    }

    // Replace the non-IDA variables.
    bool atLeastOneAccessReplaced = false;

    for (const VariableAccess &access : accesses) {
      if (atLeastOneAccessReplaced) {
        // Avoid the duplicates.
        // For example, if we have the following equation
        //   eq1: z = x + y ...
        // and both x and y have to be replaced, then the replacement of 'x'
        // would create the equation
        //   eq2: z = f(...) + y ...
        // while the replacement of 'y' would create the equation
        //   eq3: z = x + g(...) ...
        // This implies that at the next round both would have respectively 'y'
        // and 'x' replaced for a second time, thus leading to two identical
        // equations:
        //   eq4: z = f(...) + g(...) ...
        //   eq5: z = f(...) + g(...) ...

        break;
      }

      const AccessFunction &accessFunction = access.getAccessFunction();
      std::optional<IndexSet> accessedVariableIndices = std::nullopt;

      if (!equationIndices.empty()) {
        accessedVariableIndices = accessFunction.map(equationIndices);
      }

      LLVM_DEBUG({
        llvm::dbgs() << "Accessed variable indices: ";

        if (accessedVariableIndices) {
          llvm::dbgs() << *accessedVariableIndices;
        } else {
          llvm::dbgs() << "{}";
        }

        llvm::dbgs() << "\n";
      });

      auto accessedVariableName = access.getVariable();
      assert(accessedVariableName.getNestedReferences().empty());

      auto accessedVariableOp =
          symbolTableCollection->lookupSymbolIn<VariableOp>(
              modelOp, accessedVariableName.getRootReference());

      LLVM_DEBUG({
        llvm::dbgs() << "Searching for equations writing to "
                     << accessedVariableOp.getSymName() << "\n";
      });

      auto writingEquations =
          llvm::make_range(writesMap.equal_range(accessedVariableOp));

      for (const auto &entry : writingEquations) {
        MatchedEquationInstanceOp writingEquationOp = entry.second.second;

        LLVM_DEBUG({
          llvm::dbgs() << "Found the following writing equation:\n";
          writingEquationOp.printInline(llvm::dbgs());
          llvm::dbgs() << "\n";
        });

        if (hasEquation(writingEquationOp)) {
          // Ignore the equation if it is already managed by IDA.
          LLVM_DEBUG(llvm::dbgs() << "The equation is managed by IDA\n");
          continue;
        }

        const IndexSet &writtenVariableIndices = entry.second.first;
        bool overlaps = false;

        if (writtenVariableIndices.empty()) {
          // Scalar replacement.
          LLVM_DEBUG(llvm::dbgs() << "Scalar replacement\n");
          overlaps = true;
        } else {
          if (!accessedVariableIndices ||
              accessedVariableIndices->overlaps(writtenVariableIndices)) {
            // Vectorized replacement.
            LLVM_DEBUG(llvm::dbgs() << "Vectorized replacement\n");
            overlaps = true;
          }
        }

        if (!overlaps) {
          LLVM_DEBUG(
              { llvm::dbgs() << "Written and read indices do not overlap\n"; });

          continue;
        }

        atLeastOneAccessReplaced = true;

        auto explicitWritingEquationOp = writingEquationOp.cloneAndExplicitate(
            rewriter, *symbolTableCollection);

        if (!explicitWritingEquationOp) {
          return mlir::failure();
        }

        auto eraseExplicitWritingEquation = llvm::make_scope_exit(
            [&]() { rewriter.eraseOp(explicitWritingEquationOp); });

        llvm::SmallVector<MatchedEquationInstanceOp> newEquations;

        auto writeAccess =
            explicitWritingEquationOp.getMatchedAccess(*symbolTableCollection);

        if (!writeAccess) {
          return mlir::failure();
        }

        std::optional<IndexSet> newEquationsIndices = std::nullopt;

        if (!equationIndices.empty()) {
          newEquationsIndices = equationIndices;

          newEquationsIndices =
              equationIndices.intersect(accessFunction.inverseMap(
                  writtenVariableIndices, equationIndices));
        }

        std::optional<std::reference_wrapper<const IndexSet>>
            optionalNewEquationIndices = std::nullopt;

        if (newEquationsIndices) {
          optionalNewEquationIndices =
              std::reference_wrapper(*newEquationsIndices);
        }

        if (mlir::failed(equationOp.cloneWithReplacedAccess(
                rewriter, optionalNewEquationIndices, access,
                explicitWritingEquationOp.getTemplate(), *writeAccess,
                newEquations))) {
          return mlir::failure();
        }

        LLVM_DEBUG({
          llvm::dbgs() << "Equations obtained with access replacement:\n";

          for (MatchedEquationInstanceOp newEquationOp : newEquations) {
            llvm::dbgs() << "  ";
            newEquationOp.printInline(llvm::dbgs());
            llvm::dbgs() << "\n";
          }
        });

        for (MatchedEquationInstanceOp newEquation : newEquations) {
          processedEquations.push(newEquation);
        }
      }
    }

    if (atLeastOneAccessReplaced) {
      toBeErased.insert(equationOp);
    } else {
      independentEquations.push_back(equationOp);
    }

    processedEquations.pop();
  }

  for (MatchedEquationInstanceOp equationOp : toBeErased) {
    rewriter.eraseOp(equationOp);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Independent equations:\n";

    for (MatchedEquationInstanceOp equationOp : independentEquations) {
      llvm::dbgs() << "  ";
      equationOp.printInline(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
  });

  // Check that all the non-IDA variables have been replaced.
  assert(([&]() -> bool {
           llvm::SmallVector<VariableAccess> accesses;

           for (auto equationOp : independentEquations) {
             accesses.clear();

             if (mlir::failed(equationOp.getAccesses(accesses,
                                                     *symbolTableCollection))) {
               return false;
             }

             for (const VariableAccess &access : accesses) {
               auto variable = access.getVariable();
               assert(variable.getNestedReferences().empty());

               auto variableOp =
                   symbolTableCollection->lookupSymbolIn<VariableOp>(
                       modelOp, variable.getRootReference());

               if (!hasVariable(variableOp)) {
                 if (!variableOp.isReadOnly()) {
                   return false;
                 }
               }
             }
           }

           return true;
         })() &&
         "Some non-IDA variables have not been replaced");

  // The accesses to non-IDA variables have been replaced. Now we can proceed
  // to create the residual and jacobian functions.

  // Counters used to obtain unique names for the functions.
  size_t accessFunctionsCounter = 0;
  size_t residualFunctionsCounter = 0;
  size_t jacobianFunctionsCounter = 0;
  size_t partialDerTemplatesCounter = 0;

  llvm::DenseMap<VariableOp, mlir::Value> variablesMapping;

  for (const auto &[variable, idaVariable] :
       llvm::zip(algebraicVariables, idaAlgebraicVariables)) {
    variablesMapping[variable] = idaVariable;
  }

  for (const auto &[variable, idaVariable] :
       llvm::zip(stateVariables, idaStateVariables)) {
    variablesMapping[variable] = idaVariable;
  }

  for (const auto &[variable, idaVariable] :
       llvm::zip(derivativeVariables, idaStateVariables)) {
    variablesMapping[variable] = idaVariable;
  }

  for (MatchedEquationInstanceOp equationOp : independentEquations) {
    // Keep track of the accessed variables in order to reduce the amount of
    // generated partial derivatives.
    llvm::SmallVector<VariableAccess> accesses;
    llvm::DenseSet<VariableOp> accessedVariables;

    if (mlir::failed(
            equationOp.getAccesses(accesses, *symbolTableCollection))) {
      return mlir::failure();
    }

    for (const VariableAccess &access : accesses) {
      auto variableOp = symbolTableCollection->lookupSymbolIn<VariableOp>(
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

    // Collect the independent variables for automatic differentiation.
    llvm::DenseSet<VariableOp> independentVariables;

    if (mlir::failed(getIndependentVariablesForAD(independentVariables, modelOp,
                                                  equationOp))) {
      return mlir::failure();
    }

    // Create the partial derivative template.
    std::string partialDerTemplateName = getIDAFunctionName(
        "pder_" + std::to_string(partialDerTemplatesCounter++));

    llvm::DenseMap<VariableOp, size_t> independentVariablesPos;

    if (mlir::failed(createPartialDerTemplateFunction(
            rewriter, moduleOp, modelOp, variableOps, equationOp,
            independentVariables, independentVariablesPos,
            partialDerTemplateName))) {
      return mlir::failure();
    }

    for (const MultidimensionalRange &range : llvm::make_range(
             equationIndices.rangesBegin(), equationIndices.rangesEnd())) {
      // Add the equation to the IDA instance.
      auto accessFunctionOp = getOrCreateAccessFunction(
          rewriter, equationOp.getLoc(), moduleOp,
          writeAccess->getAccessFunction().getAffineMap(),
          getIDAFunctionName("access"), accessFunctionsMap,
          accessFunctionsCounter);

      if (!accessFunctionOp) {
        return mlir::failure();
      }

      auto idaEquation = rewriter.create<mlir::ida::AddEquationOp>(
          equationOp.getLoc(), identifier,
          mlir::ida::MultidimensionalRangeAttr::get(rewriter.getContext(),
                                                    range));

      if (debugInformation) {
        std::string stringRepresentation;
        llvm::raw_string_ostream stringOstream(stringRepresentation);
        equationOp.printInline(stringOstream);

        idaEquation.setStringRepresentationAttr(
            rewriter.getStringAttr(stringRepresentation));
      }

      if (reducedDerivatives) {
        // Inform IDA about the accesses performed by the equation.
        if (mlir::failed(addVariableAccessesInfoToIDA(
                rewriter, loc, modelOp, equationOp, idaEquation,
                accessFunctionsMap, accessFunctionsCounter))) {
          return mlir::failure();
        }
      }

      // Create the residual function.
      std::string residualFunctionName = getIDAFunctionName(
          "residualFunction_" + std::to_string(residualFunctionsCounter++));

      if (mlir::failed(createResidualFunction(rewriter, moduleOp, modelOp,
                                              equationOp, idaEquation,
                                              residualFunctionName))) {
        return mlir::failure();
      }

      rewriter.create<mlir::ida::SetResidualOp>(loc, identifier, idaEquation,
                                                residualFunctionName);

      // Create the Jacobian functions.
      // Notice that Jacobian functions are not created for derivative
      // variables. Those are already handled when encountering the state
      // variable through the 'alpha' parameter set into the derivative seed.

      assert(algebraicVariables.size() == idaAlgebraicVariables.size());

      for (auto [variable, idaVariable] :
           llvm::zip(algebraicVariables, idaAlgebraicVariables)) {
        if (reducedDerivatives && !accessedVariables.contains(variable)) {
          // The partial derivative is always zero.
          continue;
        }

        std::string jacobianFunctionName = getIDAFunctionName(
            "jacobianFunction_" + std::to_string(jacobianFunctionsCounter++));

        llvm::SmallVector<int64_t> seedSizes;

        if (mlir::failed(createJacobianFunction(
                rewriter, moduleOp, modelOp, equationOp, jacobianFunctionName,
                independentVariables, independentVariablesPos, variable,
                partialDerTemplateName, seedSizes))) {
          return mlir::failure();
        }

        rewriter.create<mlir::ida::AddJacobianOp>(
            loc, identifier, idaEquation, idaVariable, jacobianFunctionName,
            rewriter.getI64ArrayAttr(seedSizes));
      }

      assert(stateVariables.size() == idaStateVariables.size());

      for (auto [variable, idaVariable] :
           llvm::zip(stateVariables, idaStateVariables)) {
        if (reducedDerivatives && !accessedVariables.contains(variable)) {
          auto derivative = getDerivative(
              mlir::SymbolRefAttr::get(variable.getSymNameAttr()));

          if (!derivative) {
            return mlir::failure();
          }

          assert(derivative->getNestedReferences().empty());

          auto derivativeVariableOp =
              symbolTableCollection->lookupSymbolIn<VariableOp>(
                  modelOp, derivative->getRootReference());

          if (!accessedVariables.contains(derivativeVariableOp)) {
            continue;
          }
        }

        std::string jacobianFunctionName = getIDAFunctionName(
            "jacobianFunction_" + std::to_string(jacobianFunctionsCounter++));

        llvm::SmallVector<int64_t> seedSizes;

        if (mlir::failed(createJacobianFunction(
                rewriter, moduleOp, modelOp, equationOp, jacobianFunctionName,
                independentVariables, independentVariablesPos, variable,
                partialDerTemplateName, seedSizes))) {
          return mlir::failure();
        }

        rewriter.create<mlir::ida::AddJacobianOp>(
            loc, identifier, idaEquation, idaVariable, jacobianFunctionName,
            rewriter.getI64ArrayAttr(seedSizes));
      }
    }
  }

  return mlir::success();
}

mlir::LogicalResult IDAInstance::addVariableAccessesInfoToIDA(
    mlir::OpBuilder &builder, mlir::Location loc, ModelOp modelOp,
    MatchedEquationInstanceOp equationOp, mlir::Value idaEquation,
    llvm::DenseMap<mlir::AffineMap, mlir::sundials::AccessFunctionOp>
        &accessFunctionsMap,
    size_t &accessFunctionsCounter) {
  LLVM_DEBUG({
    llvm::dbgs() << "Adding access information for equation ";
    equationOp.printInline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();
  assert(idaEquation.getType().isa<mlir::ida::EquationType>());

  auto getIDAVariable = [&](VariableOp variableOp) -> mlir::Value {
    if (auto stateVariable = getDerivedVariable(
            mlir::SymbolRefAttr::get(variableOp.getSymNameAttr()))) {
      auto stateVariableOp = symbolTableCollection->lookupSymbolIn<VariableOp>(
          modelOp, *stateVariable);

      return idaStateVariables[stateVariablesLookup[stateVariableOp]];
    }

    if (auto derivativeVariable = getDerivative(
            mlir::SymbolRefAttr::get(variableOp.getSymNameAttr()))) {
      return idaStateVariables[stateVariablesLookup[variableOp]];
    }

    return idaAlgebraicVariables[algebraicVariablesLookup[variableOp]];
  };

  // Keep track of the discovered accesses in order to avoid adding the same
  // access map multiple times for the same variable.
  llvm::DenseMap<mlir::Value, llvm::DenseSet<mlir::AffineMap>> maps;

  llvm::SmallVector<VariableAccess> accesses;

  if (mlir::failed(equationOp.getAccesses(accesses, *symbolTableCollection))) {
    return mlir::failure();
  }

  for (const VariableAccess &access : accesses) {
    auto variableOp = symbolTableCollection->lookupSymbolIn<VariableOp>(
        modelOp, access.getVariable());

    LLVM_DEBUG({
      llvm::dbgs() << "  - Variable \"" << variableOp.getSymName() << "\"\n";
    });

    if (!hasVariable(variableOp)) {
      LLVM_DEBUG({ llvm::dbgs() << "    Not handled by IDA. Skipping.\n"; });

      continue;
    }

    mlir::Value idaVariable = getIDAVariable(variableOp);
    assert(idaVariable != nullptr);
    LLVM_DEBUG(llvm::dbgs() << "    IDA variable: " << idaVariable << "\n");

    const AccessFunction &accessFunction = access.getAccessFunction();

    if (accessFunction.isAffine()) {
      LLVM_DEBUG({
        llvm::dbgs() << "    Access function: " << accessFunction.getAffineMap()
                     << "\n";
      });

      maps[idaVariable].insert(accessFunction.getAffineMap());
    } else {
      IndexSet accessedIndices =
          accessFunction.map(equationOp.getIterationSpace());

      for (Point point : accessedIndices) {
        llvm::SmallVector<mlir::AffineExpr> results;

        for (size_t dim = 0, rank = point.rank(); dim < rank; ++dim) {
          results.push_back(
              mlir::getAffineConstantExpr(point[dim], builder.getContext()));
        }

        auto affineMap = mlir::AffineMap::get(accessFunction.getNumOfDims(), 0,
                                              results, builder.getContext());

        LLVM_DEBUG(
            { llvm::dbgs() << "    Access function: " << affineMap << "\n"; });

        maps[idaVariable].insert(affineMap);
      }
    }
  }

  // Inform IDA about the discovered accesses.
  for (const auto &entry : maps) {
    mlir::Value idaVariable = entry.getFirst();

    for (mlir::AffineMap map : entry.getSecond()) {
      auto accessFunctionOp = getOrCreateAccessFunction(
          builder, loc, moduleOp, map, getIDAFunctionName("access"),
          accessFunctionsMap, accessFunctionsCounter);

      if (!accessFunctionOp) {
        return mlir::failure();
      }

      builder.create<mlir::ida::AddVariableAccessOp>(
          loc, identifier, idaEquation, idaVariable,
          accessFunctionOp.getSymName());
    }
  }

  return mlir::success();
}

mlir::sundials::AccessFunctionOp IDAInstance::getOrCreateAccessFunction(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::ModuleOp moduleOp,
    mlir::AffineMap access, llvm::StringRef functionNamePrefix,
    llvm::DenseMap<mlir::AffineMap, mlir::sundials::AccessFunctionOp>
        &accessFunctionsMap,
    size_t &accessFunctionsCounter) {
  auto it = accessFunctionsMap.find(access);

  if (it == accessFunctionsMap.end()) {
    std::string functionName = functionNamePrefix.str() + "_" +
                               std::to_string(accessFunctionsCounter++);

    auto accessFunctionOp =
        createAccessFunction(builder, loc, moduleOp, access, functionName);

    if (!accessFunctionOp) {
      return nullptr;
    }

    accessFunctionsMap[access] = accessFunctionOp;
    return accessFunctionOp;
  }

  return it->getSecond();
}

mlir::sundials::AccessFunctionOp IDAInstance::createAccessFunction(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::ModuleOp moduleOp,
    mlir::AffineMap access, llvm::StringRef functionName) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  // Normalize the access so that it has at least one dimension and one result.
  llvm::SmallVector<mlir::AffineExpr> expressions;

  for (mlir::AffineExpr expression : access.getResults()) {
    expressions.push_back(expression);
  }

  if (expressions.empty()) {
    expressions.push_back(mlir::getAffineConstantExpr(0, builder.getContext()));
  }

  auto extendedAccess = mlir::AffineMap::get(
      std::max(static_cast<unsigned int>(1), access.getNumDims()),
      access.getNumSymbols(), expressions, builder.getContext());

  // Create the operation for the access function.
  auto accessFunctionOp = builder.create<mlir::sundials::AccessFunctionOp>(
      loc, functionName, extendedAccess.getNumDims(),
      extendedAccess.getNumResults());

  symbolTableCollection->getSymbolTable(moduleOp).insert(accessFunctionOp);

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

mlir::LogicalResult IDAInstance::createResidualFunction(
    mlir::RewriterBase &rewriter, mlir::ModuleOp moduleOp, ModelOp modelOp,
    MatchedEquationInstanceOp equationOp, mlir::Value idaEquation,
    llvm::StringRef residualFunctionName) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  mlir::Location loc = equationOp.getLoc();
  size_t numOfInductionVariables = equationOp.getInductionVariables().size();

  auto residualFunction = rewriter.create<mlir::ida::ResidualFunctionOp>(
      loc, residualFunctionName, numOfInductionVariables);

  symbolTableCollection->getSymbolTable(moduleOp).insert(residualFunction);

  mlir::Block *bodyBlock = residualFunction.addEntryBlock();
  rewriter.setInsertionPointToStart(bodyBlock);

  // Map for the SSA values.
  mlir::IRMapping mapping;

  // Map the iteration variables.
  auto originalInductions = equationOp.getInductionVariables();
  auto mappedInductions = residualFunction.getEquationIndices();
  assert(originalInductions.size() == mappedInductions.size());

  for (size_t i = 0, e = originalInductions.size(); i < e; ++i) {
    mapping.map(originalInductions[i], mappedInductions[i]);
  }

  for (auto &op : equationOp.getTemplate().getOps()) {
    if (auto timeOp = mlir::dyn_cast<TimeOp>(op)) {
      mlir::Value timeReplacement = residualFunction.getTime();

      timeReplacement = rewriter.create<CastOp>(
          timeReplacement.getLoc(), RealType::get(rewriter.getContext()),
          timeReplacement);

      mapping.map(timeOp.getResult(), timeReplacement);
    } else if (mlir::isa<EquationSideOp>(op)) {
      continue;
    } else if (auto equationSidesOp = mlir::dyn_cast<EquationSidesOp>(op)) {
      // Compute the difference between the right-hand side and the left-hand
      // side of the equation.
      mlir::Value lhs = mapping.lookup(equationSidesOp.getLhsValues()[0]);
      mlir::Value rhs = mapping.lookup(equationSidesOp.getRhsValues()[0]);
      assert(!lhs.getType().isa<mlir::ShapedType>());
      assert(!rhs.getType().isa<mlir::ShapedType>());

      mlir::Value difference =
          rewriter.create<SubOp>(loc, rewriter.getF64Type(), rhs, lhs);

      rewriter.create<mlir::ida::ReturnOp>(difference.getLoc(), difference);
    } else {
      rewriter.clone(op, mapping);
    }
  }

  // Replace the original variable accesses.
  llvm::SmallVector<VariableGetOp> getOps;

  residualFunction.walk([&](VariableGetOp getOp) { getOps.push_back(getOp); });

  if (mlir::failed(replaceVariableGetOps(rewriter, modelOp, getOps))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult IDAInstance::getIndependentVariablesForAD(
    llvm::DenseSet<VariableOp> &result, ModelOp modelOp,
    MatchedEquationInstanceOp equationOp) {
  llvm::SmallVector<VariableAccess> accesses;

  if (mlir::failed(equationOp.getAccesses(accesses, *symbolTableCollection))) {
    return mlir::failure();
  }

  for (const VariableAccess &access : accesses) {
    auto variableOp = symbolTableCollection->lookupSymbolIn<VariableOp>(
        modelOp, access.getVariable());

    if (variableOp.isReadOnly()) {
      // Treat read-only variables as if they were just numbers.
      continue;
    }

    result.insert(variableOp);

    if (auto derivative = getDerivative(access.getVariable())) {
      auto derivativeVariableOp =
          symbolTableCollection->lookupSymbolIn<VariableOp>(modelOp,
                                                            *derivative);

      result.insert(derivativeVariableOp);
    }

    if (auto state = getDerivedVariable(access.getVariable())) {
      auto stateVariableOp =
          symbolTableCollection->lookupSymbolIn<VariableOp>(modelOp, *state);

      result.insert(stateVariableOp);
    }
  }

  return mlir::success();
}

mlir::LogicalResult IDAInstance::createPartialDerTemplateFunction(
    mlir::IRRewriter &rewriter, mlir::ModuleOp moduleOp, ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variableOps,
    MatchedEquationInstanceOp equationOp,
    const llvm::DenseSet<VariableOp> &independentVariables,
    llvm::DenseMap<VariableOp, size_t> &independentVariablesPos,
    llvm::StringRef templateName) {
  LLVM_DEBUG({
    llvm::dbgs() << "Creating partial derivative function for equation:\n";
    llvm::dbgs() << "  ";
    equationOp.printInline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // TODO methanol fails because there are remains of accesses during AD. the
  // equation must be cleaned first.
  mlir::Location loc = equationOp.getLoc();

  auto partialDerTemplate = createPartialDerTemplateFromEquation(
      rewriter, moduleOp, modelOp, variableOps, equationOp,
      independentVariables, independentVariablesPos, templateName);

  if (!partialDerTemplate) {
    return mlir::failure();
  }

  // Add the time to the input variables (and signature).
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(partialDerTemplate.getBody());

  auto timeVariable = rewriter.create<VariableOp>(
      loc, "time",
      VariableType::get(std::nullopt, RealType::get(rewriter.getContext()),
                        VariabilityProperty::none, IOProperty::input));

  // Replace the TimeOp with the newly created variable.
  llvm::SmallVector<TimeOp> timeOps;

  partialDerTemplate.walk([&](TimeOp timeOp) { timeOps.push_back(timeOp); });

  for (TimeOp timeOp : timeOps) {
    rewriter.setInsertionPoint(timeOp);

    mlir::Value time = rewriter.create<VariableGetOp>(
        timeVariable.getLoc(), timeVariable.getVariableType().unwrap(),
        timeVariable.getSymName());

    rewriter.replaceOp(timeOp, time);
  }

  return mlir::success();
}

FunctionOp IDAInstance::createPartialDerTemplateFromEquation(
    mlir::IRRewriter &rewriter, mlir::ModuleOp moduleOp, ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variableOps,
    MatchedEquationInstanceOp equationOp,
    const llvm::DenseSet<VariableOp> &independentVariables,
    llvm::DenseMap<VariableOp, size_t> &independentVariablesPos,
    llvm::StringRef templateName) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  mlir::Location loc = equationOp.getLoc();

  // Create the function.
  LLVM_DEBUG(llvm::dbgs() << "Creating the function to be derived\n");
  std::string functionOpName = templateName.str() + "_base";

  // Create the function to be derived.
  auto functionOp = rewriter.create<FunctionOp>(loc, functionOpName);
  rewriter.createBlock(&functionOp.getBodyRegion());

  // Start the body of the function.
  rewriter.setInsertionPointToStart(functionOp.getBody());

  // Keep track of all the variables that have been declared for creating the
  // body of the function to be derived.
  llvm::DenseSet<llvm::StringRef> allLocalVariables;

  // Replicate the original independent variables inside the function.
  llvm::StringMap<VariableOp> mappedVariableOps;
  size_t independentVariableIndex = 0;

  for (VariableOp variableOp : variableOps) {
    if (!independentVariables.contains(variableOp)) {
      continue;
    }

    VariableType variableType =
        variableOp.getVariableType().withIOProperty(IOProperty::input);

    auto clonedVariableOp = rewriter.create<VariableOp>(
        variableOp.getLoc(), variableOp.getSymName(), variableType);

    allLocalVariables.insert(clonedVariableOp.getSymName());
    mappedVariableOps[variableOp.getSymName()] = clonedVariableOp;
    independentVariablesPos[variableOp] = independentVariableIndex++;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Mapping of model variables to local variables:\n";

    for (VariableOp variableOp : variableOps) {
      if (!independentVariables.contains(variableOp)) {
        continue;
      }

      auto mappedVariable = mappedVariableOps[variableOp.getSymName()];

      llvm::dbgs() << variableOp.getSymName() << " -> "
                   << mappedVariable.getSymName() << "\n";
    }
  });

  // Create the induction variables.
  llvm::SmallVector<VariableOp, 3> inductionVariablesOps;

  size_t numOfInductions = equationOp.getInductionVariables().size();

  for (size_t i = 0; i < numOfInductions; ++i) {
    std::string variableName = "ind" + std::to_string(i);

    auto variableType = VariableType::wrap(
        rewriter.getIndexType(), VariabilityProperty::none, IOProperty::input);

    auto variableOp =
        rewriter.create<VariableOp>(loc, variableName, variableType);

    allLocalVariables.insert(variableOp.getSymName());
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
      VariableType::wrap(RealType::get(rewriter.getContext()),
                         VariabilityProperty::none, IOProperty::output));

  allLocalVariables.insert(outputVariableOp.getSymName());

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
  llvm::DenseSet<mlir::Operation *> toBeCloned;
  llvm::SmallVector<mlir::Operation *> toBeClonedVisitStack;

  auto equationSidesOp = mlir::cast<EquationSidesOp>(
      equationOp.getTemplate().getBody()->getTerminator());

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

    op->walk([&](mlir::Operation *nestedOp) { toBeCloned.insert(nestedOp); });
  }

  // Clone the original operations and compute the residual value.
  for (auto &op : equationOp.getTemplate().getOps()) {
    if (!toBeCloned.contains(&op)) {
      continue;
    }

    if (auto globalGetOp = mlir::dyn_cast<GlobalVariableGetOp>(op)) {
      VariableOp variableOp = mappedVariableOps[globalGetOp.getVariable()];

      auto getOp =
          rewriter.create<VariableGetOp>(globalGetOp.getLoc(), variableOp);

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

  rewriter.create<VariableSetOp>(loc, outputVariableOp, result);

  // Use the qualified accesses for the non-independent variables.
  llvm::SmallVector<VariableGetOp> getOpsToQualify;

  functionOp.walk([&](VariableGetOp getOp) {
    if (!allLocalVariables.contains(getOp.getVariable())) {
      getOpsToQualify.push_back(getOp);
    }
  });

  if (mlir::failed(replaceVariableGetOps(rewriter, modelOp, getOpsToQualify))) {
    return nullptr;
  }

  // Create the derivative template function.
  LLVM_DEBUG({
    llvm::dbgs() << "Function being derived:\n" << functionOp << "\n";
  });

  ad::forward::State state;

  auto derTemplate = ad::forward::createFunctionPartialDerivative(
      rewriter, state, functionOp, templateName);

  if (!derTemplate) {
    return nullptr;
  }

  rewriter.eraseOp(functionOp);

  // Replace the mapped variables with qualified accesses.
  llvm::DenseSet<VariableOp> variablesToBeReplaced;

  for (VariableOp variableOp : derTemplate->getVariables()) {
    if (mappedVariableOps.contains(variableOp.getSymName())) {
      variablesToBeReplaced.insert(variableOp);
    }
  }

  llvm::SmallVector<VariableGetOp> variableGetOps;

  derTemplate->walk([&](VariableGetOp getOp) {
    if (mappedVariableOps.contains(getOp.getVariable())) {
      variableGetOps.push_back(getOp);
    }
  });

  if (mlir::failed(replaceVariableGetOps(rewriter, modelOp, variableGetOps))) {
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

mlir::LogicalResult IDAInstance::createJacobianFunction(
    mlir::OpBuilder &builder, mlir::ModuleOp moduleOp, ModelOp modelOp,
    MatchedEquationInstanceOp equationOp, llvm::StringRef jacobianFunctionName,
    const llvm::DenseSet<VariableOp> &independentVariables,
    const llvm::DenseMap<VariableOp, size_t> &independentVariablesPos,
    VariableOp independentVariable, llvm::StringRef partialDerTemplateName,
    llvm::SmallVectorImpl<int64_t> &seedSizes) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  mlir::Location loc = equationOp.getLoc();

  size_t numOfIndependentVars = independentVariables.size();
  size_t numOfInductions = equationOp.getInductionVariables().size();

  // Create the function.
  auto jacobianFunction = builder.create<mlir::ida::JacobianFunctionOp>(
      loc, jacobianFunctionName, numOfInductions,
      independentVariable.getVariableType().getRank(),
      independentVariables.size());

  symbolTableCollection->getSymbolTable(moduleOp).insert(jacobianFunction);

  mlir::Block *bodyBlock = jacobianFunction.addEntryBlock();
  builder.setInsertionPointToStart(bodyBlock);

  // Create the global seeds for the variables.
  llvm::SmallVector<mlir::Value> varSeeds(numOfIndependentVars, nullptr);

  for (VariableOp independentVariableOp : independentVariables) {
    assert(independentVariablesPos.count(independentVariableOp) != 0);
    size_t pos = independentVariablesPos.lookup(independentVariableOp);
    mlir::Value seed = builder.create<PoolVariableGetOp>(
        loc, independentVariableOp.getVariableType().toArrayType(),
        jacobianFunction.getMemoryPool(), jacobianFunction.getADSeeds()[pos]);
    varSeeds[pos] = seed;
  }

  for (mlir::Value seed : varSeeds) {
    auto arrayType = seed.getType().cast<ArrayType>();
    seedSizes.push_back(arrayType.getNumElements());
  }

  // Zero and one constants to be used to update the array seeds or for the
  // scalar seeds.
  mlir::Value zero =
      builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), 0));

  mlir::Value one =
      builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), 1));

  // Function to collect the arguments to be passed to the derivative template
  // function.
  auto collectArgsFn = [&](llvm::SmallVectorImpl<mlir::Value> &args) {
    // 'Time' variable.
    args.push_back(jacobianFunction.getTime());

    // Equation indices.
    for (mlir::Value equationIndex : jacobianFunction.getEquationIndices()) {
      args.push_back(equationIndex);
    }

    // Seeds of the variables.
    for (mlir::Value seed : varSeeds) {
      auto arrayType = seed.getType().cast<ArrayType>();

      auto tensorType = mlir::RankedTensorType::get(arrayType.getShape(),
                                                    arrayType.getElementType());

      seed = builder.create<ArrayToTensorOp>(loc, tensorType, seed);

      if (tensorType.getRank() == 0) {
        seed = builder.create<TensorExtractOp>(loc, seed, std::nullopt);
      }

      args.push_back(seed);
    }

    // Seeds of the equation indices. They are all equal to zero.
    for (size_t i = 0; i < jacobianFunction.getEquationIndices().size(); ++i) {
      args.push_back(zero);
    }
  };

  // Determine the positions of the seeds within the seeds list.
  std::optional<size_t> oneSeedPosition = std::nullopt;
  std::optional<size_t> derSeedPosition = std::nullopt;

  if (independentVariablesPos.contains(independentVariable)) {
    oneSeedPosition = independentVariablesPos.lookup(independentVariable);
  }

  if (auto derivative = getDerivative(
          mlir::SymbolRefAttr::get(independentVariable.getSymNameAttr()))) {
    auto derVariableOp =
        symbolTableCollection->lookupSymbolIn<VariableOp>(modelOp, *derivative);

    if (independentVariablesPos.contains(derVariableOp)) {
      derSeedPosition = independentVariablesPos.lookup(derVariableOp);
    }
  }

  if (jacobianOneSweep) {
    // Perform just one call to the template function.
    if (oneSeedPosition) {
      // Set the seed of the variable to one.
      builder.create<StoreOp>(loc, one, varSeeds[*oneSeedPosition],
                              jacobianFunction.getVariableIndices());
    }

    if (derSeedPosition) {
      // Set the seed of the derivative to alpha.
      mlir::Value alpha = jacobianFunction.getAlpha();

      alpha = builder.create<CastOp>(
          alpha.getLoc(), RealType::get(builder.getContext()), alpha);

      builder.create<StoreOp>(loc, alpha, varSeeds[*derSeedPosition],
                              jacobianFunction.getVariableIndices());
    }

    // Call the template function.
    llvm::SmallVector<mlir::Value> args;
    collectArgsFn(args);

    auto templateCall = builder.create<CallOp>(
        loc,
        mlir::SymbolRefAttr::get(builder.getContext(), partialDerTemplateName),
        RealType::get(builder.getContext()), args);

    mlir::Value result = templateCall.getResult(0);

    // Reset the seeds.
    if (oneSeedPosition) {
      builder.create<StoreOp>(loc, zero, varSeeds[*oneSeedPosition],
                              jacobianFunction.getVariableIndices());
    }

    if (derSeedPosition) {
      builder.create<StoreOp>(loc, zero, varSeeds[*derSeedPosition],
                              jacobianFunction.getVariableIndices());
    }

    // Return the result.
    result = builder.create<CastOp>(loc, builder.getF64Type(), result);
    builder.create<mlir::ida::ReturnOp>(loc, result);
  } else {
    llvm::SmallVector<mlir::Value> args;

    // Perform the first call to the template function.
    if (oneSeedPosition) {
      builder.create<StoreOp>(loc, one, varSeeds[*oneSeedPosition],
                              jacobianFunction.getVariableIndices());
    }

    args.clear();
    collectArgsFn(args);

    auto firstTemplateCall = builder.create<CallOp>(
        loc,
        mlir::SymbolRefAttr::get(builder.getContext(), partialDerTemplateName),
        RealType::get(builder.getContext()), args);

    mlir::Value result = firstTemplateCall.getResult(0);

    if (oneSeedPosition) {
      // Reset the seed of the variable.
      builder.create<StoreOp>(loc, zero, varSeeds[*oneSeedPosition],
                              jacobianFunction.getVariableIndices());
    }

    if (derSeedPosition) {
      // Set the seed of the derivative to one.
      builder.create<StoreOp>(loc, one, varSeeds[*derSeedPosition],
                              jacobianFunction.getVariableIndices());

      // Call the template function.
      args.clear();
      collectArgsFn(args);

      auto secondTemplateCall = builder.create<CallOp>(
          loc,
          mlir::SymbolRefAttr::get(builder.getContext(),
                                   partialDerTemplateName),
          RealType::get(builder.getContext()), args);

      // Reset the seed of the variable.
      builder.create<StoreOp>(loc, zero, varSeeds[*derSeedPosition],
                              jacobianFunction.getVariableIndices());

      mlir::Value secondResult = secondTemplateCall.getResult(0);

      mlir::Value secondResultTimesAlpha =
          builder.create<MulOp>(loc, RealType::get(builder.getContext()),
                                jacobianFunction.getAlpha(), secondResult);

      result = builder.create<AddOp>(loc, RealType::get(builder.getContext()),
                                     result, secondResultTimesAlpha);
    }

    // Return the result.
    result = builder.create<CastOp>(loc, builder.getF64Type(), result);
    builder.create<mlir::ida::ReturnOp>(loc, result);
  }

  return mlir::success();
}

mlir::LogicalResult IDAInstance::performCalcIC(mlir::OpBuilder &builder,
                                               mlir::Location loc) {
  builder.create<mlir::ida::CalcICOp>(loc, identifier);
  return mlir::success();
}

mlir::LogicalResult IDAInstance::performStep(mlir::OpBuilder &builder,
                                             mlir::Location loc) {
  builder.create<mlir::ida::StepOp>(loc, identifier);
  return mlir::success();
}

mlir::Value IDAInstance::getCurrentTime(mlir::OpBuilder &builder,
                                        mlir::Location loc) {
  return builder.create<mlir::ida::GetCurrentTimeOp>(loc, identifier);
}

std::string IDAInstance::getIDAFunctionName(llvm::StringRef name) const {
  return identifier + "_" + name.str();
}

mlir::LogicalResult
IDAInstance::replaceVariableGetOps(mlir::RewriterBase &rewriter,
                                   ModelOp modelOp,
                                   llvm::ArrayRef<VariableGetOp> getOps) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  for (VariableGetOp variableGetOp : getOps) {
    rewriter.setInsertionPoint(variableGetOp);

    auto variableOp = symbolTableCollection->lookupSymbolIn<VariableOp>(
        modelOp, variableGetOp.getVariableAttr());

    auto qualifiedGetOp = rewriter.create<QualifiedVariableGetOp>(
        variableGetOp.getLoc(), variableOp);

    rewriter.replaceOp(variableGetOp, qualifiedGetOp);
  }

  return mlir::success();
}

std::optional<mlir::SymbolRefAttr>
IDAInstance::getDerivative(mlir::SymbolRefAttr variable) const {
  if (!derivativesMap) {
    return std::nullopt;
  }

  return derivativesMap->getDerivative(variable);
}

std::optional<mlir::SymbolRefAttr>
IDAInstance::getDerivedVariable(mlir::SymbolRefAttr derivative) const {
  if (!derivativesMap) {
    return std::nullopt;
  }

  return derivativesMap->getDerivedVariable(derivative);
}

namespace {
class IDAPass : public mlir::bmodelica::impl::IDAPassBase<IDAPass> {
public:
  using IDAPassBase::IDAPassBase;

  void runOnOperation() override;

private:
  DerivativesMap &getDerivativesMap(ModelOp modelOp);

  mlir::LogicalResult processModelOp(ModelOp modelOp);

  mlir::LogicalResult
  solveMainModel(mlir::IRRewriter &rewriter,
                 mlir::SymbolTableCollection &symbolTableCollection,
                 ModelOp modelOp, llvm::ArrayRef<VariableOp> variables,
                 llvm::ArrayRef<SCCOp> SCCs);

  /// Add a SCC to the IDA instance.
  mlir::LogicalResult
  addMainModelSCC(mlir::SymbolTableCollection &symbolTableCollection,
                  ModelOp modelOp, const DerivativesMap &derivativesMap,
                  IDAInstance &idaInstance, SCCOp scc);

  /// Add an equation to the IDA instance together with its written
  /// variable.
  mlir::LogicalResult
  addMainModelEquation(mlir::SymbolTableCollection &symbolTableCollection,
                       ModelOp modelOp, const DerivativesMap &derivativesMap,
                       IDAInstance &idaInstance,
                       MatchedEquationInstanceOp equationOp);

  mlir::LogicalResult addEquationsWritingToIDAVariables(
      mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
      IDAInstance &idaInstance, llvm::ArrayRef<SCCOp> SCCs,
      llvm::DenseSet<SCCOp> &externalSCCs,
      llvm::function_ref<mlir::LogicalResult(SCCOp)> addFn);

  /// Create the function that instantiates the external solvers to be used
  /// during the simulation.
  mlir::LogicalResult createInitMainSolversFunction(
      mlir::IRRewriter &rewriter, mlir::ModuleOp moduleOp,
      mlir::SymbolTableCollection &symbolTableCollection, mlir::Location loc,
      ModelOp modelOp, IDAInstance *idaInstance,
      llvm::ArrayRef<VariableOp> variableOps,
      llvm::ArrayRef<SCCOp> allSCCs) const;

  /// Create the function that deallocates the external solvers used during
  /// the simulation.
  mlir::LogicalResult
  createDeinitMainSolversFunction(mlir::OpBuilder &builder,
                                  mlir::ModuleOp moduleOp, mlir::Location loc,
                                  IDAInstance *idaInstance) const;

  /// Create the function that computes the initial conditions of the "main
  /// model".
  mlir::LogicalResult createCalcICFunction(mlir::OpBuilder &builder,
                                           mlir::ModuleOp moduleOp,
                                           mlir::Location loc,
                                           IDAInstance *idaInstance) const;

  /// Create the functions that calculates the values that the variables
  /// belonging to IDA will have in the next iteration.
  mlir::LogicalResult
  createUpdateIDAVariablesFunction(mlir::OpBuilder &builder,
                                   mlir::ModuleOp moduleOp, mlir::Location loc,
                                   IDAInstance *idaInstance) const;

  /// Create the functions that calculates the values that the variables
  /// not belonging to IDA will have in the next iteration.
  mlir::LogicalResult createUpdateNonIDAVariablesFunction(
      mlir::RewriterBase &rewriter, mlir::ModuleOp moduleOp,
      mlir::SymbolTableCollection &symbolTableCollection,
      IDAInstance *idaInstance, ModelOp modelOp,
      llvm::ArrayRef<SCCOp> internalSCCs);

  /// Create the function to be used to get the time reached by IDA.
  mlir::LogicalResult createGetIDATimeFunction(mlir::OpBuilder &builder,
                                               mlir::ModuleOp moduleOp,
                                               mlir::Location loc,
                                               IDAInstance *idaInstance) const;

  mlir::LogicalResult cleanModelOp(ModelOp modelOp);
};
} // namespace

void IDAPass::runOnOperation() {
  llvm::SmallVector<ModelOp, 1> modelOps;

  walkClasses(getOperation(), [&](mlir::Operation *op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  for (ModelOp modelOp : modelOps) {
    if (mlir::failed(processModelOp(modelOp))) {
      return signalPassFailure();
    }

    if (mlir::failed(cleanModelOp(modelOp))) {
      return signalPassFailure();
    }
  }

  markAnalysesPreserved<DerivativesMap>();
}

mlir::LogicalResult IDAPass::processModelOp(ModelOp modelOp) {
  mlir::IRRewriter rewriter(&getContext());
  mlir::SymbolTableCollection symbolTableCollection;

  llvm::SmallVector<SCCOp> mainSCCs;
  modelOp.collectMainSCCs(mainSCCs);

  llvm::SmallVector<VariableOp> variables;
  modelOp.collectVariables(variables);

  // Solve the 'main' model.
  if (mlir::failed(solveMainModel(rewriter, symbolTableCollection, modelOp,
                                  variables, mainSCCs))) {
    return mlir::failure();
  }

  for (DynamicOp dynamicOp :
       llvm::make_early_inc_range(modelOp.getOps<DynamicOp>())) {
    rewriter.eraseOp(dynamicOp);
  }

  return mlir::success();
}

mlir::LogicalResult
IDAPass::solveMainModel(mlir::IRRewriter &rewriter,
                        mlir::SymbolTableCollection &symbolTableCollection,
                        ModelOp modelOp, llvm::ArrayRef<VariableOp> variables,
                        llvm::ArrayRef<SCCOp> SCCs) {
  LLVM_DEBUG(llvm::dbgs() << "Solving the 'main' model\n");
  auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();

  const DerivativesMap &derivativesMap = modelOp.getProperties().derivativesMap;

  auto idaInstance = std::make_unique<IDAInstance>(
      "ida_main", symbolTableCollection, &derivativesMap, reducedSystem,
      reducedDerivatives, jacobianOneSweep, debugInformation);

  // The SCCs that are handled by IDA.
  llvm::DenseSet<SCCOp> externalSCCs;

  if (reducedSystem) {
    LLVM_DEBUG(llvm::dbgs() << "Reduced system feature enabled\n");

    // Add the state and derivative variables.
    // All of them must always be known to IDA.

    for (VariableOp variable : variables) {
      if (auto derivative = derivativesMap.getDerivative(
              mlir::SymbolRefAttr::get(variable.getSymNameAttr()))) {
        assert(derivative->getNestedReferences().empty());

        auto derivativeVariableOp =
            symbolTableCollection.lookupSymbolIn<VariableOp>(
                modelOp, derivative->getRootReference());

        assert(derivativeVariableOp && "Derivative not found");

        LLVM_DEBUG(llvm::dbgs()
                   << "Add state variable: " << variable.getSymName() << "\n");
        idaInstance->addStateVariable(variable);

        LLVM_DEBUG(llvm::dbgs() << "Add derivative variable: "
                                << derivativeVariableOp.getSymName() << "\n");
        idaInstance->addDerivativeVariable(derivativeVariableOp);
      }
    }

    // Add the equations writing to variables handled by IDA.
    for (SCCOp scc : SCCs) {
      for (MatchedEquationInstanceOp equationOp :
           scc.getOps<MatchedEquationInstanceOp>()) {
        std::optional<VariableAccess> writeAccess =
            equationOp.getMatchedAccess(symbolTableCollection);

        if (!writeAccess) {
          LLVM_DEBUG({
            llvm::dbgs() << "Can't get write access for equation\n";
            equationOp.printInline(llvm::dbgs());
            llvm::dbgs() << "\n";
          });

          return mlir::failure();
        }

        auto writtenVariable = writeAccess->getVariable();

        auto writtenVariableOp =
            symbolTableCollection.lookupSymbolIn<VariableOp>(modelOp,
                                                             writtenVariable);

        if (idaInstance->hasVariable(writtenVariableOp)) {
          LLVM_DEBUG({
            llvm::dbgs() << "Add equation writing to variable "
                         << writtenVariableOp.getSymName() << "\n";

            equationOp.printInline(llvm::dbgs());
            llvm::dbgs() << "\n";
          });

          idaInstance->addEquation(equationOp);
        }
      }
    }

    // The SCCs with cyclic dependencies among their equations.
    llvm::DenseSet<SCCOp> cycles;

    // The SCCs whose equations could not be made explicit.
    llvm::DenseSet<SCCOp> implicitSCCs;

    // Categorize the equations.
    LLVM_DEBUG(llvm::dbgs() << "Categorizing the equations\n");

    for (SCCOp scc : SCCs) {
      // The content of an SCC may be modified, so we need to freeze the
      // initial list of equations.
      llvm::SmallVector<MatchedEquationInstanceOp> sccEquations;
      scc.collectEquations(sccEquations);

      if (sccEquations.empty()) {
        continue;
      }

      if (sccEquations.size() > 1) {
        LLVM_DEBUG({
          llvm::dbgs() << "Cyclic equations\n";

          for (MatchedEquationInstanceOp equation : sccEquations) {
            equation.printInline(llvm::dbgs());
            llvm::dbgs() << "\n";
          }
        });

        cycles.insert(scc);
        continue;
      }

      MatchedEquationInstanceOp equation = sccEquations[0];

      LLVM_DEBUG({
        llvm::dbgs() << "Explicitating equation\n";
        equation.printInline(llvm::dbgs());
        llvm::dbgs() << "\n";
      });

      auto explicitEquation =
          equation.cloneAndExplicitate(rewriter, symbolTableCollection);

      if (explicitEquation) {
        LLVM_DEBUG({
          llvm::dbgs() << "Explicit equation\n";
          explicitEquation.printInline(llvm::dbgs());
          llvm::dbgs() << "\n";
          llvm::dbgs() << "Explicitable equation found\n";
        });

        auto explicitTemplate = explicitEquation.getTemplate();
        rewriter.eraseOp(explicitEquation);
        rewriter.eraseOp(explicitTemplate);
      } else {
        LLVM_DEBUG(llvm::dbgs() << "Implicit equation found\n");
        implicitSCCs.insert(scc);
        continue;
      }
    }

    // Add the cyclic equations to the set of equations managed by IDA,
    // together with their written variables.
    LLVM_DEBUG(llvm::dbgs() << "Add the cyclic equations\n");

    for (SCCOp scc : cycles) {
      if (mlir::failed(addMainModelSCC(symbolTableCollection, modelOp,
                                       derivativesMap, *idaInstance, scc))) {
        return mlir::failure();
      }
    }

    // Add the implicit equations to the set of equations managed by IDA,
    // together with their written variables.
    LLVM_DEBUG(llvm::dbgs() << "Add the implicit equations\n");

    for (SCCOp scc : implicitSCCs) {
      if (mlir::failed(addMainModelSCC(symbolTableCollection, modelOp,
                                       derivativesMap, *idaInstance, scc))) {
        return mlir::failure();
      }
    }
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Reduced system feature disabled\n");

    // Add all the non-read-only variables to IDA.
    for (VariableOp variable : variables) {
      auto variableName = mlir::SymbolRefAttr::get(variable.getSymNameAttr());

      if (derivativesMap.getDerivative(variableName)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Add state variable: " << variable.getSymName() << "\n");
        idaInstance->addStateVariable(variable);
      } else if (derivativesMap.getDerivedVariable(variableName)) {
        LLVM_DEBUG(llvm::dbgs() << "Add derivative variable: "
                                << variable.getSymName() << "\n");
        idaInstance->addDerivativeVariable(variable);
      } else if (!variable.isReadOnly()) {
        LLVM_DEBUG(llvm::dbgs() << "Add algebraic variable: "
                                << variable.getSymName() << "\n");
        idaInstance->addAlgebraicVariable(variable);
      }
    }
  }

  if (mlir::failed(addEquationsWritingToIDAVariables(
          symbolTableCollection, modelOp, *idaInstance, SCCs, externalSCCs,
          [&](SCCOp scc) {
            return addMainModelSCC(symbolTableCollection, modelOp,
                                   derivativesMap, *idaInstance, scc);
          }))) {
    return mlir::failure();
  }

  // Determine the SCCs that can be handled internally.
  llvm::SmallVector<SCCOp> internalSCCs;

  for (SCCOp scc : SCCs) {
    if (!externalSCCs.contains(scc)) {
      internalSCCs.push_back(scc);
    }
  }

  if (mlir::failed(
          idaInstance->declareInstance(rewriter, modelOp.getLoc(), moduleOp))) {
    return mlir::failure();
  }

  if (mlir::failed(createInitMainSolversFunction(
          rewriter, moduleOp, symbolTableCollection, modelOp.getLoc(), modelOp,
          idaInstance.get(), variables, SCCs))) {
    return mlir::failure();
  }

  if (mlir::failed(createDeinitMainSolversFunction(
          rewriter, moduleOp, modelOp.getLoc(), idaInstance.get()))) {
    return mlir::failure();
  }

  if (mlir::failed(createCalcICFunction(rewriter, moduleOp, modelOp.getLoc(),
                                        idaInstance.get()))) {
    return mlir::failure();
  }

  if (mlir::failed(createUpdateIDAVariablesFunction(
          rewriter, moduleOp, modelOp.getLoc(), idaInstance.get()))) {
    return mlir::failure();
  }

  if (mlir::failed(createUpdateNonIDAVariablesFunction(
          rewriter, moduleOp, symbolTableCollection, idaInstance.get(), modelOp,
          internalSCCs))) {
    return mlir::failure();
  }

  if (mlir::failed(createGetIDATimeFunction(
          rewriter, moduleOp, modelOp.getLoc(), idaInstance.get()))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult
IDAPass::addMainModelSCC(mlir::SymbolTableCollection &symbolTableCollection,
                         ModelOp modelOp, const DerivativesMap &derivativesMap,
                         IDAInstance &idaInstance, SCCOp scc) {
  for (MatchedEquationInstanceOp equation :
       scc.getOps<MatchedEquationInstanceOp>()) {
    if (mlir::failed(addMainModelEquation(symbolTableCollection, modelOp,
                                          derivativesMap, idaInstance,
                                          equation))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult IDAPass::addMainModelEquation(
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    const DerivativesMap &derivativesMap, IDAInstance &idaInstance,
    MatchedEquationInstanceOp equationOp) {
  LLVM_DEBUG({
    llvm::dbgs() << "Add equation\n";
    equationOp.printInline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  idaInstance.addEquation(equationOp);

  std::optional<VariableAccess> writeAccess =
      equationOp.getMatchedAccess(symbolTableCollection);

  if (!writeAccess) {
    return mlir::failure();
  }

  auto writtenVariable = writeAccess->getVariable();

  auto writtenVariableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
      modelOp, writtenVariable);

  assert(!writtenVariableOp.isReadOnly());

  if (derivativesMap.getDerivedVariable(writtenVariable)) {
    LLVM_DEBUG(llvm::dbgs() << "Add derivative variable: "
                            << writtenVariableOp.getSymName() << "\n");
    idaInstance.addDerivativeVariable(writtenVariableOp);
  } else if (derivativesMap.getDerivative(writtenVariable)) {
    LLVM_DEBUG(llvm::dbgs() << "Add state variable: "
                            << writtenVariableOp.getSymName() << "\n");
    idaInstance.addStateVariable(writtenVariableOp);
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Add algebraic variable: "
                            << writtenVariableOp.getSymName() << "\n");
    idaInstance.addAlgebraicVariable(writtenVariableOp);
  }

  return mlir::success();
}

mlir::LogicalResult IDAPass::addEquationsWritingToIDAVariables(
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    IDAInstance &idaInstance, llvm::ArrayRef<SCCOp> SCCs,
    llvm::DenseSet<SCCOp> &externalSCCs,
    llvm::function_ref<mlir::LogicalResult(SCCOp)> addFn) {
  // If any of the remaining equations manageable by MARCO does write on a
  // variable managed by IDA, then the equation must be passed to IDA even
  // if not strictly necessary. Avoiding this would require either memory
  // duplication or a more severe restructuring of the solving
  // infrastructure, which would have to be able to split variables and
  // equations according to which runtime solver manages such variables.

  LLVM_DEBUG(llvm::dbgs() << "Add the equations writing to IDA variables\n");

  bool atLeastOneSCCAdded;

  do {
    atLeastOneSCCAdded = false;

    for (SCCOp scc : SCCs) {
      if (externalSCCs.contains(scc)) {
        // Already externalized SCC.
        continue;
      }

      bool shouldAddSCC = false;

      for (MatchedEquationInstanceOp equationOp :
           scc.getOps<MatchedEquationInstanceOp>()) {
        std::optional<VariableAccess> writeAccess =
            equationOp.getMatchedAccess(symbolTableCollection);

        if (!writeAccess) {
          return mlir::failure();
        }

        auto writtenVariable = writeAccess->getVariable();

        auto writtenVariableOp =
            symbolTableCollection.lookupSymbolIn<VariableOp>(modelOp,
                                                             writtenVariable);

        if (idaInstance.hasVariable(writtenVariableOp)) {
          shouldAddSCC = true;
        }
      }

      if (shouldAddSCC) {
        externalSCCs.insert(scc);
        atLeastOneSCCAdded = true;

        if (mlir::failed(addFn(scc))) {
          return mlir::failure();
        }
      }
    }
  } while (atLeastOneSCCAdded);

  return mlir::success();
}

mlir::LogicalResult IDAPass::createInitMainSolversFunction(
    mlir::IRRewriter &rewriter, mlir::ModuleOp moduleOp,
    mlir::SymbolTableCollection &symbolTableCollection, mlir::Location loc,
    ModelOp modelOp, IDAInstance *idaInstance,
    llvm::ArrayRef<VariableOp> variableOps,
    llvm::ArrayRef<SCCOp> allSCCs) const {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = rewriter.create<mlir::runtime::DynamicModelBeginOp>(loc);

  rewriter.createBlock(&functionOp.getBodyRegion());
  rewriter.setInsertionPointToStart(functionOp.getBody());

  if (mlir::failed(idaInstance->initialize(rewriter, loc))) {
    return mlir::failure();
  }

  if (mlir::failed(idaInstance->configure(rewriter, loc, moduleOp, modelOp,
                                          variableOps, allSCCs))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult IDAPass::createDeinitMainSolversFunction(
    mlir::OpBuilder &builder, mlir::ModuleOp moduleOp, mlir::Location loc,
    IDAInstance *idaInstance) const {
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = builder.create<mlir::runtime::DynamicModelEndOp>(loc);
  builder.createBlock(&functionOp.getBodyRegion());
  builder.setInsertionPointToStart(functionOp.getBody());

  if (mlir::failed(idaInstance->deleteInstance(builder, loc))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult
IDAPass::createCalcICFunction(mlir::OpBuilder &builder, mlir::ModuleOp moduleOp,
                              mlir::Location loc,
                              IDAInstance *idaInstance) const {
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = builder.create<mlir::runtime::FunctionOp>(
      loc, "calcIC", builder.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block *entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  if (mlir::failed(idaInstance->performCalcIC(builder, loc))) {
    return mlir::failure();
  }

  builder.create<mlir::runtime::ReturnOp>(loc, std::nullopt);
  return mlir::success();
}

mlir::LogicalResult IDAPass::createUpdateIDAVariablesFunction(
    mlir::OpBuilder &builder, mlir::ModuleOp moduleOp, mlir::Location loc,
    IDAInstance *idaInstance) const {
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = builder.create<mlir::runtime::FunctionOp>(
      loc, "updateIDAVariables",
      builder.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block *entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  if (mlir::failed(idaInstance->performStep(builder, loc))) {
    return mlir::failure();
  }

  builder.create<mlir::runtime::ReturnOp>(loc, std::nullopt);
  return mlir::success();
}

mlir::LogicalResult IDAPass::createUpdateNonIDAVariablesFunction(
    mlir::RewriterBase &rewriter, mlir::ModuleOp moduleOp,
    mlir::SymbolTableCollection &symbolTableCollection,
    IDAInstance *idaInstance, ModelOp modelOp,
    llvm::ArrayRef<SCCOp> internalSCCs) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  // Create the function.
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = rewriter.create<mlir::runtime::FunctionOp>(
      modelOp.getLoc(), "updateNonIDAVariables",
      rewriter.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block *entryBlock = functionOp.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  if (!internalSCCs.empty()) {
    // Create the schedule operation.
    rewriter.setInsertionPointToEnd(modelOp.getBody());
    auto scheduleOp = rewriter.create<ScheduleOp>(modelOp.getLoc(), "dynamic");
    rewriter.createBlock(&scheduleOp.getBodyRegion());
    rewriter.setInsertionPointToStart(scheduleOp.getBody());

    auto dynamicOp = rewriter.create<DynamicOp>(modelOp.getLoc());
    rewriter.createBlock(&dynamicOp.getBodyRegion());
    rewriter.setInsertionPointToStart(dynamicOp.getBody());

    for (SCCOp scc : internalSCCs) {
      scc->moveBefore(dynamicOp.getBody(), dynamicOp.getBody()->end());
    }

    // Call the schedule.
    rewriter.setInsertionPointToEnd(entryBlock);
    rewriter.create<RunScheduleOp>(modelOp.getLoc(), scheduleOp);
  }

  rewriter.create<mlir::runtime::ReturnOp>(modelOp.getLoc(), std::nullopt);
  return mlir::success();
}

mlir::LogicalResult
IDAPass::createGetIDATimeFunction(mlir::OpBuilder &builder,
                                  mlir::ModuleOp moduleOp, mlir::Location loc,
                                  IDAInstance *idaInstance) const {
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = builder.create<mlir::runtime::FunctionOp>(
      loc, "getIDATime",
      builder.getFunctionType(std::nullopt, builder.getF64Type()));

  mlir::Block *entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  mlir::Value time = idaInstance->getCurrentTime(builder, loc);
  builder.create<mlir::runtime::ReturnOp>(loc, time);
  return mlir::success();
}

mlir::LogicalResult IDAPass::cleanModelOp(ModelOp modelOp) {
  mlir::RewritePatternSet patterns(&getContext());
  ModelOp::getCleaningPatterns(patterns, &getContext());
  return mlir::applyPatternsAndFoldGreedily(modelOp, std::move(patterns));
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createIDAPass() {
  return std::make_unique<IDAPass>();
}

std::unique_ptr<mlir::Pass> createIDAPass(const IDAPassOptions &options) {
  return std::make_unique<IDAPass>(options);
}
} // namespace mlir::bmodelica
