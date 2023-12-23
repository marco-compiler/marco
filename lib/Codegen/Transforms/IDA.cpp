#include "marco/Codegen/Transforms/IDA.h"
#include "marco/Codegen/Transforms/SolverPassBase.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "marco/Codegen/Transforms/Solvers/IDAInstance.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ida"

namespace mlir::modelica
{
#define GEN_PASS_DEF_IDAPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  class IDAPass : public mlir::modelica::impl::IDAPassBase<IDAPass>,
                  public mlir::modelica::impl::ModelSolver
  {
    public:
      using IDAPassBase::IDAPassBase;

      void runOnOperation() override;

    private:
      DerivativesMap& getDerivativesMap(ModelOp modelOp);

    protected:
      mlir::LogicalResult solveICModel(
          mlir::IRRewriter& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variableOps,
          const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
          llvm::ArrayRef<SCCOp> SCCs) override;

      mlir::LogicalResult solveMainModel(
          mlir::IRRewriter& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variableOps,
          const DerivativesMap& derivativesMap,
          const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
          llvm::ArrayRef<SCCOp> SCCs) override;

    private:
      /// Add an equation to the IDA instance together with its written
      /// variable.
      mlir::LogicalResult addICModelEquation(
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          IDAInstance& idaInstance,
          ScheduledEquationInstanceOp equationOp);

      /// Add an equation to the IDA instance together with its written
      /// variable.
      mlir::LogicalResult addMainModelEquation(
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          const DerivativesMap& derivativesMap,
          IDAInstance& idaInstance,
          ScheduledEquationInstanceOp equationOp);

      /// Create the function that instantiates the external solvers to be used
      /// during the IC computation.
      mlir::LogicalResult createInitICSolversFunction(
          mlir::IRRewriter& rewriter,
          mlir::ModuleOp moduleOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::Location loc,
          ModelOp modelOp,
          IDAInstance* idaInstance,
          llvm::ArrayRef<VariableOp> variableOps,
          const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
          llvm::ArrayRef<SCCOp> SCCs) const;

      /// Create the function that deallocates the external solvers used during
      /// the IC computation.
      mlir::LogicalResult createDeinitICSolversFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::Location loc,
          IDAInstance* idaInstance) const;

      /// Create the function that instantiates the external solvers to be used
      /// during the simulation.
      mlir::LogicalResult createInitMainSolversFunction(
          mlir::IRRewriter& rewriter,
          mlir::ModuleOp moduleOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::Location loc,
          ModelOp modelOp,
          IDAInstance* idaInstance,
          llvm::ArrayRef<VariableOp> variableOps,
          const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
          llvm::ArrayRef<SCCOp> SCCs) const;

      /// Create the function that deallocates the external solvers used during
      /// the simulation.
      mlir::LogicalResult createDeinitMainSolversFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::Location loc,
          IDAInstance* idaInstance) const;

      /// Create the function that computes the initial conditions of the
      /// "initial conditions model".
      mlir::LogicalResult createSolveICModelFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::Location loc,
          IDAInstance* idaInstance,
          llvm::ArrayRef<SCCOp> SCCs,
          const llvm::DenseMap<
              ScheduledEquationInstanceOp,
              RawFunctionOp>& equationFunctions) const;

      /// Create the function that computes the initial conditions of the "main
      /// model".
      mlir::LogicalResult createCalcICFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::Location loc,
          IDAInstance* idaInstance) const;

      /// Create the functions that calculates the values that the variables
      /// belonging to IDA will have in the next iteration.
      mlir::LogicalResult createUpdateIDAVariablesFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::Location loc,
          IDAInstance* idaInstance) const;

      /// Create the functions that calculates the values that the variables
      /// not belonging to IDA will have in the next iteration.
      mlir::LogicalResult createUpdateNonIDAVariablesFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::Location loc,
          IDAInstance* idaInstance,
          llvm::ArrayRef<SCCOp> SCCs,
          const llvm::DenseMap<
              ScheduledEquationInstanceOp, RawFunctionOp>& equationFunctions) const;

      /// Create the function to be used to get the time reached by IDA.
      mlir::LogicalResult createGetIDATimeFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::Location loc,
          IDAInstance* idaInstance) const;
  };
}

void IDAPass::runOnOperation()
{
  mlir::ModuleOp moduleOp = getOperation();
  llvm::SmallVector<ModelOp, 1> modelOps;

  for (ModelOp modelOp : moduleOp.getOps<ModelOp>()) {
    modelOps.push_back(modelOp);
  }

  auto expectedVariablesFilter =
      marco::VariableFilter::fromString(variablesFilter);

  std::unique_ptr<marco::VariableFilter> variablesFilterInstance;

  if (!expectedVariablesFilter) {
    getOperation().emitWarning()
        << "Invalid variable filter string. No filtering will take place";

    variablesFilterInstance = std::make_unique<marco::VariableFilter>();
  } else {
    variablesFilterInstance = std::make_unique<marco::VariableFilter>(
        std::move(*expectedVariablesFilter));
  }

  for (ModelOp modelOp : modelOps) {
    if (mlir::failed(convert(
            modelOp, getDerivativesMap(modelOp), *variablesFilterInstance,
            processICModel, processMainModel))) {
      return signalPassFailure();
    }
  }
}

DerivativesMap& IDAPass::getDerivativesMap(ModelOp modelOp)
{
  if (auto analysis = getCachedChildAnalysis<DerivativesMap>(modelOp)) {
    return *analysis;
  }

  auto& analysis = getChildAnalysis<DerivativesMap>(modelOp);
  analysis.initialize();
  return analysis;
}

mlir::LogicalResult IDAPass::solveICModel(
    mlir::IRRewriter& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variableOps,
    const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
    llvm::ArrayRef<SCCOp> SCCs)
{
  LLVM_DEBUG(llvm::dbgs() << "Solving the 'initial conditions' model\n");
  auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();

  auto idaInstance = std::make_unique<IDAInstance>(
      "ida_ic", symbolTableCollection, nullptr,
      reducedSystem, reducedDerivatives, jacobianOneSweep, debugInformation);

  // Set the start and end times.
  idaInstance->setStartTime(0);
  idaInstance->setEndTime(0);

  // Map from explicit equations to their algorithmically equivalent function.
  llvm::DenseMap<
      ScheduledEquationInstanceOp, RawFunctionOp> equationFunctions;

  if (reducedSystem) {
    LLVM_DEBUG(llvm::dbgs() << "Reduced system feature enabled\n");

    // The list of strongly connected components with cyclic dependencies.
    llvm::SmallVector<SCCOp> cycles;

    // Determine which equations can be processed by just making them explicit
    // with respect to the variable they match.
    llvm::DenseSet<ScheduledEquationInstanceOp> explicitableEquations;

    // Map between the original equations and their explicit version (if
    // computable) w.r.t. the matched variables.
    llvm::DenseMap<
        ScheduledEquationInstanceOp,
        ScheduledEquationInstanceOp> explicitEquationsMap;

    // The list of equations which could not be made explicit.
    llvm::DenseSet<ScheduledEquationInstanceOp> implicitEquations;

    // The list of equations that can be handled internally.
    llvm::DenseSet<ScheduledEquationInstanceOp> internalEquations;

    // The list of equations that are handled by IDA.
    llvm::DenseSet<ScheduledEquationInstanceOp> externalEquations;

    auto isExternalEquationFn = [&](ScheduledEquationInstanceOp equationOp) {
      for (SCCOp scc : cycles) {
        for (ScheduledEquationInstanceOp sccEquation :
             scc.getOps<ScheduledEquationInstanceOp>()) {
          if (sccEquation == equationOp) {
            return true;
          }
        }
      }

      return implicitEquations.contains(equationOp);
    };

    // Categorize the equations.
    LLVM_DEBUG(llvm::dbgs() << "Identifying the explicitable equations\n");

    for (SCCOp scc : SCCs) {
      if (scc.getCycle()) {
        cycles.push_back(scc);
      } else {
        // The content of an SCC may be modified, so we need to freeze the
        // initial list of equations.
        llvm::SmallVector<ScheduledEquationInstanceOp> equationOps;
        scc.collectEquations(equationOps);

        for (ScheduledEquationInstanceOp equationOp : equationOps) {
          LLVM_DEBUG({
              llvm::dbgs() << "Explicitating equation\n";
              equationOp.printInline(llvm::dbgs());
              llvm::dbgs() << "\n";
          });

          auto explicitEquationOp = equationOp.cloneAndExplicitate(
              rewriter, symbolTableCollection);

          if (explicitEquationOp) {
            LLVM_DEBUG({
                llvm::dbgs() << "Explicit equation\n";
                explicitEquationOp.printInline(llvm::dbgs());
                llvm::dbgs() << "\n";
            });

            explicitableEquations.insert(equationOp);
            explicitEquationsMap[equationOp] = explicitEquationOp;
          } else {
            LLVM_DEBUG(llvm::dbgs() << "Implicit equation found\n");
            implicitEquations.insert(equationOp);
            continue;
          }
        }
      }
    }

    // Determine the equations that can be handled internally.
    for (ScheduledEquationInstanceOp equationOp : explicitableEquations) {
      if (!isExternalEquationFn(equationOp)) {
        internalEquations.insert(equationOp);
      }
    }

    // Add the cyclic equations to the set of equations managed by IDA,
    // together with their written variables.
    LLVM_DEBUG(llvm::dbgs() << "Add the cyclic equations\n");

    for (SCCOp scc : cycles) {
      for (ScheduledEquationInstanceOp equationOp :
           scc.getOps<ScheduledEquationInstanceOp>()) {
        if (mlir::failed(addICModelEquation(
                symbolTableCollection, modelOp, *idaInstance, equationOp))) {
          return mlir::failure();
        }
      }
    }

    // Add the implicit equations to the set of equations managed by IDA,
    // together with their written variables.
    LLVM_DEBUG(llvm::dbgs() << "Add the implicit equations\n");

    for (ScheduledEquationInstanceOp equationOp : implicitEquations) {
      if (mlir::failed(addICModelEquation(
              symbolTableCollection, modelOp, *idaInstance, equationOp))) {
        return mlir::failure();
      }
    }

    // Create the functions for the equations managed internally.
    size_t equationFunctionsCounter = 0;

    for (ScheduledEquationInstanceOp equationOp : internalEquations) {
      ScheduledEquationInstanceOp explicitEquationOp =
          explicitEquationsMap[equationOp];

      RawFunctionOp templateFunction = createEquationTemplateFunction(
          rewriter, moduleOp, symbolTableCollection,
          explicitEquationOp.getTemplate(),
          explicitEquationOp.getViewElementIndex(),
          explicitEquationOp.getIterationDirections(),
          "initial_equation_" +
              std::to_string(equationFunctionsCounter++),
          localToGlobalVariablesMap);

      equationFunctions[equationOp] = templateFunction;
    }
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Reduced system feature disabled");

    // Add all the variables to IDA.
    for (VariableOp variableOp : variableOps) {
      LLVM_DEBUG(llvm::dbgs() << "Add algebraic variable: "
                              << variableOp.getSymName() << "\n");

      idaInstance->addAlgebraicVariable(variableOp);
    }
  }

  // If any of the remaining equations manageable by MARCO does write on a
  // variable managed by IDA, then the equation must be passed to IDA even
  // if not strictly necessary. Avoiding this would require either memory
  // duplication or a more severe restructuring of the solving
  // infrastructure, which would have to be able to split variables and
  // equations according to which runtime solver manages such variables.
  LLVM_DEBUG(llvm::dbgs() << "Add the equations writing to IDA variables\n");

  for (SCCOp scc : SCCs) {
    for (ScheduledEquationInstanceOp equationOp :
         scc.getOps<ScheduledEquationInstanceOp>()) {
      std::optional<VariableAccess> writeAccess =
          equationOp.getMatchedAccess(symbolTableCollection);

      if (!writeAccess) {
        return mlir::failure();
      }

      auto writtenVariable = writeAccess->getVariable();

      auto writtenVariableOp =
          symbolTableCollection.lookupSymbolIn<VariableOp>(
              modelOp, writtenVariable);

      if (idaInstance->hasVariable(writtenVariableOp)) {
        LLVM_DEBUG({
            llvm::dbgs() << "Add equation\n";
            equationOp.printInline(llvm::dbgs());
            llvm::dbgs() << "\n";
        });

        idaInstance->addEquation(equationOp);
      }
    }
  }

  if (mlir::failed(idaInstance->declareInstance(
          rewriter, modelOp.getLoc(), moduleOp))) {
    return mlir::failure();
  }

  if (mlir::failed(createInitICSolversFunction(
          rewriter, moduleOp, symbolTableCollection, modelOp.getLoc(), modelOp,
          idaInstance.get(), variableOps, localToGlobalVariablesMap, SCCs))) {
    return mlir::failure();
  }

  if (mlir::failed(createDeinitICSolversFunction(
          rewriter, moduleOp, modelOp.getLoc(), idaInstance.get()))) {
    return mlir::failure();
  }

  if (mlir::failed(createSolveICModelFunction(
          rewriter, moduleOp, symbolTableCollection, modelOp.getLoc(),
          idaInstance.get(), SCCs, equationFunctions))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult IDAPass::addICModelEquation(
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    IDAInstance& idaInstance,
    ScheduledEquationInstanceOp equationOp)
{
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

  auto writtenVariableOp =
      symbolTableCollection.lookupSymbolIn<VariableOp>(
          modelOp, writeAccess->getVariable());

  LLVM_DEBUG(llvm::dbgs() << "Add algebraic variable: "
                          << writtenVariableOp.getSymName() << "\n");

  idaInstance.addAlgebraicVariable(writtenVariableOp);
  return mlir::success();
}

mlir::LogicalResult IDAPass::solveMainModel(
    mlir::IRRewriter& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variableOps,
    const DerivativesMap& derivativesMap,
    const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
    llvm::ArrayRef<SCCOp> SCCs)
{
  LLVM_DEBUG(llvm::dbgs() << "Solving the 'main' model\n");
  auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();

  auto idaInstance = std::make_unique<IDAInstance>(
      "ida_main", symbolTableCollection, &derivativesMap,
      reducedSystem, reducedDerivatives, jacobianOneSweep, debugInformation);

  // Map from explicit equations to their algorithmically equivalent function.
  llvm::DenseMap<
      ScheduledEquationInstanceOp, RawFunctionOp> equationFunctions;

  if (reducedSystem) {
    LLVM_DEBUG(llvm::dbgs() << "Reduced system feature enabled");

    // Add the state and derivative variables.
    // All of them must always be known to IDA.

    for (VariableOp variableOp : variableOps) {
      if (auto derivative = derivativesMap.getDerivative(
              mlir::SymbolRefAttr::get(variableOp.getSymNameAttr()))) {
        assert(derivative->getNestedReferences().empty());

        auto derivativeVariableOp =
            symbolTableCollection.lookupSymbolIn<VariableOp>(
                modelOp, derivative->getRootReference());

        assert(derivativeVariableOp && "Derivative not found");

        LLVM_DEBUG(llvm::dbgs() << "Add state variable: "
                                << variableOp.getSymName() << "\n");
        idaInstance->addStateVariable(variableOp);

        LLVM_DEBUG(llvm::dbgs() << "Add derivative variable: "
                                << derivativeVariableOp.getSymName() << "\n");
        idaInstance->addDerivativeVariable(derivativeVariableOp);
      }
    }

    // Add the equations writing to variables handled by IDA.
    for (SCCOp scc : SCCs) {
      for (ScheduledEquationInstanceOp equationOp :
           scc.getOps<ScheduledEquationInstanceOp>()) {
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
            symbolTableCollection.lookupSymbolIn<VariableOp>(
                modelOp, writtenVariable);

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

    // Determine which equations can be processed by just making them explicit
    // with respect to the variable they match.
    llvm::DenseSet<ScheduledEquationInstanceOp> explicitEquations;
    llvm::SmallVector<SCCOp> cycles;
    llvm::SmallVector<ScheduledEquationInstanceOp> implicitEquations;

    for (SCCOp scc : SCCs) {
      if (scc.getCycle()) {
        cycles.push_back(scc);
      } else {
        // The content of an SCC may be modified, so we need to freeze the
        // initial list of equations.
        llvm::SmallVector<ScheduledEquationInstanceOp> equationOps;
        scc.collectEquations(equationOps);

        for (ScheduledEquationInstanceOp equationOp : equationOps) {
          if (idaInstance->hasEquation(equationOp)) {
            // Skip the equation if it is already handled by IDA.
            LLVM_DEBUG({
                llvm::dbgs() << "Equation already handled by IDA\n";
                equationOp.printInline(llvm::dbgs());
                llvm::dbgs() << "\n";
            });

            continue;
          }

          LLVM_DEBUG({
              llvm::dbgs() << "Explicitating equation\n";
              equationOp.printInline(llvm::dbgs());
              llvm::dbgs() << "\n";
          });

          auto explicitEquationOp =
              equationOp.cloneAndExplicitate(rewriter, symbolTableCollection);

          if (explicitEquationOp) {
            LLVM_DEBUG({
                llvm::dbgs() << "Add explicit equation\n";
                explicitEquationOp.printInline(llvm::dbgs());
            });

            explicitEquations.insert(explicitEquationOp);
            rewriter.eraseOp(equationOp);
          } else {
            LLVM_DEBUG(llvm::dbgs() << "Implicit equation found\n");
            implicitEquations.push_back(equationOp);
            continue;
          }
        }
      }
    }

    size_t equationFunctionsCounter = 0;

    for (ScheduledEquationInstanceOp equationOp : explicitEquations) {
      RawFunctionOp templateFunction = createEquationTemplateFunction(
          rewriter, moduleOp, symbolTableCollection,
          equationOp.getTemplate(),
          equationOp.getViewElementIndex(),
          equationOp.getIterationDirections(),
          "equation_" + std::to_string(equationFunctionsCounter++),
          localToGlobalVariablesMap);

      equationFunctions[equationOp] = templateFunction;
    }

    // Add the implicit equations to the set of equations managed by IDA,
    // together with their written variables.
    LLVM_DEBUG(llvm::dbgs() << "Add the implicit equations\n");

    for (ScheduledEquationInstanceOp equationOp : implicitEquations) {
      if (mlir::failed(addMainModelEquation(
              symbolTableCollection, modelOp, derivativesMap, *idaInstance,
              equationOp))) {
        return mlir::failure();
      }
    }

    // Add the cyclic equations to the set of equations managed by IDA,
    // together with their written variables.
    LLVM_DEBUG(llvm::dbgs() << "Add the cyclic equations\n");

    for (SCCOp scc : cycles) {
      for (ScheduledEquationInstanceOp equationOp :
           scc.getOps<ScheduledEquationInstanceOp>()) {
        if (mlir::failed(addMainModelEquation(
                symbolTableCollection, modelOp, derivativesMap, *idaInstance,
                equationOp))) {
          return mlir::failure();
        }
      }
    }
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Reduced system feature disabled");

    // Add all the variables to IDA.
    for (VariableOp variableOp : variableOps) {
      auto variableName =
          mlir::SymbolRefAttr::get(variableOp.getSymNameAttr());

      if (derivativesMap.getDerivative(variableName)) {
        LLVM_DEBUG(llvm::dbgs() << "Add state variable: "
                                << variableOp.getSymName() << "\n");
        idaInstance->addStateVariable(variableOp);
      } else if (derivativesMap.getDerivedVariable(variableName)) {
        LLVM_DEBUG(llvm::dbgs() << "Add derivative variable: "
                                << variableOp.getSymName() << "\n");
        idaInstance->addDerivativeVariable(variableOp);
      } else if (!variableOp.isReadOnly()) {
        LLVM_DEBUG(llvm::dbgs() << "Add algebraic variable: "
                                << variableOp.getSymName() << "\n");
        idaInstance->addAlgebraicVariable(variableOp);
      }
    }
  }

  // If any of the remaining equations manageable by MARCO does write on a
  // variable managed by IDA, then the equation must be passed to IDA even
  // if not strictly necessary. Avoiding this would require either memory
  // duplication or a more severe restructuring of the solving
  // infrastructure, which would have to be able to split variables and
  // equations according to which runtime solver manages such variables.
  LLVM_DEBUG(llvm::dbgs() << "Add the equations writing to IDA variables\n");

  for (SCCOp scc : SCCs) {
    for (ScheduledEquationInstanceOp equationOp :
         scc.getOps<ScheduledEquationInstanceOp>()) {
      std::optional<VariableAccess> writeAccess =
          equationOp.getMatchedAccess(symbolTableCollection);

      if (!writeAccess) {
        return mlir::failure();
      }

      auto writtenVariable = writeAccess->getVariable();

      auto writtenVariableOp =
          symbolTableCollection.lookupSymbolIn<VariableOp>(
              modelOp, writtenVariable);

      if (idaInstance->hasVariable(writtenVariableOp)) {
        LLVM_DEBUG({
            llvm::dbgs() << "Add equation\n";
            equationOp.printInline(llvm::dbgs());
            llvm::dbgs() << "\n";
        });

        idaInstance->addEquation(equationOp);
      }
    }
  }

  if (mlir::failed(idaInstance->declareInstance(
          rewriter, modelOp.getLoc(), moduleOp))) {
    return mlir::failure();
  }

  if (mlir::failed(createInitMainSolversFunction(
          rewriter, moduleOp, symbolTableCollection, modelOp.getLoc(), modelOp,
          idaInstance.get(), variableOps, localToGlobalVariablesMap, SCCs))) {
    return mlir::failure();
  }

  if (mlir::failed(createDeinitMainSolversFunction(
          rewriter, moduleOp, modelOp.getLoc(), idaInstance.get()))) {
    return mlir::failure();
  }

  if (mlir::failed(createCalcICFunction(
          rewriter, moduleOp, modelOp.getLoc(), idaInstance.get()))) {
    return mlir::failure();
  }

  if (mlir::failed(createUpdateIDAVariablesFunction(
          rewriter, moduleOp, modelOp.getLoc(), idaInstance.get()))) {
    return mlir::failure();
  }

  if (mlir::failed(createUpdateNonIDAVariablesFunction(
          rewriter, moduleOp, symbolTableCollection, modelOp.getLoc(),
          idaInstance.get(), SCCs, equationFunctions))) {
    return mlir::failure();
  }

  if (mlir::failed(createGetIDATimeFunction(
          rewriter, moduleOp, modelOp.getLoc(), idaInstance.get()))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult IDAPass::addMainModelEquation(
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    const DerivativesMap& derivativesMap,
    IDAInstance& idaInstance,
    ScheduledEquationInstanceOp equationOp)
{
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

  auto writtenVariableOp =
      symbolTableCollection.lookupSymbolIn<VariableOp>(
          modelOp, writtenVariable);

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

mlir::LogicalResult IDAPass::createInitICSolversFunction(
    mlir::IRRewriter& rewriter,
    mlir::ModuleOp moduleOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::Location loc,
    ModelOp modelOp,
    IDAInstance* idaInstance,
    llvm::ArrayRef<VariableOp> variableOps,
    const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
    llvm::ArrayRef<SCCOp> SCCs) const
{
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = rewriter.create<mlir::simulation::FunctionOp>(
      loc, "initICSolvers",
      rewriter.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  if (mlir::failed(idaInstance->initialize(rewriter, loc))) {
    return mlir::failure();
  }

  if (mlir::failed(idaInstance->configure(
          rewriter, loc, moduleOp, modelOp, variableOps,
          localToGlobalVariablesMap, SCCs))) {
    return mlir::failure();
  }

  rewriter.create<mlir::simulation::ReturnOp>(loc, std::nullopt);
  return mlir::success();
}

mlir::LogicalResult IDAPass::createDeinitICSolversFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    mlir::Location loc,
    IDAInstance* idaInstance) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "deinitICSolvers",
      builder.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Deallocate the IDA solver.
  if (mlir::failed(idaInstance->deleteInstance(builder, loc))) {
    return mlir::failure();
  }

  builder.create<mlir::simulation::ReturnOp>(loc, std::nullopt);
  return mlir::success();
}

mlir::LogicalResult IDAPass::createInitMainSolversFunction(
    mlir::IRRewriter& rewriter,
    mlir::ModuleOp moduleOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::Location loc,
    ModelOp modelOp,
    IDAInstance* idaInstance,
    llvm::ArrayRef<VariableOp> variableOps,
    const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
    llvm::ArrayRef<SCCOp> SCCs) const
{
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = rewriter.create<mlir::simulation::FunctionOp>(
      loc, "initMainSolvers",
      rewriter.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  if (mlir::failed(idaInstance->initialize(rewriter, loc))) {
    return mlir::failure();
  }

  if (mlir::failed(idaInstance->configure(
          rewriter, loc, moduleOp, modelOp, variableOps,
          localToGlobalVariablesMap, SCCs))) {
    return mlir::failure();
  }

  rewriter.create<mlir::simulation::ReturnOp>(loc, std::nullopt);
  return mlir::success();
}

mlir::LogicalResult IDAPass::createDeinitMainSolversFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    mlir::Location loc,
    IDAInstance* idaInstance) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "deinitMainSolvers",
      builder.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  if (mlir::failed(idaInstance->deleteInstance(builder, loc))) {
    return mlir::failure();
  }

  builder.create<mlir::simulation::ReturnOp>(loc, std::nullopt);
  return mlir::success();
}

mlir::LogicalResult IDAPass::createSolveICModelFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::Location loc,
    IDAInstance* idaInstance,
    llvm::ArrayRef<SCCOp> SCCs,
    const llvm::DenseMap<
        ScheduledEquationInstanceOp, RawFunctionOp>& equationFunctions) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "solveICModel",
      builder.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Compute the initial conditions for the equations managed by IDA.
  if (mlir::failed(idaInstance->performCalcIC(builder, loc))) {
    return mlir::failure();
  }

  // Call the equation functions for the equations managed internally.
  for (SCCOp scc : SCCs) {
    for (ScheduledEquationInstanceOp equation :
         scc.getOps<ScheduledEquationInstanceOp>()) {
      auto rawFunctionIt = equationFunctions.find(equation);

      if (rawFunctionIt == equationFunctions.end()) {
        // Equation not handled internally.
        continue;
      }

      RawFunctionOp equationRawFunction = rawFunctionIt->getSecond();

      if (mlir::failed(callEquationFunction(
              builder, loc, equation, equationRawFunction))) {
        return mlir::failure();
      }
    }
  }

  builder.setInsertionPointToEnd(entryBlock);
  builder.create<mlir::simulation::ReturnOp>(loc, std::nullopt);
  return mlir::success();
}

mlir::LogicalResult IDAPass::createCalcICFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    mlir::Location loc,
    IDAInstance* idaInstance) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "calcIC",
      builder.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  if (mlir::failed(idaInstance->performCalcIC(builder, loc))) {
    return mlir::failure();
  }

  builder.create<mlir::simulation::ReturnOp>(loc, std::nullopt);
  return mlir::success();
}

mlir::LogicalResult IDAPass::createUpdateIDAVariablesFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    mlir::Location loc,
    IDAInstance* idaInstance) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "updateIDAVariables",
      builder.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  if (mlir::failed(idaInstance->performStep(builder, loc))) {
    return mlir::failure();
  }

  builder.create<mlir::simulation::ReturnOp>(loc, std::nullopt);
  return mlir::success();
}

mlir::LogicalResult IDAPass::createUpdateNonIDAVariablesFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::Location loc,
    IDAInstance* idaInstance,
    llvm::ArrayRef<SCCOp> SCCs,
    const llvm::DenseMap<
        ScheduledEquationInstanceOp, RawFunctionOp>& equationFunctions) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "updateNonIDAVariables",
      builder.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Call the equation functions for the equations managed internally.
  for (SCCOp scc : SCCs) {
    for (ScheduledEquationInstanceOp equation :
         scc.getOps<ScheduledEquationInstanceOp>()) {
      auto rawFunctionIt = equationFunctions.find(equation);

      if (rawFunctionIt == equationFunctions.end()) {
        // Equation not handled internally.
        continue;
      }

      RawFunctionOp equationRawFunction = rawFunctionIt->getSecond();

      if (mlir::failed(callEquationFunction(
              builder, loc, equation, equationRawFunction))) {
        return mlir::failure();
      }
    }
  }

  builder.create<mlir::simulation::ReturnOp>(loc, std::nullopt);
  return mlir::success();
}

mlir::LogicalResult IDAPass::createGetIDATimeFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp moduleOp,
    mlir::Location loc,
    IDAInstance* idaInstance) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = builder.create<mlir::simulation::FunctionOp>(
      loc, "getIDATime",
      builder.getFunctionType(std::nullopt, builder.getF64Type()));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  mlir::Value time = idaInstance->getCurrentTime(builder, loc);
  builder.create<mlir::simulation::ReturnOp>(loc, time);
  return mlir::success();
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createIDAPass()
  {
    return std::make_unique<IDAPass>();
  }

  std::unique_ptr<mlir::Pass> createIDAPass(const IDAPassOptions& options)
  {
    return std::make_unique<IDAPass>(options);
  }
}
