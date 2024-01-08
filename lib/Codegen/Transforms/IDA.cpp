#include "marco/Codegen/Transforms/IDA.h"
#include "marco/Codegen/Transforms/Solvers/Common.h"
#include "marco/Codegen/Transforms/Solvers/IDAInstance.h"
#include "marco/Dialect/IDA/IDADialect.h"
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

      mlir::LogicalResult processModelOp(ModelOp modelOp);

      mlir::LogicalResult solveICModel(
          mlir::IRRewriter& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variables,
          llvm::ArrayRef<SCCOp> SCCs);

      mlir::LogicalResult solveMainModel(
          mlir::IRRewriter& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variables,
          llvm::ArrayRef<SCCOp> SCCs);

      /// Add a SCC to the IDA instance.
      mlir::LogicalResult addICModelSCC(
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          IDAInstance& idaInstance,
          SCCOp scc);

      /// Add an equation to the IDA instance together with its written
      /// variable.
      mlir::LogicalResult addICModelEquation(
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          IDAInstance& idaInstance,
        MatchedEquationInstanceOp equationOp);

      /// Add a SCC to the IDA instance.
      mlir::LogicalResult addMainModelSCC(
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          const DerivativesMap& derivativesMap,
          IDAInstance& idaInstance,
          SCCOp scc);

      /// Add an equation to the IDA instance together with its written
      /// variable.
      mlir::LogicalResult addMainModelEquation(
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          const DerivativesMap& derivativesMap,
          IDAInstance& idaInstance,
          MatchedEquationInstanceOp equationOp);

      mlir::LogicalResult addEquationsWritingToIDAVariables(
          mlir::SymbolTableCollection& symbolTableCollection,
          ModelOp modelOp,
          IDAInstance& idaInstance,
          llvm::ArrayRef<SCCOp> SCCs,
          llvm::DenseSet<SCCOp>& externalSCCs,
          llvm::function_ref<mlir::LogicalResult(SCCOp)> addFn);

      /// Create the function that instantiates the external solvers to be used
      /// during the IC computation.
      mlir::LogicalResult createInitICSolversFunction(
          mlir::IRRewriter& rewriter,
          mlir::ModuleOp moduleOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::Location loc,
          ModelOp modelOp,
          IDAInstance* idaInstance,
          llvm::ArrayRef<VariableOp> variables,
          llvm::ArrayRef<SCCOp> allSCCs) const;

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
          llvm::ArrayRef<SCCOp> allSCCs) const;

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
          mlir::RewriterBase& rewriter,
          mlir::ModuleOp moduleOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          IDAInstance* idaInstance,
          ModelOp modelOp,
          llvm::ArrayRef<SCCOp> internalSCCs);

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
          mlir::RewriterBase& rewriter,
          mlir::ModuleOp moduleOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          IDAInstance* idaInstance,
          ModelOp modelOp,
          llvm::ArrayRef<SCCOp> internalSCCs);

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
  moduleOp.dump();
  llvm::SmallVector<ModelOp, 1> modelOps;

  for (ModelOp modelOp : moduleOp.getOps<ModelOp>()) {
    modelOps.push_back(modelOp);
  }

  for (ModelOp modelOp : modelOps) {
    if (mlir::failed(processModelOp(modelOp))) {
      return signalPassFailure();
    }
  }

  markAnalysesPreserved<DerivativesMap>();
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

mlir::LogicalResult IDAPass::processModelOp(ModelOp modelOp)
{
  mlir::IRRewriter rewriter(&getContext());
  mlir::SymbolTableCollection symbolTableCollection;

  llvm::SmallVector<SCCOp> initialSCCs;
  llvm::SmallVector<SCCOp> mainSCCs;

  modelOp.collectInitialSCCs(initialSCCs);
  modelOp.collectMainSCCs(mainSCCs);

  llvm::SmallVector<VariableOp> variables;
  modelOp.collectVariables(variables);

  // Solve the 'initial conditions' model.
  if (mlir::failed(solveICModel(
          rewriter, symbolTableCollection, modelOp, variables, initialSCCs))) {
    return mlir::failure();
  }

  // Solve the 'main' model.
  if (mlir::failed(solveMainModel(
          rewriter, symbolTableCollection, modelOp, variables, mainSCCs))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult IDAPass::solveICModel(
    mlir::IRRewriter& rewriter,
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variables,
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

  // The SCCs that are handled by IDA.
  llvm::DenseSet<SCCOp> externalSCCs;

  if (reducedSystem) {
    LLVM_DEBUG(llvm::dbgs() << "Reduced system feature enabled\n");

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

      auto explicitEquation = equation.cloneAndExplicitate(
          rewriter, symbolTableCollection);

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
      if (mlir::failed(addICModelSCC(
              symbolTableCollection, modelOp, *idaInstance, scc))) {
        return mlir::failure();
      }

      externalSCCs.insert(scc);
    }

    // Add the implicit equations to the set of equations managed by IDA,
    // together with their written variables.
    LLVM_DEBUG(llvm::dbgs() << "Add the implicit equations\n");

    for (SCCOp scc : implicitSCCs) {
      if (mlir::failed(addICModelSCC(
              symbolTableCollection, modelOp, *idaInstance, scc))) {
        return mlir::failure();
      }

      externalSCCs.insert(scc);
    }
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Reduced system feature disabled\n");

    // Add all the variables to IDA.
    for (VariableOp variable : variables) {
      LLVM_DEBUG(llvm::dbgs() << "Add algebraic variable: "
                              << variable.getSymName() << "\n");

      idaInstance->addAlgebraicVariable(variable);
    }

    // The equations will be automatically discovered later, while searching
    // for equations writing in variables managed by IDA.
  }

  if (mlir::failed(addEquationsWritingToIDAVariables(
          symbolTableCollection, modelOp, *idaInstance, SCCs, externalSCCs,
          [&](SCCOp scc) {
            return addICModelSCC(
                symbolTableCollection, modelOp, *idaInstance, scc);
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

  if (mlir::failed(idaInstance->declareInstance(
          rewriter, modelOp.getLoc(), moduleOp))) {
    return mlir::failure();
  }

  if (mlir::failed(createInitICSolversFunction(
          rewriter, moduleOp, symbolTableCollection, modelOp.getLoc(), modelOp,
          idaInstance.get(), variables, SCCs))) {
    return mlir::failure();
  }

  if (mlir::failed(createDeinitICSolversFunction(
          rewriter, moduleOp, modelOp.getLoc(), idaInstance.get()))) {
    return mlir::failure();
  }

  if (mlir::failed(createSolveICModelFunction(
          rewriter, moduleOp, symbolTableCollection, idaInstance.get(),
          modelOp, internalSCCs))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult IDAPass::addICModelSCC(
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    IDAInstance& idaInstance,
    SCCOp scc)
{
  for (MatchedEquationInstanceOp equation :
       scc.getOps<MatchedEquationInstanceOp>()) {
    if (mlir::failed(addICModelEquation(
            symbolTableCollection, modelOp, idaInstance, equation))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult IDAPass::addICModelEquation(
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    IDAInstance& idaInstance,
    MatchedEquationInstanceOp equationOp)
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
    llvm::ArrayRef<VariableOp> variables,
    llvm::ArrayRef<SCCOp> SCCs)
{
  LLVM_DEBUG(llvm::dbgs() << "Solving the 'main' model\n");
  auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();
  auto& derivativesMap = getDerivativesMap(modelOp);

  auto idaInstance = std::make_unique<IDAInstance>(
      "ida_main", symbolTableCollection, &derivativesMap,
      reducedSystem, reducedDerivatives, jacobianOneSweep, debugInformation);

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

        LLVM_DEBUG(llvm::dbgs() << "Add state variable: "
                                << variable.getSymName() << "\n");
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

      auto explicitEquation = equation.cloneAndExplicitate(
          rewriter, symbolTableCollection);

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
      if (mlir::failed(addMainModelSCC(
              symbolTableCollection, modelOp, derivativesMap, *idaInstance,
              scc))) {
        return mlir::failure();
      }
    }

    // Add the implicit equations to the set of equations managed by IDA,
    // together with their written variables.
    LLVM_DEBUG(llvm::dbgs() << "Add the implicit equations\n");

    for (SCCOp scc : implicitSCCs) {
      if (mlir::failed(addMainModelSCC(
              symbolTableCollection, modelOp, derivativesMap, *idaInstance,
              scc))) {
        return mlir::failure();
      }
    }
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Reduced system feature disabled\n");

    // Add all the variables to IDA.
    for (VariableOp variable : variables) {
      auto variableName =
          mlir::SymbolRefAttr::get(variable.getSymNameAttr());

      if (derivativesMap.getDerivative(variableName)) {
        LLVM_DEBUG(llvm::dbgs() << "Add state variable: "
                                << variable.getSymName() << "\n");
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
            return addMainModelSCC(
                symbolTableCollection, modelOp, derivativesMap, *idaInstance,
                scc);
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

  if (mlir::failed(idaInstance->declareInstance(
          rewriter, modelOp.getLoc(), moduleOp))) {
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

  if (mlir::failed(createCalcICFunction(
          rewriter, moduleOp, modelOp.getLoc(), idaInstance.get()))) {
    return mlir::failure();
  }

  if (mlir::failed(createUpdateIDAVariablesFunction(
          rewriter, moduleOp, modelOp.getLoc(), idaInstance.get()))) {
    return mlir::failure();
  }

  if (mlir::failed(createUpdateNonIDAVariablesFunction(
          rewriter, moduleOp, symbolTableCollection, idaInstance.get(),
          modelOp, internalSCCs))) {
    return mlir::failure();
  }

  if (mlir::failed(createGetIDATimeFunction(
          rewriter, moduleOp, modelOp.getLoc(), idaInstance.get()))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult IDAPass::addMainModelSCC(
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    const DerivativesMap& derivativesMap,
    IDAInstance& idaInstance,
    SCCOp scc)
{
  for (MatchedEquationInstanceOp equation :
       scc.getOps<MatchedEquationInstanceOp>()) {
    if (mlir::failed(addMainModelEquation(
            symbolTableCollection, modelOp, derivativesMap, idaInstance,
            equation))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult IDAPass::addMainModelEquation(
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    const DerivativesMap& derivativesMap,
    IDAInstance& idaInstance,
    MatchedEquationInstanceOp equationOp)
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

mlir::LogicalResult IDAPass::addEquationsWritingToIDAVariables(
    mlir::SymbolTableCollection& symbolTableCollection,
    ModelOp modelOp,
    IDAInstance& idaInstance,
    llvm::ArrayRef<SCCOp> SCCs,
    llvm::DenseSet<SCCOp>& externalSCCs,
    llvm::function_ref<mlir::LogicalResult(SCCOp)> addFn)
{
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
            symbolTableCollection.lookupSymbolIn<VariableOp>(
                modelOp, writtenVariable);

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

mlir::LogicalResult IDAPass::createInitICSolversFunction(
    mlir::IRRewriter& rewriter,
    mlir::ModuleOp moduleOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::Location loc,
    ModelOp modelOp,
    IDAInstance* idaInstance,
    llvm::ArrayRef<VariableOp> variables,
    llvm::ArrayRef<SCCOp> allSCCs) const
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
          rewriter, loc, moduleOp, modelOp, variables, allSCCs))) {
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
    llvm::ArrayRef<SCCOp> allSCCs) const
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
          rewriter, loc, moduleOp, modelOp, variableOps, allSCCs))) {
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
    mlir::RewriterBase& rewriter,
    mlir::ModuleOp moduleOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    IDAInstance* idaInstance,
    ModelOp modelOp,
    llvm::ArrayRef<SCCOp> internalSCCs)
{
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  // Create the function.
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = rewriter.create<mlir::simulation::FunctionOp>(
      modelOp.getLoc(), "solveICModel",
      rewriter.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  // Compute the initial conditions for the equations managed by IDA.
  if (mlir::failed(idaInstance->performCalcIC(rewriter, modelOp.getLoc()))) {
    return mlir::failure();
  }

  if (!internalSCCs.empty()) {
    // Create the schedule operation.
    ScheduleOp scheduleOp = createSchedule(
        rewriter, symbolTableCollection, moduleOp, modelOp.getLoc(),
        "ic_equations", internalSCCs);

    if (!scheduleOp) {
      return mlir::failure();
    }

    rewriter.create<RunScheduleOp>(
        modelOp.getLoc(),
        mlir::SymbolRefAttr::get(scheduleOp.getSymNameAttr()));
  }

  rewriter.create<mlir::simulation::ReturnOp>(modelOp.getLoc(), std::nullopt);
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
    mlir::RewriterBase& rewriter,
    mlir::ModuleOp moduleOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    IDAInstance* idaInstance,
    ModelOp modelOp,
    llvm::ArrayRef<SCCOp> internalSCCs)
{
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  // Create the function.
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = rewriter.create<mlir::simulation::FunctionOp>(
      modelOp.getLoc(), "updateNonIDAVariables",
      rewriter.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block* entryBlock = functionOp.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  if (!internalSCCs.empty()) {
    // Create the schedule operation.
    ScheduleOp scheduleOp = createSchedule(
        rewriter, symbolTableCollection, moduleOp, modelOp.getLoc(),
        "main_equations", internalSCCs);

    if (!scheduleOp) {
      return mlir::failure();
    }

    rewriter.create<RunScheduleOp>(
        modelOp.getLoc(), mlir::SymbolRefAttr::get(scheduleOp.getSymNameAttr()));
  }

  rewriter.create<mlir::simulation::ReturnOp>(modelOp.getLoc(), std::nullopt);
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
