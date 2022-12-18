#include "marco/Codegen/Transforms/VariablesPromotion.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Codegen/Transforms/ModelSolving/Utils.h"
#include "marco/Modeling/Dependency.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_VARIABLESPROMOTIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace
{
  class VariablesPromotionPass
      : public mlir::modelica::impl::VariablesPromotionPassBase<
          VariablesPromotionPass>
  {
    public:
      using VariablesPromotionPassBase::VariablesPromotionPassBase;

      void runOnOperation() override
      {
        mlir::OpBuilder builder(getOperation());
        llvm::SmallVector<ModelOp, 1> modelOps;

        getOperation()->walk([&](ModelOp modelOp) {
          if (modelOp.getSymName() == modelName) {
            modelOps.push_back(modelOp);
          }
        });

        for (ModelOp modelOp : modelOps) {
          if (mlir::failed(processModelOp(builder, modelOp))) {
            return signalPassFailure();
          }
        }
      }

    private:
      mlir::LogicalResult processModelOp(
        mlir::OpBuilder& builder, ModelOp modelOp);

      mlir::LogicalResult getInitialConditionsModel(
          Model<MatchedEquation>& model) const;

      mlir::LogicalResult getMainModel(
          Model<MatchedEquation>& model) const;

      mlir::LogicalResult getModel(
          Model<MatchedEquation>& model,
          std::function<bool(EquationInterface)> equationsFilter) const;

      /// Get the writes map of a model, that is the knowledge of which
      /// equation writes into a variable and in which indices.
      // The variables are mapped by their argument number.
      std::multimap<unsigned int, std::pair<IndexSet, MatchedEquation*>>
      getWritesMap(const Model<MatchedEquation>& model) const;
  };
}

mlir::LogicalResult VariablesPromotionPass::processModelOp(
    mlir::OpBuilder& builder, ModelOp modelOp)
{
  // Retrieve the derivatives map.
  DerivativesMap derivativesMap;

  if (mlir::failed(readDerivativesMap(modelOp, derivativesMap))) {
    return mlir::failure();
  }

  // Obtain the 'initial conditions' matched model.
  Model<MatchedEquation> initialConditionsModel(modelOp);
  initialConditionsModel.setDerivativesMap(derivativesMap);

  if (mlir::failed(getInitialConditionsModel(initialConditionsModel))) {
    return mlir::failure();
  }

  // Determine the writes map of the 'initial conditions' model. This must be
  // used to avoid having different initial equations writing into the same
  // scalar variables.
  auto initialConditionsModelWritesMap = getWritesMap(initialConditionsModel);

  // Obtain the 'main' matched model.
  Model<MatchedEquation> mainModel(modelOp);
  mainModel.setDerivativesMap(derivativesMap);

  if (mlir::failed(getMainModel(mainModel))) {
    return mlir::failure();
  }

  auto mainModelWritesMap = getWritesMap(mainModel);

  // The indices of the variables.
  llvm::DenseMap<unsigned int, IndexSet> variablesIndices;

  for (const auto& variable : mainModel.getVariables()) {
    unsigned int argNumber =
        variable->getValue().cast<mlir::BlockArgument>().getArgNumber();

    variablesIndices[argNumber] += variable->getIndices();
  }

  // The variables that are already marked as parameters.
  llvm::DenseSet<unsigned int> parameters;

  // The variables that can be promoted to parameters.
  llvm::DenseMap<unsigned int, IndexSet> promotableVariablesIndices;

  // Collect the instructions used to create the variables, which are later
  // modified if the variables are promoted.
  llvm::DenseMap<unsigned int, MemberCreateOp> memberCreateOps;

  for (const auto& variable : mainModel.getVariables()) {
    unsigned int argNumber =
        variable->getValue().cast<mlir::BlockArgument>().getArgNumber();

    if (variable->isConstant()) {
      parameters.insert(argNumber);
    }

    memberCreateOps[argNumber] = variable->getDefiningOp();
  }

  // Determine the promotable equations by creating the dependency graph and
  // doing a post-order visit.

  using VectorDependencyGraph =
      ArrayVariablesDependencyGraph<Variable*, MatchedEquation*>;

  using SCC = VectorDependencyGraph::SCC;

  VectorDependencyGraph vectorDependencyGraph;
  llvm::SmallVector<MatchedEquation*> equations;

  for (auto& equation : mainModel.getEquations()) {
    equations.push_back(equation.get());
  }

  vectorDependencyGraph.addEquations(equations);

  SCCDependencyGraph<SCC> sccDependencyGraph;
  sccDependencyGraph.addSCCs(vectorDependencyGraph.getSCCs());

  auto scheduledSCCs = sccDependencyGraph.postOrder();

  llvm::DenseSet<MatchedEquation*> promotableEquations;

  for (const auto& sccDescriptor : scheduledSCCs) {
    const SCC& scc = sccDependencyGraph[sccDescriptor];

    // Collect the equations of the SCC for a faster lookup.
    llvm::DenseSet<MatchedEquation*> sccEquations;

    for (const auto& equationDescriptor : scc) {
      MatchedEquation* equation =
          scc.getGraph()[equationDescriptor].getProperty();

      sccEquations.insert(equation);
    }

    // Check if the current SCC depends only on parametric variables or
    // variables that are written by the equations of the SCC.

    bool promotable = true;

    for (const auto& equationDescriptor : scc) {
      MatchedEquation* equation =
          scc.getGraph()[equationDescriptor].getProperty();

      auto accesses = equation->getAccesses();

      // Check if the equation uses the 'time' variable. If it does, then it
      // must not be promoted to an initial equation.
      bool timeUsage = false;

      equation->getOperation().walk([&](TimeOp timeOp) {
        timeUsage = true;
      });

      if (timeUsage) {
        promotable = false;
        break;
      }

      // Do not promote the equation if it writes to a derivative variable.
      auto writtenVariable = equation->getWrite().getVariable()->getValue();

      unsigned int writtenArgNumber =
          writtenVariable.cast<mlir::BlockArgument>().getArgNumber();

      if (mainModel.getDerivativesMap().isDerivative(writtenArgNumber)) {
        promotable = false;
        break;
      }

      // Check the accesses to the variables.
      promotable &= llvm::all_of(accesses, [&](const Access& access) {
        mlir::Value readVariable = access.getVariable()->getValue();

        unsigned int argNumber =
            readVariable.cast<mlir::BlockArgument>().getArgNumber();

        if (parameters.contains(argNumber)) {
          // If the read variable is a parameter, then there is no need for
          // additional analyses.
          return true;
        }

        auto readIndices = access.getAccessFunction().map(
            equation->getIterationRanges());

        auto writingEquations =
            llvm::make_range(mainModelWritesMap.equal_range(argNumber));

        if (writingEquations.empty()) {
          // If there is no equation writing to the variable, then the variable
          // may be a state.
          return false;
        }

        return llvm::all_of(writingEquations, [&](const auto& entry) {
          MatchedEquation* writingEquation = entry.second.second;
          const IndexSet& writtenIndices = entry.second.first;

          if (promotableEquations.contains(writingEquation)) {
            // The writing equation (and the scalar variables it writes to) has
            // already been marked as promotable.
            return true;
          }

          if (sccEquations.contains(writingEquation)) {
            // If the writing equation belongs to the current SCC, then the
            // whole SCC may still be turned into initial equations.
            return true;
          }

          if (!writtenIndices.overlaps(readIndices)) {
            // Ignore the equation (consider it valid) if its written indices
            // don't overlap the read ones.
            return true;
          }

          return false;
        });
      });
    }

    if (promotable) {
      // Promote all the equations of the SCC.

      for (const auto& equationDescriptor : scc) {
        MatchedEquation* equation =
            scc.getGraph()[equationDescriptor].getProperty();

        promotableEquations.insert(equation);

        const auto& writeAccess = equation->getWrite();
        mlir::Value writtenVariable = writeAccess.getVariable()->getValue();

        IndexSet writtenIndices = writeAccess.getAccessFunction().map(
                equation->getIterationRanges());

        unsigned int argNumber =
            writtenVariable.cast<mlir::BlockArgument>().getArgNumber();

        promotableVariablesIndices[argNumber] += std::move(writtenIndices);
      }
    }
  }

  // Determine the promotable variables.
  // A variable can be promoted only if all the equations writing it (and thus
  // all the scalar variables) are promotable.

  llvm::DenseSet<unsigned int> promotableVariables;

  for (const auto& variable : mainModel.getVariables()) {
    unsigned int argNumber =
        variable->getValue().cast<mlir::BlockArgument>().getArgNumber();

    if (!promotableVariables.contains(argNumber) &&
        variablesIndices[argNumber] == promotableVariablesIndices[argNumber]) {
      promotableVariables.insert(argNumber);
    }
  }

  // Promote the variables (and the equations writing to them).
  for (unsigned int variable : promotableVariables) {
    // Change the member type.
    MemberCreateOp memberCreateOp = memberCreateOps[variable];
    auto newMemberType = memberCreateOp.getMemberType().asConstant();
    memberCreateOp.getResult().setType(newMemberType);

    // Determine the indices of the variable that are currently handled only by
    // equations that are not initial equations.
    IndexSet variableIndices;

    // Initially, consider all the variable indices.
    for (const auto& entry : llvm::make_range(
             mainModelWritesMap.equal_range(variable))) {
      variableIndices += entry.second.first;
    }

    // Then, remove the indices that are written by already existing initial
    // equations.
    for (const auto& entry : llvm::make_range(
             initialConditionsModelWritesMap.equal_range(variable))) {
      variableIndices -= entry.second.first;
    }

    variableIndices = variableIndices.getCanonicalRepresentation();

    if (variableIndices.empty()) {
      // Skip the variable if all of its indices are already handled by the
      // initial equations.
      continue;
    }

    // Convert the writing non-initial equations into initial equations.
    auto writingEquations =
        llvm::make_range(mainModelWritesMap.equal_range(variable));

    for (const auto& entry : writingEquations) {
      MatchedEquation* equation = entry.second.second;

      std::vector<mlir::Value> inductionVariables =
          equation->getInductionVariables();

      EquationInterface equationInt = equation->getOperation();
      IndexSet writingEquationIndices = equation->getIterationRanges();
      const Access& writeAccess = equation->getWrite();

      IndexSet writtenIndices =
          writeAccess.getAccessFunction().map(writingEquationIndices);

      // Restrict the indices to the ones not handled by the initial
      // equations.
      writtenIndices = writtenIndices.intersect(variableIndices);

      if (!writtenIndices.empty()) {
        // Get the indices of the equation that actually writes the scalar
        // variables of interest.
        writingEquationIndices =
            writeAccess.getAccessFunction().inverseMap(
                writtenIndices, writingEquationIndices);

        writingEquationIndices =
            writingEquationIndices.getCanonicalRepresentation();

        // Create the initial equations.
        for (const auto& range :
             llvm::make_range(writingEquationIndices.rangesBegin(),
                              writingEquationIndices.rangesEnd())) {
          mlir::BlockAndValueMapping mapping;
          llvm::SmallVector<mlir::Value> newInductionVariables;

          builder.setInsertionPointToEnd(&modelOp.getBodyRegion().front());

          if (newMemberType.hasRank()) {
            for (int64_t i = 0, e = newMemberType.getRank(); i < e; ++i) {
              auto forEquationOp = builder.create<ForEquationOp>(
                  equationInt.getLoc(),
                  range[i].getBegin(),
                  range[i].getEnd() - 1,
                  1);

              mapping.map(inductionVariables[i], forEquationOp.induction());
              builder.setInsertionPointToStart(forEquationOp.bodyBlock());
            }
          }

          auto initialEquationOp =
              builder.create<InitialEquationOp>(equationInt.getLoc());

          initialEquationOp->setAttrs(equationInt->getAttrs());
          assert(initialEquationOp.getBodyRegion().empty());

          mlir::Block* body =
              builder.createBlock(&initialEquationOp.getBodyRegion());

          builder.setInsertionPointToStart(body);

          // Clone the equation body.
          for (auto& op : equationInt.getBodyRegion().getOps()) {
            builder.clone(op, mapping);
          }
        }
      }
    }
  }

  // Delete the equations that have been promoted to initial equations.
  llvm::DenseSet<mlir::Operation*> erasedEquations;

  for (unsigned int variable : promotableVariables) {
    auto writingEquations = llvm::make_range(
        mainModelWritesMap.equal_range(variable));

    for (const auto& entry : writingEquations) {
      MatchedEquation* equation = entry.second.second;
      EquationInterface equationInt = equation->getOperation();

      if (!erasedEquations.contains(equationInt.getOperation())) {
        equation->eraseIR();
        erasedEquations.insert(equationInt.getOperation());
      }
    }
  }

  return mlir::success();
}

mlir::LogicalResult VariablesPromotionPass::getInitialConditionsModel(
    Model<MatchedEquation>& model) const
{
  auto equationsFilter = [](EquationInterface op) {
    return mlir::isa<InitialEquationOp>(op);
  };

  return getModel(model, equationsFilter);
}

mlir::LogicalResult VariablesPromotionPass::getMainModel(
    Model<MatchedEquation>& model) const
{
  auto equationsFilter = [](EquationInterface op) {
    return mlir::isa<EquationOp>(op);
  };

  return getModel(model, equationsFilter);
}

mlir::LogicalResult VariablesPromotionPass::getModel(
    Model<MatchedEquation>& model,
    std::function<bool(EquationInterface)> equationsFilter) const
{
  model.setVariables(discoverVariables(model.getOperation()));
  return readMatchingAttributes(model, equationsFilter);
}

std::multimap<unsigned int, std::pair<IndexSet, MatchedEquation*>>
VariablesPromotionPass::getWritesMap(const Model<MatchedEquation>& model) const
{
  std::multimap<unsigned int, std::pair<IndexSet, MatchedEquation*>> writesMap;

  for (const auto& equation : model.getEquations()) {
    const Access& write = equation->getWrite();
    mlir::Value variable = write.getVariable()->getValue();

    unsigned int argNumber =
        variable.cast<mlir::BlockArgument>().getArgNumber();

    IndexSet writtenIndices = write.getAccessFunction().map(
        equation->getIterationRanges());

    writesMap.emplace(
        argNumber,
        std::make_pair(writtenIndices, equation.get()));
  }

  return writesMap;
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createVariablesPromotionPass()
  {
    return std::make_unique<VariablesPromotionPass>();
  }

  std::unique_ptr<mlir::Pass> createVariablesPromotionPass(
      const VariablesPromotionPassOptions& options)
  {
    return std::make_unique<VariablesPromotionPass>(options);
  }
}
