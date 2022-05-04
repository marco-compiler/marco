#include "marco/Codegen/Transforms/ModelSolving/ModelSolving.h"
#include "marco/Codegen/Transforms/ModelSolving/Cycles.h"
#include "marco/Codegen/Transforms/ModelSolving/Equation.h"
#include "marco/Codegen/Transforms/ModelSolving/EquationImpl.h"
#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include "marco/Codegen/Transforms/ModelSolving/ModelConverter.h"
#include "marco/Codegen/Transforms/ModelSolving/Scheduling.h"
#include "marco/Codegen/Transforms/ModelSolving/TypeConverter.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation/Common.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/VariableFilter/VariableFilter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include <cassert>
#include <map>
#include <memory>
#include <queue>
#include <set>

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace
{
  struct EquationOpMultipleValuesPattern : public mlir::OpRewritePattern<EquationOp>
  {
    using mlir::OpRewritePattern<EquationOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(EquationOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto terminator = mlir::cast<EquationSidesOp>(op.bodyBlock()->getTerminator());

      if (terminator.lhsValues().size() != terminator.rhsValues().size()) {
        return rewriter.notifyMatchFailure(op, "Different amount of values in left-hand and right-hand sides of the equation");
      }

      auto amountOfValues = terminator.lhsValues().size();

      for (size_t i = 0; i < amountOfValues; ++i) {
        rewriter.setInsertionPointAfter(op);

        auto clone = rewriter.create<EquationOp>(loc);
        assert(clone.bodyRegion().empty());
        mlir::Block* cloneBodyBlock = rewriter.createBlock(&clone.bodyRegion());
        rewriter.setInsertionPointToStart(cloneBodyBlock);

        mlir::BlockAndValueMapping mapping;

        for (auto& originalOp : op.bodyBlock()->getOperations()) {
          if (mlir::isa<EquationSideOp>(originalOp)) {
            continue;
          }

          if (mlir::isa<EquationSidesOp>(originalOp)) {
            auto lhsOp = mlir::cast<EquationSideOp>(terminator.lhs().getDefiningOp());
            auto rhsOp = mlir::cast<EquationSideOp>(terminator.rhs().getDefiningOp());

            auto newLhsOp = rewriter.create<EquationSideOp>(lhsOp.getLoc(), mapping.lookup(terminator.lhsValues()[i]));
            auto newRhsOp = rewriter.create<EquationSideOp>(rhsOp.getLoc(), mapping.lookup(terminator.rhsValues()[i]));

            rewriter.create<EquationSidesOp>(terminator.getLoc(), newLhsOp, newRhsOp);
          } else {
            rewriter.clone(originalOp, mapping);
          }
        }
      }

      rewriter.eraseOp(op);
      return mlir::success();
    }
  };
}

static void collectDerivedVariablesIndices(DerivativesMap& derivativesMap, const Equations<Equation>& equations)
{
  for (const auto& equation : equations) {
    auto accesses = equation->getAccesses();

    equation->getOperation().walk([&](DerOp derOp) {
      auto it = llvm::find_if(accesses, [&](const auto& access) {
        auto value = equation->getValueAtPath(access.getPath());
        return value == derOp.operand();
      });

      assert(it != accesses.end());
      const auto& access = *it;
      auto derivedIndices = access.getAccessFunction().map(equation->getIterationRanges());
      auto argNumber = access.getVariable()->getValue().cast<mlir::BlockArgument>().getArgNumber();
      derivativesMap.addDerivedIndices(argNumber, derivedIndices);
    });
  }
}

static mlir::LogicalResult createDerivativeVariables(mlir::OpBuilder& builder, ModelOp modelOp, DerivativesMap& derivativesMap)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  auto derivativeOrder = [&](unsigned int variable) -> unsigned int {
    return 0;
  };

  // The list of the new variables
  std::vector<mlir::Value> variables;
  auto terminator = mlir::cast<YieldOp>(modelOp.initRegion().back().getTerminator());

  for (auto variable : terminator.values()) {
    variables.push_back(variable);
  }

  // Create the new variables for the derivatives
  builder.setInsertionPoint(terminator);
  size_t derArgumentPosition = terminator.values().size();

  for (auto variable : llvm::enumerate(terminator.values())) {
    auto derivedIndices = derivativesMap.getDerivedIndices(variable.index());

    if (derivedIndices.empty()) {
      continue;
    }

    auto memberCreateOp = variable.value().getDefiningOp<MemberCreateOp>();
    auto variableMemberType = memberCreateOp.getMemberType();
    auto currentOrder = derivativeOrder(variable.index());

    for (const auto& range : llvm::enumerate(derivedIndices)) {
      // Determine the name of the derivative variable.
      // We need to append a counter in order to avoid name clashes.
      auto derivativeName = getNextFullDerVariableName(memberCreateOp.name(), currentOrder + 1) + "_range" + std::to_string(range.index());

      // Determine the dimensions of the new variable.
      // While doing so, we must take into account the possibility for the variable to be a scalar
      // (and thus having rank 0, despite the rank 1 reported by the modeling library).
      std::vector<long> derivativeDimensions;
      std::vector<Range> derivativeIndices(range.value().rank(), Range(0, 1));

      for (size_t i = 0, e = std::min(range.value().rank(), variableMemberType.getRank()); i < e; ++i) {
        auto dimension = range.value()[i].size();
        derivativeDimensions.push_back(dimension);
        derivativeIndices[i] = Range(0, dimension);
      }

      auto arrayType = ArrayType::get(builder.getContext(), RealType::get(builder.getContext()), derivativeDimensions);
      assert(arrayType.hasConstantShape());

      // Create the variable and initialize it at zero
      auto derMemberOp = builder.create<MemberCreateOp>(
          memberCreateOp.getLoc(), derivativeName, MemberType::wrap(arrayType), llvm::None);

      variables.push_back(derMemberOp);

      mlir::Value zero = builder.create<ConstantOp>(derMemberOp.getLoc(), RealAttr::get(builder.getContext(), 0));
      mlir::Value derivative = builder.create<MemberLoadOp>(derMemberOp.getLoc(), derMemberOp);
      builder.create<ArrayFillOp>(derMemberOp.getLoc(), derivative, zero);

      // Add the variable to the signature of the existing regions
      [[maybe_unused]] auto equationsRegionArg = modelOp.equationsRegion().addArgument(arrayType);
      [[maybe_unused]] auto initialEquationsRegionArg = modelOp.initialEquationsRegion().addArgument(arrayType);
      assert(derArgumentPosition == equationsRegionArg.getArgNumber());
      assert(derArgumentPosition == initialEquationsRegionArg.getArgNumber());

      derivativesMap.setDerivative(variable.index(), range.value(), derArgumentPosition, MultidimensionalRange(derivativeIndices));

      ++derArgumentPosition;
    }
  }

  builder.create<YieldOp>(terminator.getLoc(), variables);
  terminator.erase();

  return mlir::success();
}

static mlir::modelica::EquationOp cloneEquationWithNewIndices(
    mlir::OpBuilder& builder, const Equation& equation, const MultidimensionalRange& indices, mlir::BlockAndValueMapping& mapping)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto equationOp = equation.getOperation();
  builder.setInsertionPointAfter(equationOp.getOperation());
  ForEquationOp parent = equationOp->getParentOfType<ForEquationOp>();

  while (parent != nullptr) {
    builder.setInsertionPointAfter(parent.getOperation());
    parent = parent->getParentOfType<ForEquationOp>();
  }

  auto oldIterationVariables = equation.getInductionVariables();
  assert(indices.rank() >= oldIterationVariables.size());

  for (size_t i = 0; i < oldIterationVariables.size(); ++i) {
    auto loop = builder.create<ForEquationOp>(oldIterationVariables[i].getLoc(), indices[i].getBegin(), indices[i].getEnd() - 1);
    builder.setInsertionPointToStart(loop.bodyBlock());
    mapping.map(oldIterationVariables[i], loop.induction());
  }

  return mlir::cast<EquationOp>(builder.clone(*equationOp.getOperation(), mapping));
}

static mlir::LogicalResult removeDerOps(mlir::OpBuilder& builder, Model<Equation>& model)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto& derivativesMap = model.getDerivativesMap();

  std::vector<mlir::Value> variables;
  variables.resize(model.getVariables().size());

  for (const auto& variable : model.getVariables()) {
    variables[variable->getValue().cast<mlir::BlockArgument>().getArgNumber()] = variable->getValue();
  }

  for (auto& originalEquation : model.getEquations()) {
    // The equations to be processed
    std::queue<std::unique_ptr<Equation>> equations;

    // Add the original equation to the processing queue
    equations.push(Equation::build(originalEquation->cloneIR(), originalEquation->getVariables()));
    originalEquation->eraseIR();

    // Keep iterating until no DerOp exists
    while (!equations.empty()) {
      auto& equation = equations.front();

      std::vector<DerOp> derOps;

      equation->getOperation().walk([&](DerOp derOp) {
        derOps.push_back(derOp);
      });

      if (!derOps.empty()) {
        auto equationIndices = IndexSet(equation->getIterationRanges());
        auto inductionVariables = equation->getInductionVariables();
        auto accesses = equation->getAccesses();

        // We need to process the DerOps one by one, because multiple DerOps in the same
        // equations would otherwise wrongly create duplicates of the equation itself.
        auto& derOp = derOps[0];

        auto access = llvm::find_if(accesses, [&](const auto& access) {
          auto value = equation->getValueAtPath(access.getPath());
          return value == derOp.operand();
        });

        assert(access != accesses.end());
        auto variable = access->getVariable()->getValue();
        auto varArgNumber = variable.cast<mlir::BlockArgument>().getArgNumber();
        auto requestedDerIndices = access->getAccessFunction().map(equationIndices);

        for (const auto& derivative : derivativesMap.getDerivative(varArgNumber)) {
          const auto& availableDerIndices = *derivative.first;

          if (!requestedDerIndices.overlaps(availableDerIndices)) {
            continue;
          }

          auto coveredDerIndices = requestedDerIndices.intersect(availableDerIndices);
          auto newEquationIndices = access->getAccessFunction().inverseMap(coveredDerIndices, equationIndices);

          for (const auto& newRange : newEquationIndices) {
            mlir::BlockAndValueMapping mapping;
            auto cloneOp = cloneEquationWithNewIndices(builder, *equation, newRange, mapping);
            auto clone = Equation::build(cloneOp, equation->getVariables());
            auto newIterationVariables = clone->getInductionVariables();

            auto mappedDerOp = mapping.lookup(derOp.getResult()).getDefiningOp<DerOp>();
            builder.setInsertionPoint(mappedDerOp);

            mlir::Value replacement = variables[derivative.second->getArgNumber()];
            size_t rank = replacement.getType().cast<ArrayType>().getRank();
            assert(rank <= access->getAccessFunction().size());
            std::vector<mlir::Value> indices;

            for (size_t i = 0; i < rank; ++i) {
              const auto& dimensionAccess = access->getAccessFunction()[i];
              auto derivativeOffset = derivative.second->getOffsets()[i];

              if (dimensionAccess.isConstantAccess()) {
                mlir::Value index = builder.create<ConstantOp>(replacement.getLoc(), builder.getIndexAttr(dimensionAccess.getPosition() - derivativeOffset));
                indices.push_back(index);
              } else {
                mlir::Value inductionVar = mapping.lookup(inductionVariables[dimensionAccess.getInductionVariableIndex()]);
                mlir::Value offset = builder.create<ConstantOp>(replacement.getLoc(), builder.getIndexAttr(dimensionAccess.getOffset() - derivativeOffset));
                mlir::Value index = builder.create<AddOp>(replacement.getLoc(), builder.getIndexType(), inductionVar, offset);
                indices.push_back(index);
              }
            }

            replacement = builder.create<LoadOp>(replacement.getLoc(), replacement, indices);
            mappedDerOp.replaceAllUsesWith(replacement);
            mappedDerOp.erase();

            equations.push(std::move(clone));
          }
        }

        equation->eraseIR();
      }

      equations.pop();
    }
  }

  return mlir::success();
}

namespace
{
  /// Model solving pass.
  /// Its objective is to convert a descriptive (and thus not sequential) model into an
  /// algorithmic one and to create the functions to be called during the simulation.
  class ModelSolvingPass: public mlir::PassWrapper<ModelSolvingPass, mlir::OperationPass<mlir::ModuleOp>>
  {
    public:
      explicit ModelSolvingPass(ModelSolvingOptions options, unsigned int bitWidth)
          : options(std::move(options)),
            bitWidth(std::move(bitWidth))
      {
      }

      void getDependentDialects(mlir::DialectRegistry& registry) const override
      {
        registry.insert<ModelicaDialect>();
        registry.insert<mlir::ida::IDADialect>();
        registry.insert<mlir::scf::SCFDialect>();
        registry.insert<mlir::LLVM::LLVMDialect>();
      }

      void runOnOperation() override
      {
        llvm::SmallVector<ModelOp, 1> models;

        getOperation().walk([&](ModelOp op) {
          models.push_back(op);
        });

        if (models.size() > 1) {
          // There must be at most one ModelOp inside the module
          return signalPassFailure();
        }

        mlir::OpBuilder builder(models[0]);

        // Copy the equations into the initial equations' region, in order to use
        // them when computing the initial values of the variables.

        if (mlir::failed(copyEquationsAmongInitialEquations(builder, models[0]))) {
          return signalPassFailure();
        }

        if (mlir::failed(convertEquationsWithMultipleValues())) {
          return signalPassFailure();
        }

        // Split the loops containing more than one operation within their bodies
        if (mlir::failed(convertToSingleEquationBody(models[0]))) {
          return signalPassFailure();
        }

        // The initial conditions are determined by resolving a separate model, with
        // indeed more equations than the model used during the simulation loop.
        Model<Equation> initialModel(models[0]);
        Model<Equation> model(models[0]);

        initialModel.setVariables(discoverVariables(initialModel.getOperation().initialEquationsRegion()));
        initialModel.setEquations(discoverEquations(initialModel.getOperation().initialEquationsRegion(), initialModel.getVariables()));

        model.setVariables(discoverVariables(model.getOperation().equationsRegion()));
        model.setEquations(discoverEquations(model.getOperation().equationsRegion(), model.getVariables()));

        // Determine which scalar variables do appear as argument to the derivative operation
        DerivativesMap derivativesMap;

        collectDerivedVariablesIndices(derivativesMap, initialModel.getEquations());
        collectDerivedVariablesIndices(derivativesMap, model.getEquations());

        // Create the variables for the derivatives
        if (mlir::failed(createDerivativeVariables(builder, models[0], derivativesMap))) {
          return signalPassFailure();
        }

        // The derivatives mapping is now complete, thus we can set the derivatives map inside the models
        initialModel.setDerivativesMap(derivativesMap);
        model.setDerivativesMap(derivativesMap);

        // Now that the derivatives have been converted to variables, we need perform a new scan
        // of the variables so that they become available inside the model.
        initialModel.setVariables(discoverVariables(initialModel.getOperation().initialEquationsRegion()));
        model.setVariables(discoverVariables(model.getOperation().equationsRegion()));

        // Remove the derivative operations
        if (mlir::failed(removeDerOps(builder, initialModel))) {
          return signalPassFailure();
        }

        if (mlir::failed(removeDerOps(builder, model))) {
          return signalPassFailure();
        }

        // Erasing the derivative operations may have caused equations splitting.
        // For this reason we need to perform again the discovery process.

        initialModel.setEquations(discoverEquations(initialModel.getOperation().initialEquationsRegion(), initialModel.getVariables()));
        model.setEquations(discoverEquations(model.getOperation().equationsRegion(), model.getVariables()));

        // Convert both the models to scheduled ones
        Model<ScheduledEquationsBlock> scheduledInitialModel(initialModel.getOperation());
        Model<ScheduledEquationsBlock> scheduledModel(model.getOperation());

        /*
        auto initialModelVariableMatchableFn = [](const Variable& variable) -> bool {
          return !variable.isConstant();
        };

        if (mlir::failed(convertToScheduledModel(builder, scheduledInitialModel, initialModel, initialModelVariableMatchableFn))) {
          scheduledInitialModel.getOperation().emitError("Can't solve the initialization problem");
          return signalPassFailure();
        }
         */

        auto modelVariableMatchableFn = [&](const Variable& variable) -> bool {
          mlir::Value var = variable.getValue();
          auto argNumber = var.cast<mlir::BlockArgument>().getArgNumber();

          return !model.getDerivativesMap().hasDerivative(argNumber) && !variable.isConstant();
        };

        if (mlir::failed(convertToScheduledModel(builder, scheduledModel, model, modelVariableMatchableFn))) {
          scheduledInitialModel.getOperation().emitError("Can't solve the model");
          return signalPassFailure();
        }

        // Create the simulation functions
        mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
        marco::codegen::TypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);
        ModelConverter modelConverter(options, typeConverter);

        if (auto status = modelConverter.convert(builder, scheduledModel); mlir::failed(status)) {
          return signalPassFailure();
        }

        // Erase the model operation, which has been converted to algorithmic code
        models[0].erase();
      }

    private:
      /// Copy the equations declared into the 'equations' region into the 'initial equations' region.
      mlir::LogicalResult copyEquationsAmongInitialEquations(mlir::OpBuilder& builder, ModelOp modelOp)
      {
        if (!modelOp.hasEquationsBlock()) {
          // There is no equation to be copied
          return mlir::success();
        }

        mlir::OpBuilder::InsertionGuard guard(builder);

        if (!modelOp.hasInitialEquationsBlock()) {
          builder.createBlock(&modelOp.initialEquationsRegion(), {}, modelOp.equationsRegion().getArgumentTypes());
        }

        // Map the variables declared into the equations region to the ones declared into the initial equations' region
        mlir::BlockAndValueMapping mapping;
        auto originalVariables = modelOp.equationsRegion().getArguments();
        auto mappedVariables = modelOp.initialEquationsRegion().getArguments();
        assert(originalVariables.size() == mappedVariables.size());

        for (const auto& [original, mapped] : llvm::zip(originalVariables, mappedVariables)) {
          mapping.map(original, mapped);
        }

        // Clone the equations
        builder.setInsertionPointToEnd(modelOp.initialEquationsBlock());

        for (auto& op : modelOp.equationsBlock()->getOperations()) {
          builder.clone(op, mapping);
        }

        return mlir::success();
      }

      mlir::LogicalResult convertEquationsWithMultipleValues()
      {
        mlir::ConversionTarget target(getContext());

        target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
          return true;
        });

        target.addDynamicallyLegalOp<EquationOp>([](EquationOp op) {
          auto terminator = mlir::cast<EquationSidesOp>(op.bodyBlock()->getTerminator());
          return terminator.lhsValues().size() == 1 && terminator.rhsValues().size() == 1;
        });

        mlir::OwningRewritePatternList patterns(&getContext());
        patterns.insert<EquationOpMultipleValuesPattern>(&getContext());

        return applyPartialConversion(getOperation(), target, std::move(patterns));
      }

      mlir::LogicalResult convertToSingleEquationBody(ModelOp modelOp)
      {
        llvm::SmallVector<EquationOp> equations;

        for (auto op : modelOp.equationsBlock()->getOps<EquationOp>()) {
          equations.push_back(op);
        }

        mlir::OpBuilder builder(modelOp);

        mlir::BlockAndValueMapping mapping;

        for (auto& equationOp : equations) {
          builder.setInsertionPointToEnd(modelOp.equationsBlock());
          std::vector<ForEquationOp> parents;

          ForEquationOp parent = equationOp->getParentOfType<ForEquationOp>();

          while (parent != nullptr) {
            parents.push_back(parent);
            parent = parent->getParentOfType<ForEquationOp>();
          }

          for (size_t i = 0, e = parents.size(); i < e; ++i) {
            auto clonedParent = mlir::cast<ForEquationOp>(builder.clone(*parents[e - i - 1].getOperation(), mapping));
            builder.setInsertionPointToEnd(clonedParent.bodyBlock());
          }

          builder.clone(*equationOp.getOperation(), mapping);
        }

        for (auto& equationOp : equations) {
          ForEquationOp parent = equationOp->getParentOfType<ForEquationOp>();
          equationOp.erase();

          while (parent != nullptr && parent.bodyBlock()->empty()) {
            ForEquationOp newParent = parent->getParentOfType<ForEquationOp>();
            parent.erase();
            parent = newParent;
          }
        }

        return mlir::success();
      }

      mlir::LogicalResult splitEquations(mlir::OpBuilder& builder, Model<MatchedEquation>& model)
      {
        Equations<MatchedEquation> equations;

        for (const auto& equation : model.getEquations()) {
          auto write = equation->getWrite();
          auto iterationRanges = equation->getIterationRanges();
          auto writtenIndices = write.getAccessFunction().map(iterationRanges);

          IndexSet result;

          for (const auto& access : equation->getAccesses()) {
            if (access.getPath() == write.getPath()) {
              continue;
            }

            if (access.getVariable() != write.getVariable()) {
              continue;
            }

            auto accessedIndices = access.getAccessFunction().map(iterationRanges);

            if (!accessedIndices.overlaps(writtenIndices)) {
              continue;
            }

            result += write.getAccessFunction().inverseMap(
                IndexSet(accessedIndices.intersect(writtenIndices)),
                IndexSet(iterationRanges));
          }

          for (const auto& range : result) {
            auto clone = Equation::build(equation->getOperation(), equation->getVariables());

            auto matchedClone = std::make_unique<MatchedEquation>(
                std::move(clone), range, write.getPath());

            equations.add(std::move(matchedClone));
          }

          for (const auto& range : IndexSet(iterationRanges) - result) {
            auto clone = Equation::build(equation->getOperation(), equation->getVariables());

            auto matchedClone = std::make_unique<MatchedEquation>(
                std::move(clone), range, write.getPath());

            equations.add(std::move(matchedClone));
          }
        }

        model.setEquations(equations);
        return mlir::success();
      }

      mlir::LogicalResult convertToScheduledModel(
          mlir::OpBuilder& builder,
          Model<ScheduledEquationsBlock>& result,
          const Model<Equation>& model,
          std::function<bool(const Variable&)> isMatchableFn)
      {
        // Matching process
        Model<MatchedEquation> matchedModel(model.getOperation());
        matchedModel.setDerivativesMap(model.getDerivativesMap());

        if (auto res = match(matchedModel, model, isMatchableFn); mlir::failed(res)) {
          return res;
        }

        if (auto res = splitEquations(builder, matchedModel); mlir::failed(res)) {
          return res;
        }

        // Resolve the algebraic loops
        if (auto res = solveCycles(matchedModel, builder); mlir::failed(res)) {
          if (options.solver != Solver::ida) {
            // Check if the selected solver can deal with cycles. If not, fail.
            return res;
          }
        }

        // Schedule the equations
        if (auto res = schedule(result, matchedModel); mlir::failed(res)) {
          return res;
        }

        result.setDerivativesMap(model.getDerivativesMap());
        return mlir::success();
      }

    private:
      ModelSolvingOptions options;
      unsigned int bitWidth;
  };
}

namespace marco::codegen
{
  std::unique_ptr<mlir::Pass> createModelSolvingPass(ModelSolvingOptions options, unsigned int bitWidth)
  {
    return std::make_unique<ModelSolvingPass>(options, bitWidth);
  }
}
