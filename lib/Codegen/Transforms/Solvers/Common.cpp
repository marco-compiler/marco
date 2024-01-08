#include "marco/Codegen/Transforms/Solvers/Common.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace ::marco;
using namespace ::mlir::modelica;

namespace
{
  class EquationTemplateEquationSidesOpPattern
      : public mlir::OpRewritePattern<EquationSidesOp>
  {
    public:
      using mlir::OpRewritePattern<EquationSidesOp>::OpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          EquationSidesOp op, mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        mlir::Value lhsValue = op.getLhsValues()[0];
        mlir::Value rhsValue = op.getRhsValues()[0];

        if (mlir::failed(convertToAssignment(
                rewriter, loc, lhsValue, rhsValue))) {
          return mlir::failure();
        }

        // Erase the equation terminator and also the operations for its sides.
        auto lhsOp = op.getLhs().getDefiningOp<EquationSideOp>();
        auto rhsOp = op.getRhs().getDefiningOp<EquationSideOp>();
        rewriter.eraseOp(op);
        rewriter.eraseOp(lhsOp);
        rewriter.eraseOp(rhsOp);

        return mlir::success();
      }

    private:
      /// Convert the equality to an assignment.
      mlir::LogicalResult convertToAssignment(
          mlir::OpBuilder& builder, mlir::Location loc,
          mlir::Value lhsValue, mlir::Value rhsValue) const
      {
        assert(!lhsValue.getType().isa<ArrayType>());
        assert(!rhsValue.getType().isa<ArrayType>());

        if (auto loadOp = lhsValue.getDefiningOp<LoadOp>()) {
          // Left-hand side is a scalar element extract from an array variable.
          rhsValue = builder.create<CastOp>(loc, lhsValue.getType(), rhsValue);

          builder.create<StoreOp>(
              loc, rhsValue, loadOp.getArray(), loadOp.getIndices());

          return mlir::success();
        }

        if (auto variableGetOp = lhsValue.getDefiningOp<VariableGetOp>()) {
          // Left-hand side is a scalar variable.
          rhsValue = builder.create<CastOp>(loc, lhsValue.getType(), rhsValue);

          builder.create<VariableSetOp>(
              loc, variableGetOp.getVariable(), rhsValue);

          return mlir::success();
        }

        return mlir::failure();
      }
  };
}

namespace mlir::modelica::impl
{
  ModelSolver::ModelSolver() = default;

  ModelSolver::~ModelSolver() = default;

  ScheduleOp ModelSolver::createSchedule(
      mlir::RewriterBase& rewriter,
      mlir::SymbolTableCollection& symbolTableCollection,
      mlir::ModuleOp moduleOp,
      mlir::Location loc,
      llvm::StringRef scheduleName,
      llvm::ArrayRef<SCCOp> SCCs)
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(moduleOp.getBody());

    // Create the schedule operation.
    auto scheduleOp = rewriter.create<ScheduleOp>(loc, scheduleName);

    symbolTableCollection.getSymbolTable(moduleOp).insert(scheduleOp);
    rewriter.createBlock(&scheduleOp.getBodyRegion());

    // Clone the equations templates into the schedule.
    mlir::IRMapping templatesMapping;

    for (SCCOp scc : SCCs) {
      for (MatchedEquationInstanceOp equation :
           scc.getOps<MatchedEquationInstanceOp>()) {
        EquationTemplateOp templateOp = equation.getTemplate();

        if (templatesMapping.contains(templateOp.getResult())) {
          continue;
        }

        rewriter.setInsertionPointToEnd(scheduleOp.getBody());

        EquationTemplateOp clonedTemplate = cloneEquationTemplateOutsideModel(
            rewriter, symbolTableCollection, moduleOp, equation.getTemplate());

        if (!clonedTemplate) {
          return nullptr;
        }

        templatesMapping.map(templateOp.getResult(), clonedTemplate.getResult());
      }
    }

    // Clone the SCCs.
    rewriter.setInsertionPointToEnd(scheduleOp.getBody());

    for (SCCOp scc : SCCs) {
      rewriter.clone(*scc.getOperation(), templatesMapping);
    }

    return scheduleOp;
  }

  EquationTemplateOp ModelSolver::cloneEquationTemplateOutsideModel(
      mlir::RewriterBase& rewriter,
      mlir::SymbolTableCollection& symbolTableCollection,
      mlir::ModuleOp moduleOp,
      EquationTemplateOp equationTemplateOp)
  {
    auto clonedEquationTemplateOp = mlir::cast<EquationTemplateOp>(
        rewriter.clone(*equationTemplateOp.getOperation()));

    if (mlir::failed(replaceLocalWithSimulationVariables(
            rewriter, symbolTableCollection, moduleOp, clonedEquationTemplateOp))) {
      return nullptr;
    }

    return clonedEquationTemplateOp;
  }

  mlir::LogicalResult ModelSolver::replaceLocalWithSimulationVariables(
      mlir::RewriterBase& rewriter,
      mlir::SymbolTableCollection& symbolTableCollection,
      mlir::ModuleOp moduleOp,
      mlir::Operation* op)
  {
    llvm::SmallVector<VariableGetOp> getOps;

    op->walk([&](VariableGetOp getOp) {
      getOps.push_back(getOp);
    });

    for (VariableGetOp getOp : getOps) {
      auto simulationVariableOp =
          symbolTableCollection.lookupSymbolIn<SimulationVariableOp>(
              moduleOp, getOp.getVariableAttr());

      if (!simulationVariableOp) {
        getOp.emitError() << "simulation variable not found";
        return mlir::failure();
      }

      rewriter.setInsertionPoint(getOp);

      rewriter.replaceOpWithNewOp<SimulationVariableGetOp>(
          getOp, simulationVariableOp);
    }

    return mlir::success();
  }

  RawFunctionOp ModelSolver::createEquationTemplateFunction(
      mlir::OpBuilder& builder,
      mlir::ModuleOp moduleOp,
      mlir::SymbolTableCollection& symbolTableCollection,
      EquationTemplateOp equationTemplateOp,
      uint64_t viewElementIndex,
      llvm::StringRef functionName,
      const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(moduleOp.getBody());

    mlir::Location loc = equationTemplateOp.getLoc();

    // The arguments are the indices.
    size_t numOfExplicitInductions =
        equationTemplateOp.getInductionVariables().size();

    llvm::SmallVector<mlir::Type, 10> argTypes(
        numOfExplicitInductions, builder.getIndexType());

    // Create the function and add it to the symbol table.
    auto rawFunctionOp = builder.create<RawFunctionOp>(
        loc, functionName, builder.getFunctionType(argTypes, std::nullopt));

    symbolTableCollection.getSymbolTable(moduleOp).insert(rawFunctionOp);

    mlir::Block* entryBlock = rawFunctionOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Map the induction variables.
    mlir::IRMapping mapping;

    for (size_t i = 0; i < numOfExplicitInductions; ++i) {
      mapping.map(
          equationTemplateOp.getInductionVariables()[i],
          rawFunctionOp.getArgument(i));
    }

    // Clone the equation body.
    for (auto& op : equationTemplateOp.getOps()) {
      builder.clone(op, mapping);
    }

    mlir::ConversionTarget target(*builder.getContext());
    target.addLegalDialect<ModelicaDialect>();
    target.addIllegalOp<EquationSidesOp>();

    target.addDynamicallyLegalOp<VariableGetOp>([&](VariableGetOp op) {
      auto globalVariableIt = localToGlobalVariablesMap.find(op.getVariable());
      return globalVariableIt == localToGlobalVariablesMap.end();
    });

    target.addDynamicallyLegalOp<VariableSetOp>([&](VariableSetOp op) {
      auto globalVariableIt = localToGlobalVariablesMap.find(op.getVariable());
      return globalVariableIt == localToGlobalVariablesMap.end();
    });

    target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
      return true;
    });

    mlir::RewritePatternSet patterns(builder.getContext());

    patterns.insert<EquationTemplateEquationSidesOpPattern>(
        builder.getContext(), viewElementIndex);

    /*
    patterns.insert<EquationTemplateVariableGetOpPattern>(
        builder.getContext(), localToGlobalVariablesMap);

    patterns.insert<EquationTemplateVariableSetOpPattern>(
        builder.getContext(), localToGlobalVariablesMap);
    */

    if (mlir::failed(applyPartialConversion(
            rawFunctionOp, target, std::move(patterns)))) {
      return nullptr;
    }

    // Create the return operation.
    builder.setInsertionPointToEnd(entryBlock);
    builder.create<RawReturnOp>(loc);

    return rawFunctionOp;
  }

  mlir::LogicalResult ModelSolver::callEquationTemplateFunction(
      mlir::OpBuilder& builder,
      mlir::Location loc,
      MatchedEquationInstanceOp equationOp,
      RawFunctionOp rawFunctionOp) const
  {
    /*
    llvm::SmallVector<mlir::Value, 3> lowerBounds;
    llvm::SmallVector<mlir::Value, 3> upperBounds;
    llvm::SmallVector<mlir::Value, 3> steps;

    auto iterationDirections = equationOp.getIterationDirections();

    auto upperBoundFn = [&](EquationScheduleDirection iterationDirection,
                     const Range& range) -> Point::data_type {
      assert(iterationDirection != marco::modeling::scheduling::Direction::Unknown);

      if (iterationDirection == marco::modeling::scheduling::Direction::Forward) {
        return range.getEnd() - range.getBegin();
      } else {
        return range.getBegin() - range.getEnd();
      }
    };

    // Explicit indices.
    if (auto indices = equationOp.getIndices()) {
      for (size_t i = 0, e = indices->getValue().rank(); i < e; ++i) {
        auto iterationDirection = iterationDirections[i]
                                      .cast<EquationScheduleDirectionAttr>()
                                      .getValue();

        // Begin index.
        lowerBounds.push_back(builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(0)));

        // End index.
        upperBounds.push_back(builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(upperBoundFn(
                     iterationDirection, indices->getValue()[i]))));

        // Step.
        steps.push_back(builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(1)));
      }
    }

    // Implicit indices.
    if (auto indices = equationOp.getImplicitIndices()) {
      for (size_t i = 0, e = indices->getValue().rank(); i < e; ++i) {
        auto iterationDirection = marco::modeling::scheduling::Direction::Forward;

        // Begin index.
        lowerBounds.push_back(builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(0)));

        // End index.
        upperBounds.push_back(builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(upperBoundFn(
                     iterationDirection, indices->getValue()[i]))));

        // Step.
        steps.push_back(builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(1)));
      }
    }

    scf::buildLoopNest(
        builder, loc, lowerBounds, upperBounds, steps,
        [&](mlir::OpBuilder& nestedBuilder, mlir::Location nestedLoc,
            mlir::ValueRange inductions) {
          builder.create<CallOp>(loc, rawFunctionOp, inductions);
        });
        */

    return mlir::success();
  }
}
