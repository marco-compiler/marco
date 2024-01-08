#include "marco/Codegen/Transforms/EquationExplicitation.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_EQUATIONEXPLICITATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  class EquationExplicitationPass
      : public impl::EquationExplicitationPassBase<
            EquationExplicitationPass>
  {
    public:
      using EquationExplicitationPassBase<EquationExplicitationPass>
          ::EquationExplicitationPassBase;

      void runOnOperation() override;

    private:
      mlir::LogicalResult processSchedule(
          mlir::RewriterBase& rewriter,
          mlir::ModuleOp moduleOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          ScheduleOp schedule);

      mlir::LogicalResult processSCCGroup(
          mlir::RewriterBase& rewriter,
          mlir::ModuleOp moduleOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          SCCGroupOp sccGroup);

      mlir::LogicalResult processSCC(
          mlir::RewriterBase& rewriter,
          mlir::ModuleOp moduleOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          SCCOp scc);

      EquationFunctionOp createEquationFunction(
          mlir::RewriterBase& rewriter,
          mlir::ModuleOp moduleOp,
          mlir::SymbolTableCollection& symbolTableCollection,
          ScheduledEquationInstanceOp equation);

      llvm::SmallVector<mlir::Value> shiftInductions(
          mlir::RewriterBase& rewriter,
          mlir::ArrayAttr iterationDirections,
          const MultidimensionalRange& indices,
          mlir::ValueRange loopInductions);

      mlir::LogicalResult cloneEquationTemplateIntoFunction(
          mlir::RewriterBase& rewriter,
          EquationTemplateOp templateOp,
          llvm::ArrayRef<mlir::Value> inductions);
  };
}

void EquationExplicitationPass::runOnOperation()
{
  mlir::ModuleOp moduleOp = getOperation();
  mlir::IRRewriter rewriter(&getContext());
  mlir::SymbolTableCollection symbolTableCollection;
  llvm::SmallVector<ScheduleOp, 2> schedules;

  for (ScheduleOp schedule : moduleOp.getOps<ScheduleOp>()) {
    if (mlir::failed(processSchedule(
            rewriter, moduleOp, symbolTableCollection, schedule))) {
      return signalPassFailure();
    }
  }
}

mlir::LogicalResult EquationExplicitationPass::processSchedule(
    mlir::RewriterBase& rewriter,
    mlir::ModuleOp moduleOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    ScheduleOp schedule)
{
  llvm::SmallVector<SCCGroupOp> SCCGroups;
  schedule.collectSCCGroups(SCCGroups);

  for (SCCGroupOp sccGroup : SCCGroups) {
    if (mlir::failed(processSCCGroup(
            rewriter, moduleOp, symbolTableCollection, sccGroup))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult EquationExplicitationPass::processSCCGroup(
    mlir::RewriterBase& rewriter,
    mlir::ModuleOp moduleOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    SCCGroupOp sccGroup)
{
  llvm::SmallVector<SCCOp> SCCs;
  sccGroup.collectSCCs(SCCs);

  for (SCCOp scc : SCCs) {
    if (mlir::failed(processSCC(
            rewriter, moduleOp, symbolTableCollection, scc))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult EquationExplicitationPass::processSCC(
    mlir::RewriterBase& rewriter,
    mlir::ModuleOp moduleOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    mlir::modelica::SCCOp scc)
{
  llvm::SmallVector<ScheduledEquationInstanceOp> equations;
  scc.collectEquations(equations);

  if (equations.size() > 1) {
    // Cycle among the equations.
    return mlir::success();
  }

  bool isSCCErasable = true;

  for (ScheduledEquationInstanceOp equation : equations) {
    ScheduledEquationInstanceOp explicitEquation =
        equation.cloneAndExplicitate(rewriter, symbolTableCollection);

    if (explicitEquation) {
      EquationFunctionOp eqFunc = createEquationFunction(
          rewriter, moduleOp, symbolTableCollection, explicitEquation);

      if (!eqFunc) {
        return mlir::failure();
      }

      rewriter.setInsertionPoint(scc);
      rewriter.create<CallOp>(eqFunc.getLoc(), eqFunc);
      rewriter.eraseOp(explicitEquation);
    } else {
      isSCCErasable = false;
    }
  }

  if (isSCCErasable) {
    rewriter.eraseOp(scc);
  }

  return mlir::success();
}

EquationFunctionOp EquationExplicitationPass::createEquationFunction(
    mlir::RewriterBase& rewriter,
    mlir::ModuleOp moduleOp,
    mlir::SymbolTableCollection& symbolTableCollection,
    ScheduledEquationInstanceOp equation)
{
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto eqFunc = rewriter.create<EquationFunctionOp>(
      equation.getLoc(), "equation");

  symbolTableCollection.getSymbolTable(moduleOp).insert(eqFunc);

  mlir::Block* entryBlock = eqFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  if (auto indicesAttr = equation.getIndices()) {
    llvm::SmallVector<mlir::Value, 3> lowerBounds;
    llvm::SmallVector<mlir::Value, 3> upperBounds;
    llvm::SmallVector<mlir::Value, 3> steps;

    size_t rank = indicesAttr->getValue().rank();
    auto iterationDirections = equation.getIterationDirections();

    mlir::Value zeroValue = rewriter.create<mlir::arith::ConstantOp>(
        equation.getLoc(), rewriter.getIndexAttr(0));

    mlir::Value oneValue = rewriter.create<mlir::arith::ConstantOp>(
        equation.getLoc(), rewriter.getIndexAttr(1));

    for (size_t i = 0; i < rank; ++i) {
      auto iterationDirection =
          iterationDirections[i].cast<EquationScheduleDirectionAttr>();

      lowerBounds.push_back(zeroValue);
      steps.push_back(oneValue);

      if (iterationDirection.getValue() ==
          EquationScheduleDirection::Forward) {
        auto upperBound = indicesAttr->getValue()[i].getEnd() -
            indicesAttr->getValue()[i].getBegin();

        upperBounds.push_back(rewriter.create<mlir::arith::ConstantOp>(
            equation.getLoc(), rewriter.getIndexAttr(upperBound)));
      } else {
        assert(iterationDirection.getValue() ==
               EquationScheduleDirection::Backward);

        auto upperBound = indicesAttr->getValue()[i].getBegin() -
            indicesAttr->getValue()[i].getEnd();

        upperBounds.push_back(rewriter.create<mlir::arith::ConstantOp>(
            equation.getLoc(), rewriter.getIndexAttr(upperBound)));
      }
    }

    llvm::SmallVector<mlir::Value> inductions;

    for (size_t i = 0; i < rank; ++i) {
      auto forOp = rewriter.create<mlir::scf::ForOp>(
          equation.getLoc(), lowerBounds[i], upperBounds[i], steps[i]);

      inductions.push_back(forOp.getInductionVar());
      rewriter.setInsertionPointToStart(forOp.getBody());
    }

    auto shiftedInductions = shiftInductions(
        rewriter, iterationDirections, indicesAttr->getValue(),
        inductions);

    if (mlir::failed(cloneEquationTemplateIntoFunction(
            rewriter, equation.getTemplate(),
            shiftedInductions))) {
      return nullptr;
    }
  } else {
    if (mlir::failed(cloneEquationTemplateIntoFunction(
            rewriter, equation.getTemplate(),
            std::nullopt))) {
      return nullptr;
    }
  }

  rewriter.setInsertionPointToEnd(entryBlock);
  rewriter.create<YieldOp>(eqFunc.getLoc());
  return eqFunc;
}

llvm::SmallVector<mlir::Value> EquationExplicitationPass::shiftInductions(
    mlir::RewriterBase& rewriter,
    mlir::ArrayAttr iterationDirections,
    const MultidimensionalRange& indices,
    mlir::ValueRange loopInductions)
{
  llvm::SmallVector<mlir::Value> mappedInductions;

  for (size_t i = 0, e = indices.rank(); i < e; ++i) {
    mlir::Value induction = loopInductions[i];

    mlir::Value fromValue = rewriter.create<mlir::arith::ConstantOp>(
        induction.getLoc(),
        rewriter.getIndexAttr(indices[i].getBegin()));

    auto iterationDirection =
        iterationDirections[i].cast<EquationScheduleDirectionAttr>();

    if (iterationDirection.getValue() ==
        EquationScheduleDirection::Forward) {
      mlir::Value mappedInduction = rewriter.create<mlir::arith::AddIOp>(
          induction.getLoc(), rewriter.getIndexType(),
          fromValue, induction);

      mappedInductions.push_back(mappedInduction);
    } else {
      assert(iterationDirection.getValue() ==
             EquationScheduleDirection::Backward);
      mlir::Value mappedInduction = rewriter.create<mlir::arith::SubIOp>(
          induction.getLoc(), rewriter.getIndexType(),
          fromValue, induction);

      mappedInductions.push_back(mappedInduction);
    }
  }

  return mappedInductions;
}

mlir::LogicalResult
EquationExplicitationPass::cloneEquationTemplateIntoFunction(
    mlir::RewriterBase& rewriter,
    EquationTemplateOp templateOp,
    llvm::ArrayRef<mlir::Value> inductions)
{
  assert(templateOp.getInductionVariables().size() == inductions.size());
  mlir::IRMapping mapping;

  for (const auto& [oldInduction, newInduction] : llvm::zip(
           templateOp.getInductionVariables(), inductions)) {
    mapping.map(oldInduction, newInduction);
  }

  for (auto& op : templateOp.getOps()) {
    if (mlir::isa<EquationSideOp>(op)) {
      continue;
    }

    if (auto equationSidesOp = mlir::dyn_cast<EquationSidesOp>(op)) {
      mlir::Value lhs = equationSidesOp.getLhsValues()[0];
      mlir::Value rhs = equationSidesOp.getRhsValues()[0];

      mlir::Value mappedLhs = mapping.lookup(lhs);
      mlir::Value mappedRhs = mapping.lookup(rhs);

      if (auto loadOp = mappedLhs.getDefiningOp<LoadOp>()) {
        // Left-hand side is a scalar element extracted from an array
        // variable.
        if (auto lhsType = mappedLhs.getType();
            lhsType != mappedRhs.getType()) {
          mappedRhs = rewriter.create<CastOp>(
              mappedRhs.getLoc(), mappedLhs.getType(), mappedRhs);
        }

        rewriter.create<StoreOp>(
            equationSidesOp.getLoc(), mappedRhs,
            loadOp.getArray(), loadOp.getIndices());

        continue;
      }

      if (auto getOp = mappedLhs.getDefiningOp<SimulationVariableGetOp>()) {
        if (getOp.getType().isa<ArrayType>()) {
          rewriter.create<ArrayCopyOp>(
              equationSidesOp.getLoc(), mappedRhs, getOp);
        } else {
          // Left-hand side is a scalar variable.
          if (auto lhsType = mappedLhs.getType();
              lhsType != mappedRhs.getType()) {
            mappedRhs = rewriter.create<CastOp>(
                mappedRhs.getLoc(), mappedLhs.getType(), mappedRhs);
          }

          rewriter.create<SimulationVariableSetOp>(
              equationSidesOp.getLoc(), getOp.getVariable(),
              mappedRhs);

          return mlir::success();
        }
      }
    } else {
      rewriter.clone(op, mapping);
    }
  }

  return mlir::success();
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createEquationExplicitationPass()
  {
    return std::make_unique<EquationExplicitationPass>();
  }
}
