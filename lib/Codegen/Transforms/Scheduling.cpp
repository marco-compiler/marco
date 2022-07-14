#include "marco/Codegen/Transforms/Scheduling.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include "marco/Codegen/Transforms/ModelSolving/Scheduling.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include <set>

#include "marco/Codegen/Transforms/PassDetail.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

static mlir::Operation* getEquationRoot(EquationInterface equation)
{
  ForEquationOp parent = equation->getParentOfType<ForEquationOp>();

  if (parent == nullptr) {
    return equation.getOperation();
  }

  while (parent->getParentOfType<ForEquationOp>() != nullptr) {
    parent = parent->getParentOfType<ForEquationOp>();
  }

  return parent.getOperation();
}

namespace
{
  class SchedulingPass : public SchedulingBase<SchedulingPass>
  {
    public:
      void runOnOperation() override
      {
        mlir::OpBuilder builder(getOperation());
        llvm::SmallVector<ModelOp, 1> models;

        getOperation()->walk([&](ModelOp model) {
          models.push_back(model);
        });

        for (auto model : models) {
          if (mlir::failed(processModel(builder, model))) {
            return signalPassFailure();
          }
        }
      }

      mlir::LogicalResult processModel(mlir::OpBuilder& builder, ModelOp modelOp) const
      {
        Model<MatchedEquation> matchedModel(modelOp);
        matchedModel.setVariables(discoverVariables(modelOp));

        Equations<MatchedEquation> matchedEquations;

        for (auto& equation : discoverEquations(modelOp, matchedModel.getVariables())) {
          // Matched indices
          std::vector<modeling::Range> ranges;

          if (equation->getOperation()->hasAttrOfType<mlir::ArrayAttr>("matched_indices")) {
            auto matchedIndices = equation->getOperation()->getAttr("matched_indices").cast<mlir::ArrayAttr>();

            for (const auto& range : matchedIndices) {
              auto rangeAttr = range.cast<mlir::ArrayAttr>();
              assert(rangeAttr.size() == 2);

              ranges.emplace_back(
                  rangeAttr[0].cast<mlir::IntegerAttr>().getInt(),
                  rangeAttr[1].cast<mlir::IntegerAttr>().getInt() + 1);
            }
          } else {
            auto iterationRangesSet = equation->getIterationRanges(); //todo: handle ragged case
            assert(iterationRangesSet.isSingleMultidimensionalRange());
            auto iterationRanges = equation->getIterationRanges().minContainingRange();

            for (size_t i = 0; i < equation->getNumOfIterationVars(); ++i) {
              ranges.push_back(iterationRanges[i]);
            }
          }

          // Matched path
          EquationPath::EquationSide side = marco::codegen::EquationPath::LEFT;
          std::vector<size_t> pathIndices;

          if (equation->getOperation()->hasAttrOfType<mlir::ArrayAttr>("matched_path")) {
            auto matchedPath = equation->getOperation()->getAttr("matched_path").cast<mlir::ArrayAttr>();
            side = matchedPath[0].cast<mlir::StringAttr>().getValue() == "L" ? EquationPath::EquationSide::LEFT : EquationPath::EquationSide::RIGHT;

            for (size_t i = 1; i < matchedPath.size(); ++i) {
              pathIndices.push_back(matchedPath[i].cast<mlir::IntegerAttr>().getInt());
            }
          }

          // Create the matched equation
          auto matchedEquation = std::make_unique<MatchedEquation>(
              std::move(equation), modeling::IndexSet(modeling::MultidimensionalRange(ranges)), EquationPath(side, pathIndices));

          matchedEquations.add(std::move(matchedEquation));
        }

        matchedModel.setEquations(matchedEquations);

        Model<ScheduledEquationsBlock> scheduledModel(modelOp);

        if (auto res = schedule(scheduledModel, matchedModel); mlir::failed(res)) {
          return res;
        }

        std::set<mlir::Operation*> toBeErased;
        size_t cyclesCounter = 0;

        for (auto& scheduledBlock : scheduledModel.getScheduledBlocks()) {
          if (scheduledBlock->hasCycle()) {
            for (auto& equation : *scheduledBlock) {
              auto clone = equation->cloneIR();

              clone->removeAttr("matched_indices");
              clone->removeAttr("matched_path");

              clone->setAttr("cycle", builder.getI64IntegerAttr(cyclesCounter));

              builder.setInsertionPointToEnd(modelOp.bodyBlock());
              builder.clone(*getEquationRoot(clone));

              toBeErased.insert(clone.getOperation());
              toBeErased.insert(equation->getOperation().getOperation());
            }

            ++cyclesCounter;

          } else {
            for (auto& equation : *scheduledBlock) {
              auto clone = equation->cloneIR();

              clone->removeAttr("matched_indices");
              clone->removeAttr("matched_path");

              // Scheduled indices
              std::vector<mlir::Attribute> ranges;

              auto iterationRangesSet = equation->getIterationRanges(); //todo: handle ragged case
              assert(iterationRangesSet.isSingleMultidimensionalRange());
              auto iterationRanges = equation->getIterationRanges().minContainingRange();

              for (size_t i = 0; i < iterationRanges.rank(); ++i) {
                ranges.push_back(builder.getArrayAttr({
                    builder.getI64IntegerAttr(iterationRanges[i].getBegin()),
                    builder.getI64IntegerAttr(iterationRanges[i].getEnd() - 1)
                }));
              }

              clone->setAttr("scheduled_indices", builder.getArrayAttr(ranges));

              // Scheduling direction
              mlir::Attribute schedulingDirection;

              if (auto direction = equation->getSchedulingDirection(); direction == marco::modeling::scheduling::Direction::None) {
                schedulingDirection = builder.getStringAttr("none");
              } else if (direction == marco::modeling::scheduling::Direction::Forward) {
                schedulingDirection = builder.getStringAttr("forward");
              } else if (direction == marco::modeling::scheduling::Direction::Backward) {
                schedulingDirection = builder.getStringAttr("backward");
              } else if (direction == marco::modeling::scheduling::Direction::Constant) {
                schedulingDirection = builder.getStringAttr("constant");
              } else if (direction == marco::modeling::scheduling::Direction::Mixed) {
                schedulingDirection = builder.getStringAttr("mixed");
              } else {
                schedulingDirection = builder.getStringAttr("unknown");
              }

              clone->setAttr("scheduled_direction", schedulingDirection);

              builder.setInsertionPointToEnd(modelOp.bodyBlock());
              builder.clone(*getEquationRoot(clone));

              toBeErased.insert(clone.getOperation());
              toBeErased.insert(equation->getOperation().getOperation());
            }
          }
        }

        for (auto op : toBeErased) {
          Equation::build(mlir::cast<EquationOp>(op), scheduledModel.getVariables())->eraseIR();
        }

        return mlir::success();
      }
  };
}

namespace marco::codegen
{
  std::unique_ptr<mlir::Pass> createSchedulingPass()
  {
    return std::make_unique<SchedulingPass>();
  }
}
