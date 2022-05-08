#include "marco/Codegen/Transforms/Matching.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include <set>

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace
{
  class MatchingPass : public mlir::PassWrapper<MatchingPass, mlir::OperationPass<mlir::ModuleOp>>
  {
    public:
      void getDependentDialects(mlir::DialectRegistry& registry) const override
      {
        registry.insert<ModelicaDialect>();
      }

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
        Model<Equation> model(modelOp);
        model.setVariables(discoverVariables(model.getOperation().equationsRegion()));
        model.setEquations(discoverEquations(model.getOperation().equationsRegion(), model.getVariables()));

        Model<MatchedEquation> matchedModel(model.getOperation());

        auto isMatchableFn = [](const Variable& variable) -> IndexSet {
          return IndexSet(variable.getIndices());
        };

        if (auto res = match(matchedModel, model, isMatchableFn); mlir::failed(res)) {
          return res;
        }

        std::set<mlir::Operation*> toBeErased;

        for (auto& equation : matchedModel.getEquations()) {
          auto clone = equation->cloneIR();

          // Matched path
          std::vector<mlir::Attribute> path;
          auto matchedPath = equation->getWrite().getPath();

          if (matchedPath.getEquationSide() == EquationPath::LEFT) {
            path.push_back(builder.getStringAttr("L"));
          } else {
            path.push_back(builder.getStringAttr("R"));
          }

          for (const auto& index : matchedPath) {
            path.push_back(builder.getIndexAttr(index));
          }

          clone->setAttr("matched_path", builder.getArrayAttr(path));

          // Matched indices
          std::vector<mlir::Attribute> ranges;
          auto iterationRanges = equation->getIterationRanges();

          for (size_t i = 0; i < iterationRanges.rank(); ++i) {
            ranges.push_back(builder.getArrayAttr({
              builder.getI64IntegerAttr(iterationRanges[i].getBegin()),
              builder.getI64IntegerAttr(iterationRanges[i].getEnd() - 1)
            }));
          }

          clone->setAttr("matched_indices", builder.getArrayAttr(ranges));

          toBeErased.insert(equation->getOperation());
        }

        for (auto op : toBeErased) {
          Equation::build(mlir::cast<EquationOp>(op), model.getVariables())->eraseIR();
        }

        return mlir::success();
      }
    };
}

namespace marco::codegen
{
  std::unique_ptr<mlir::Pass> createMatchingPass()
  {
    return std::make_unique<MatchingPass>();
  }
}
