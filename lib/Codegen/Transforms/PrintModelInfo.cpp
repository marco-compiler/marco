#include "marco/Codegen/Transforms/PrintModelInfo.h"
#include "marco/Dialect/BaseModelica/BaseModelicaDialect.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_PRINTMODELINFOPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

namespace
{
  class PrintModelInfoPass
      : public mlir::bmodelica::impl::PrintModelInfoPassBase<PrintModelInfoPass>
  {
    public:
      using PrintModelInfoPassBase<PrintModelInfoPass>::PrintModelInfoPassBase;

      void runOnOperation() override;

    private:
      mlir::LogicalResult processModelOp(
        ModelOp modelOp, llvm::raw_ostream& os);

      int64_t getScalarEquationsCount(
          llvm::ArrayRef<EquationInstanceOp> equations);
  };
}

void PrintModelInfoPass::runOnOperation()
{
  mlir::ModuleOp moduleOp = getOperation();
  llvm::SmallVector<ModelOp> modelOps;

  moduleOp.walk([&](ModelOp modelOp) {
    modelOps.push_back(modelOp);
  });

  for (ModelOp modelOp : modelOps) {
    if (mlir::failed(processModelOp(modelOp, llvm::errs()))) {
      return signalPassFailure();
    }
  }

  markAllAnalysesPreserved();
}

mlir::LogicalResult PrintModelInfoPass::processModelOp(
    ModelOp modelOp, llvm::raw_ostream& os)
{
  os << "Model name: " << modelOp.getSymName() << "\n";

  llvm::SmallVector<EquationInstanceOp> initialEquations;
  llvm::SmallVector<EquationInstanceOp> dynamicEquations;

  modelOp.collectInitialEquations(initialEquations);
  modelOp.collectMainEquations(dynamicEquations);

  int64_t numOfScalarInitialEquations =
      getScalarEquationsCount(initialEquations);

  int64_t numOfScalarDynamicEquations =
      getScalarEquationsCount(dynamicEquations);

  os << "# Initial equations: " << numOfScalarInitialEquations << "\n";
  os << "# Dynamic equations: " << numOfScalarDynamicEquations << "\n";

  return mlir::success();
}

int64_t PrintModelInfoPass::getScalarEquationsCount(
    llvm::ArrayRef<EquationInstanceOp> equations)
{
  int64_t result = 0;

  for (EquationInstanceOp equationOp : equations) {
    auto indices = equationOp.getIndices();

    if (indices) {
      // Array equation.
      result += indices->getValue().flatSize();
    } else {
      // Scalar equation.
      ++result;
    }
  }

  return result;
}

namespace mlir::bmodelica
{
  std::unique_ptr<mlir::Pass> createPrintModelInfoPass()
  {
    return std::make_unique<PrintModelInfoPass>();
  }
}
