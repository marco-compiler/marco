#include "marco/Codegen/Conversion/Generic/UnrealizedCastsElimination.h"
#include "marco/Codegen/Conversion/Modelica/LowerToLLVM.h"
#include "marco/Codegen/Conversion/Modelica/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/Support/Debug.h"

namespace
{
  struct UnrealizedCastOpLowering : public mlir::OpRewritePattern<mlir::UnrealizedConversionCastOp>
  {
    using mlir::OpRewritePattern<mlir::UnrealizedConversionCastOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::UnrealizedConversionCastOp op, mlir::PatternRewriter& rewriter) const override {
      rewriter.replaceOp(op, op->getOperands());
      return mlir::success();
    }
  };

  class UnrealizedCastsEliminationPass : public mlir::PassWrapper<UnrealizedCastsEliminationPass, mlir::OperationPass<mlir::ModuleOp>>
  {
    public:
      void runOnOperation() override
      {
        auto module = getOperation();

        if (mlir::failed(castsFolderPass(module))) {
          signalPassFailure();
          return;
        }
      }

    private:
      mlir::LogicalResult castsFolderPass(mlir::ModuleOp module)
      {
        mlir::ConversionTarget target(getContext());

        target.addIllegalOp<mlir::UnrealizedConversionCastOp>();

        target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
          return true;
        });

        mlir::OwningRewritePatternList patterns(&getContext());
        patterns.insert<UnrealizedCastOpLowering>(&getContext());

        return applyPartialConversion(module, target, std::move(patterns));
      }
  };
}

namespace marco::codegen
{
  std::unique_ptr<mlir::Pass> createUnrealizedCastsEliminationPass()
  {
    return std::make_unique<UnrealizedCastsEliminationPass>();
  }
}