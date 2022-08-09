#include "marco/Codegen/Conversion/ModelicaToFunc/ModelicaToFunc.h"
#include "marco/Codegen/Conversion/ModelicaCommon/TypeConverter.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include "marco/Codegen/Conversion/PassDetail.h"

using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace
{
  struct RawFunctionOpLowering : public mlir::OpConversionPattern<RawFunctionOp>
  {
    using mlir::OpConversionPattern<RawFunctionOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(RawFunctionOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Type, 3> argsTypes;
      llvm::SmallVector<mlir::Type, 3> resultsTypes;

      for (const auto& argType : op.getFunctionType().getInputs()) {
        argsTypes.push_back(getTypeConverter()->convertType(argType));
      }

      for (const auto& resultType : op.getFunctionType().getResults()) {
        resultsTypes.push_back(getTypeConverter()->convertType(resultType));
      }

      auto functionType = rewriter.getFunctionType(argsTypes, resultsTypes);
      auto funcOp = rewriter.replaceOpWithNewOp<mlir::func::FuncOp>(op, op.getSymName(), functionType);

      rewriter.inlineRegionBefore(op.getBody(), funcOp.getBody(), funcOp.end());

      if (mlir::failed(rewriter.convertRegionTypes(&funcOp.getBody(), *typeConverter))) {
        return mlir::failure();
      }

      return mlir::success();
    }
  };

  struct RawReturnOpLowering : public mlir::OpConversionPattern<RawReturnOp>
  {
    using mlir::OpConversionPattern<RawReturnOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(RawReturnOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, adaptor.getOperands());
      return mlir::success();
    }
  };

  struct CallOpLowering : public mlir::OpConversionPattern<CallOp>
  {
    using mlir::OpConversionPattern<CallOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(CallOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Type, 3> resultsTypes;

      if (auto res = getTypeConverter()->convertTypes(op->getResultTypes(), resultsTypes); mlir::failed(res)) {
        return res;
      }

      rewriter.replaceOpWithNewOp<mlir::func::CallOp>(op, op.getCallee(), resultsTypes, adaptor.getOperands());
      return mlir::success();
    }
  };
}

static void populateModelicaToFuncPatterns(
    mlir::RewritePatternSet& patterns,
    mlir::MLIRContext* context,
    mlir::TypeConverter& typeConverter)
{
  patterns.insert<
      RawFunctionOpLowering,
      RawReturnOpLowering,
      CallOpLowering>(typeConverter, context);
}

namespace
{
  class ModelicaToFuncConversionPass : public ModelicaToFuncBase<ModelicaToFuncConversionPass>
  {
    public:
      ModelicaToFuncConversionPass(ModelicaToFuncOptions options)
          : options(std::move(options))
      {
      }

      void runOnOperation() override
      {
        if (mlir::failed(convertOperations())) {
          mlir::emitError(getOperation().getLoc(), "Error in converting the Modelica operations");
          return signalPassFailure();
        }
      }

    private:
      mlir::LogicalResult convertOperations()
      {
        auto module = getOperation();
        mlir::ConversionTarget target(getContext());

        target.addIllegalOp<RawFunctionOp, RawReturnOp, CallOp>();
        target.addLegalDialect<mlir::func::FuncDialect>();

        mlir::modelica::TypeConverter typeConverter(options.bitWidth);

        mlir::RewritePatternSet patterns(&getContext());
        populateModelicaToFuncPatterns(patterns, &getContext(), typeConverter);

        return applyPartialConversion(module, target, std::move(patterns));
      }

      private:
        ModelicaToFuncOptions options;
    };
}

namespace marco::codegen
{
  const ModelicaToFuncOptions& ModelicaToFuncOptions::getDefaultOptions()
  {
    static ModelicaToFuncOptions options;
    return options;
  }

  std::unique_ptr<mlir::Pass> createModelicaToFuncPass(ModelicaToFuncOptions options)
  {
    return std::make_unique<ModelicaToFuncConversionPass>(options);
  }
}
