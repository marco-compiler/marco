#include "marco/Codegen/Conversion/KINSOLToFunc/KINSOLToFunc.h"
#include "marco/Codegen/Conversion/KINSOLCommon/LLVMTypeConverter.h"
#include "marco/Dialect/KINSOL/KINSOLDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

namespace mlir
{
#define GEN_PASS_DEF_KINSOLTOFUNCCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
}

using namespace ::mlir::kinsol;

namespace
{
  /// Generic conversion pattern that provides some utility functions.
  template<typename Op>
  class KINSOLOpConversionPattern : public mlir::ConvertOpToLLVMPattern<Op>
  {
    public:
      using mlir::ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

      mlir::Value materializeTargetConversion(
          mlir::OpBuilder& builder, mlir::Value value) const
      {
        mlir::Type type =
            this->getTypeConverter()->convertType(value.getType());

        return this->getTypeConverter()->materializeTargetConversion(
            builder, value.getLoc(), type, value);
      }

      void materializeTargetConversion(
          mlir::OpBuilder& builder,
          llvm::SmallVectorImpl<mlir::Value>& values) const
      {
        for (auto& value : values) {
          value = materializeTargetConversion(builder, value);
        }
      }
    };
}

namespace
{
  struct ResidualFunctionOpLowering
      : public KINSOLOpConversionPattern<ResidualFunctionOp>
  {
    using KINSOLOpConversionPattern<ResidualFunctionOp>
        ::KINSOLOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        ResidualFunctionOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      mlir::SmallVector<mlir::Type, 3> argsTypes;

      argsTypes.push_back(
          mlir::LLVM::LLVMPointerType::get(rewriter.getContext()));

      auto functionType = rewriter.getFunctionType(
          argsTypes,
          op.getFunctionType().getResult(0));

      auto newOp = rewriter.create<mlir::func::FuncOp>(
          loc, op.getSymName(), functionType);

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      // The arguments to be passed to the entry block of the old operation.
      llvm::SmallVector<mlir::Value, 3> branchArgs;

      // The equation indices are also passed through an array.
      mlir::Value equationIndicesPtr = newOp.getArgument(0);

      for (auto equationIndex : llvm::enumerate(op.getEquationIndices())) {
        mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(
            equationIndicesPtr.getLoc(),
            rewriter.getIntegerAttr(getIndexType(), equationIndex.index()));

        mlir::Value equationIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(
            equationIndicesPtr.getLoc(),
            equationIndicesPtr.getType(),
            rewriter.getI64Type(),
            equationIndicesPtr,
            index);

        mlir::Value mappedEquationIndex = rewriter.create<mlir::LLVM::LoadOp>(
            equationIndexPtr.getLoc(), rewriter.getI64Type(), equationIndexPtr);

        mappedEquationIndex = getTypeConverter()->materializeSourceConversion(
            rewriter,
            mappedEquationIndex.getLoc(),
            rewriter.getIndexType(),
            mappedEquationIndex);

        branchArgs.push_back(mappedEquationIndex);
      }

      // Create a branch to the entry block of the old region.
      rewriter.create<mlir::cf::BranchOp>(
          loc, &op.getBody().front(), branchArgs);

      // Convert the return operation.
      auto returnOp =
          mlir::cast<ReturnOp>(op.getBody().back().getTerminator());

      llvm::SmallVector<mlir::Value, 1> returnValues;
      rewriter.setInsertionPoint(returnOp);

      for (mlir::Value returnValue : returnOp.getOperands()) {
        returnValues.push_back(
            materializeTargetConversion(rewriter, returnValue));
      }

      rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(
          returnOp, returnValues);

      // Inline the old region.
      rewriter.inlineRegionBefore(
          op.getBody(),
          newOp.getFunctionBody(),
          newOp.getFunctionBody().end());

      rewriter.eraseOp(op);
      return mlir::success();
    }
  };

  struct JacobianFunctionOpLowering
      : public KINSOLOpConversionPattern<JacobianFunctionOp>
  {
    using KINSOLOpConversionPattern<JacobianFunctionOp>
        ::KINSOLOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        JacobianFunctionOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      mlir::SmallVector<mlir::Type, 5> argsTypes;

      argsTypes.push_back(
          mlir::LLVM::LLVMPointerType::get(rewriter.getContext()));

      argsTypes.push_back(
          mlir::LLVM::LLVMPointerType::get(rewriter.getContext()));

      auto functionType = rewriter.getFunctionType(
          argsTypes,
          op.getFunctionType().getResult(0));

      auto newOp = rewriter.create<mlir::func::FuncOp>(
          loc, op.getSymName(), functionType);

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      // The arguments to be passed to the entry block of the old operation.
      llvm::SmallVector<mlir::Value, 3> branchArgs;

      // The equation indices are also passed through an array.
      mlir::Value equationIndicesPtr = newOp.getArgument(0);

      for (auto equationIndex : llvm::enumerate(op.getEquationIndices())) {
        mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(
            equationIndicesPtr.getLoc(),
            rewriter.getIntegerAttr(getIndexType(), equationIndex.index()));

        mlir::Value equationIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(
            equationIndicesPtr.getLoc(),
            equationIndicesPtr.getType(),
            rewriter.getI64Type(),
            equationIndicesPtr,
            index);

        mlir::Value mappedEquationIndex = rewriter.create<mlir::LLVM::LoadOp>(
            equationIndexPtr.getLoc(), rewriter.getI64Type(), equationIndexPtr);

        mappedEquationIndex = rewriter.create<mlir::arith::IndexCastOp>(
            mappedEquationIndex.getLoc(),
            rewriter.getIndexType(),
            mappedEquationIndex);

        branchArgs.push_back(mappedEquationIndex);
      }

      // The variable indices are also passed through an array.
      mlir::Value variableIndicesPtr = newOp.getArgument(1);

      for (auto variableIndex : llvm::enumerate(op.getVariableIndices())) {
        mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(
            equationIndicesPtr.getLoc(),
            rewriter.getIntegerAttr(getIndexType(), variableIndex.index()));

        mlir::Value variableIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(
            variableIndicesPtr.getLoc(),
            variableIndicesPtr.getType(),
            rewriter.getI64Type(),
            variableIndicesPtr,
            index);

        mlir::Value mappedVariableIndex = rewriter.create<mlir::LLVM::LoadOp>(
            variableIndexPtr.getLoc(), rewriter.getI64Type(), variableIndexPtr);

        mappedVariableIndex = getTypeConverter()->materializeSourceConversion(
            rewriter,
            mappedVariableIndex.getLoc(),
            rewriter.getIndexType(),
            mappedVariableIndex);

        branchArgs.push_back(mappedVariableIndex);
      }

      // Create a branch to the entry block of the old region.
      rewriter.create<mlir::cf::BranchOp>(
          loc, &op.getBody().front(), branchArgs);

      // Convert the return operation.
      auto returnOp =
          mlir::cast<ReturnOp>(op.getBody().back().getTerminator());

      llvm::SmallVector<mlir::Value, 1> returnValues;
      rewriter.setInsertionPoint(returnOp);

      for (mlir::Value returnValue : returnOp.getOperands()) {
        returnValues.push_back(
            materializeTargetConversion(rewriter, returnValue));
      }

      rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(
          returnOp, returnValues);

      // Inline the old region.
      rewriter.inlineRegionBefore(
          op.getBody(),
          newOp.getFunctionBody(),
          newOp.getFunctionBody().end());

      rewriter.eraseOp(op);
      return mlir::success();
    }
  };
}

static void populateKINSOLToFuncPatterns(
    mlir::RewritePatternSet& patterns,
    mlir::kinsol::LLVMTypeConverter& typeConverter)
{
  patterns.insert<
      ResidualFunctionOpLowering,
      JacobianFunctionOpLowering>(typeConverter);
}

namespace
{
  class KINSOLToFuncConversionPass
      : public mlir::impl::KINSOLToFuncConversionPassBase<
          KINSOLToFuncConversionPass>
  {
    public:
      using KINSOLToFuncConversionPassBase<KINSOLToFuncConversionPass>
          ::KINSOLToFuncConversionPassBase;

      void runOnOperation() override
      {
        if (mlir::failed(convertOperations())) {
          getOperation().emitError() << "Error in converting KINSOL to Func";

          return signalPassFailure();
        }
      }

    private:
      mlir::LogicalResult convertOperations()
      {
        mlir::ModuleOp moduleOp = getOperation();
        mlir::ConversionTarget target(getContext());

        target.addLegalDialect<
            mlir::arith::ArithDialect,
            mlir::cf::ControlFlowDialect,
            mlir::func::FuncDialect,
            mlir::LLVM::LLVMDialect>();

        target.addIllegalOp<
            ResidualFunctionOp,
            JacobianFunctionOp,
            ReturnOp>();

        target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
          return true;
        });

        mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
        LLVMTypeConverter typeConverter(&getContext(), llvmLoweringOptions);

        mlir::RewritePatternSet patterns(&getContext());
        populateKINSOLToFuncPatterns(patterns, typeConverter);

        return applyPartialConversion(moduleOp, target, std::move(patterns));
      }
  };
}

namespace mlir
{
  std::unique_ptr<mlir::Pass> createKINSOLToFuncConversionPass()
  {
    return std::make_unique<KINSOLToFuncConversionPass>();
  }

  std::unique_ptr<mlir::Pass> createKINSOLToFuncConversionPass(
      const KINSOLToFuncConversionPassOptions& options)
  {
    return std::make_unique<KINSOLToFuncConversionPass>(options);
  }
}
