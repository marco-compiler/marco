#include "marco/Codegen/Conversion/SUNDIALSToFunc/SUNDIALSToFunc.h"
#include "marco/Dialect/SUNDIALS/IR/SUNDIALSDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

namespace mlir
{
#define GEN_PASS_DEF_SUNDIALSTOFUNCCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
}

using namespace ::mlir::sundials;

namespace
{
  /// Generic conversion pattern that provides some utility functions.
  template<typename Op>
  class SUNDIALSOpConversionPattern : public mlir::ConvertOpToLLVMPattern<Op>
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
  struct VariableGetterOpLowering
      : public SUNDIALSOpConversionPattern<VariableGetterOp>
  {
    using SUNDIALSOpConversionPattern<VariableGetterOp>
        ::SUNDIALSOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        VariableGetterOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      mlir::SmallVector<mlir::Type, 1> argsTypes;

      argsTypes.push_back(
          mlir::LLVM::LLVMPointerType::get(rewriter.getContext()));

      auto functionType = rewriter.getFunctionType(
          argsTypes,
          getTypeConverter()->convertType(op.getFunctionType().getResult(0)));

      auto newOp = rewriter.create<mlir::func::FuncOp>(
          loc, op.getSymName(), functionType);

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      // The arguments to be passed to the entry block of the old operation.
      llvm::SmallVector<mlir::Value, 3> branchArgs;

      // The equation indices are passed through an array.
      mlir::Value variableIndicesPtr = newOp.getArgument(0);

      for (auto variableIndex : llvm::enumerate(op.getVariableIndices())) {
        auto index = static_cast<int32_t>(variableIndex.index());

        mlir::Value variableIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(
            variableIndicesPtr.getLoc(),
            variableIndicesPtr.getType(),
            rewriter.getI64Type(),
            variableIndicesPtr,
            llvm::ArrayRef<mlir::LLVM::GEPArg>{index});

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

  struct VariableSetterOpLowering
      : public SUNDIALSOpConversionPattern<VariableSetterOp>
  {
    using SUNDIALSOpConversionPattern<VariableSetterOp>
        ::SUNDIALSOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        VariableSetterOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      mlir::SmallVector<mlir::Type, 1> argsTypes;
      argsTypes.push_back(op.getValue().getType());

      argsTypes.push_back(
          mlir::LLVM::LLVMPointerType::get(rewriter.getContext()));

      auto functionType = rewriter.getFunctionType(argsTypes, std::nullopt);

      auto newOp = rewriter.create<mlir::func::FuncOp>(
          loc, op.getSymName(), functionType);

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      // The arguments to be passed to the entry block of the old operation.
      llvm::SmallVector<mlir::Value, 3> branchArgs;

      // Map the value.
      branchArgs.push_back(newOp.getArgument(0));

      // The equation indices are also passed through an array.
      mlir::Value variableIndicesPtr = newOp.getArgument(1);

      for (auto variableIndex : llvm::enumerate(op.getVariableIndices())) {
        auto index = static_cast<int32_t>(variableIndex.index());

        mlir::Value variableIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(
            variableIndicesPtr.getLoc(),
            variableIndicesPtr.getType(),
            rewriter.getI64Type(),
            variableIndicesPtr,
            llvm::ArrayRef<mlir::LLVM::GEPArg>{index});

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

  struct AccessFunctionOpLowering
      : public SUNDIALSOpConversionPattern<AccessFunctionOp>
  {
    using SUNDIALSOpConversionPattern<AccessFunctionOp>
        ::SUNDIALSOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        AccessFunctionOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      mlir::SmallVector<mlir::Type, 2> argsTypes;

      auto pointerType =
          mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

      argsTypes.push_back(pointerType);
      argsTypes.push_back(pointerType);

      auto functionType = rewriter.getFunctionType(argsTypes, std::nullopt);

      auto newOp = rewriter.create<mlir::func::FuncOp>(
          loc, op.getSymName(), functionType);

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      // The arguments to be passed to the entry block of the old operation.
      llvm::SmallVector<mlir::Value, 3> branchArgs;

      // The equation indices are passed through an array.
      mlir::Value equationIndicesPtr = newOp.getArgument(0);

      for (auto equationIndex : llvm::enumerate(op.getEquationIndices())) {
        auto index = static_cast<int32_t>(equationIndex.index());

        mlir::Value equationIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(
            equationIndicesPtr.getLoc(),
            equationIndicesPtr.getType(),
            rewriter.getI64Type(),
            equationIndicesPtr,
            llvm::ArrayRef<mlir::LLVM::GEPArg>{index});

        mlir::Value mappedEquationIndex = rewriter.create<mlir::LLVM::LoadOp>(
            equationIndexPtr.getLoc(), rewriter.getI64Type(), equationIndexPtr);

        mappedEquationIndex = rewriter.create<mlir::arith::IndexCastOp>(
            mappedEquationIndex.getLoc(),
            rewriter.getIndexType(),
            mappedEquationIndex);

        branchArgs.push_back(mappedEquationIndex);
      }

      // Create a branch to the entry block of the old region.
      rewriter.create<mlir::cf::BranchOp>(
          loc, &op.getBody().front(), branchArgs);

      // Create the exit block.
      unsigned int numOfResults = op.getFunctionType().getNumResults();

      llvm::SmallVector<mlir::Type, 3> exitBlockArgTypes(
          numOfResults, rewriter.getIndexType());

      llvm::SmallVector<mlir::Location, 3> exitBlockArgLocs(numOfResults, loc);

      mlir::Block* exitBlock = rewriter.createBlock(
          &newOp.getFunctionBody(),
          newOp.getFunctionBody().end(),
          exitBlockArgTypes, exitBlockArgLocs);

      rewriter.setInsertionPointToStart(exitBlock);

      // Store the results.
      mlir::Value variableIndicesPtr = newOp.getArgument(1);

      for (unsigned int i = 0; i < numOfResults; ++i) {
        auto index = static_cast<int32_t>(i);

        mlir::Value variableIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(
            variableIndicesPtr.getLoc(),
            variableIndicesPtr.getType(),
            rewriter.getI64Type(),
            variableIndicesPtr,
            llvm::ArrayRef<mlir::LLVM::GEPArg>{index});

        mlir::Value variableIndex = exitBlock->getArgument(i);

        variableIndex = rewriter.create<mlir::arith::IndexCastOp>(
            variableIndex.getLoc(), rewriter.getI64Type(), variableIndex);

        rewriter.create<mlir::LLVM::StoreOp>(
            variableIndexPtr.getLoc(), variableIndex, variableIndexPtr);
      }

      rewriter.create<mlir::func::ReturnOp>(loc, std::nullopt);

      // Convert the return operation.
      auto returnOp =
          mlir::cast<ReturnOp>(op.getBody().back().getTerminator());

      rewriter.setInsertionPoint(returnOp);

      rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(
          returnOp, exitBlock, returnOp.getOperands());

      // Inline the old region.
      rewriter.inlineRegionBefore(op.getBody(), exitBlock);

      rewriter.eraseOp(op);
      return mlir::success();
    }
  };
}

static void populateSUNDIALSToFuncPatterns(
    mlir::RewritePatternSet& patterns,
    mlir::LLVMTypeConverter& typeConverter)
{
  patterns.insert<
      VariableGetterOpLowering,
      VariableSetterOpLowering,
      AccessFunctionOpLowering>(typeConverter);
}

namespace
{
  class SUNDIALSToFuncConversionPass
      : public mlir::impl::SUNDIALSToFuncConversionPassBase<
          SUNDIALSToFuncConversionPass>
  {
    public:
      using SUNDIALSToFuncConversionPassBase::SUNDIALSToFuncConversionPassBase;

      void runOnOperation() override
      {
        if (mlir::failed(convertOperations())) {
          mlir::emitError(getOperation().getLoc())
              << "Error in converting SUNDIALS to Func";

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
            VariableGetterOp,
            VariableSetterOp,
            AccessFunctionOp,
            ReturnOp>();

        target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
          return true;
        });

        mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());

        mlir::LLVMTypeConverter typeConverter(
            &getContext(), llvmLoweringOptions);

        mlir::RewritePatternSet patterns(&getContext());
        populateSUNDIALSToFuncPatterns(patterns, typeConverter);

        return applyPartialConversion(moduleOp, target, std::move(patterns));
      }
  };
}

namespace mlir
{
  std::unique_ptr<mlir::Pass> createSUNDIALSToFuncConversionPass()
  {
    return std::make_unique<SUNDIALSToFuncConversionPass>();
  }
}
