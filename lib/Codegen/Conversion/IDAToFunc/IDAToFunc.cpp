#include "marco/Codegen/Conversion/IDAToFunc/IDAToFunc.h"
#include "marco/Codegen/Conversion/IDACommon/LLVMTypeConverter.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

namespace mlir
{
#define GEN_PASS_DEF_IDATOFUNCCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
}

using namespace ::mlir::ida;

namespace
{
  /// Generic conversion pattern that provides some utility functions.
  template<typename Op>
  class IDAOpConversionPattern : public mlir::ConvertOpToLLVMPattern<Op>
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
      : public IDAOpConversionPattern<VariableGetterOp>
  {
    using IDAOpConversionPattern<VariableGetterOp>::IDAOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        VariableGetterOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      mlir::SmallVector<mlir::Type, 1> argsTypes;

      argsTypes.push_back(
          mlir::LLVM::LLVMPointerType::get(rewriter.getI64Type()));

      auto functionType = rewriter.getFunctionType(
          argsTypes,
          getTypeConverter()->convertType(op.getFunctionType().getResult(0)));

      auto newOp = rewriter.replaceOpWithNewOp<mlir::func::FuncOp>(
          op, op.getSymName(), functionType);

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      // The arguments to be passed to the entry block of the old operation.
      llvm::SmallVector<mlir::Value, 3> branchArgs;

      // The equation indices are passed through an array.
      mlir::Value variableIndicesPtr = newOp.getArgument(0);

      for (auto variableIndex : llvm::enumerate(op.getVariableIndices())) {
        mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(
            variableIndicesPtr.getLoc(),
            rewriter.getIntegerAttr(getIndexType(), variableIndex.index()));

        mlir::Value variableIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(
            variableIndicesPtr.getLoc(),
            variableIndicesPtr.getType(),
            variableIndicesPtr,
            index);

        mlir::Value mappedVariableIndex = rewriter.create<mlir::LLVM::LoadOp>(
            variableIndexPtr.getLoc(), variableIndexPtr);

        mappedVariableIndex = getTypeConverter()->materializeSourceConversion(
            rewriter,
            mappedVariableIndex.getLoc(),
            rewriter.getIndexType(),
            mappedVariableIndex);

        branchArgs.push_back(mappedVariableIndex);
      }

      // Create a branch to the entry block of the old region.
      rewriter.create<mlir::cf::BranchOp>(
          loc, &op.getBodyRegion().front(), branchArgs);

      // Convert the return operation.
      auto returnOp =
          mlir::cast<ReturnOp>(op.getBodyRegion().back().getTerminator());

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
          op.getBodyRegion(),
          newOp.getFunctionBody(),
          newOp.getFunctionBody().end());

      return mlir::success();
    }
  };

  struct VariableSetterOpLowering
      : public IDAOpConversionPattern<VariableSetterOp>
  {
    using IDAOpConversionPattern<VariableSetterOp>::IDAOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        VariableSetterOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      mlir::SmallVector<mlir::Type, 1> argsTypes;
      argsTypes.push_back(op.getValue().getType());

      argsTypes.push_back(mlir::LLVM::LLVMPointerType::get(
          rewriter.getI64Type()));

      auto functionType = rewriter.getFunctionType(argsTypes, llvm::None);

      auto newOp = rewriter.replaceOpWithNewOp<mlir::func::FuncOp>(
          op, op.getSymName(), functionType);

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      // The arguments to be passed to the entry block of the old operation.
      llvm::SmallVector<mlir::Value, 3> branchArgs;

      // Map the value.
      branchArgs.push_back(newOp.getArgument(0));

      // The equation indices are also passed through an array.
      mlir::Value variableIndicesPtr = newOp.getArgument(1);

      for (auto variableIndex : llvm::enumerate(op.getVariableIndices())) {
        mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(
            variableIndicesPtr.getLoc(),
            rewriter.getIntegerAttr(getIndexType(), variableIndex.index()));

        mlir::Value variableIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(
            variableIndicesPtr.getLoc(),
            variableIndicesPtr.getType(),
            variableIndicesPtr,
            index);

        mlir::Value mappedVariableIndex = rewriter.create<mlir::LLVM::LoadOp>(
            variableIndexPtr.getLoc(), variableIndexPtr);

        mappedVariableIndex = getTypeConverter()->materializeSourceConversion(
            rewriter,
            mappedVariableIndex.getLoc(),
            rewriter.getIndexType(),
            mappedVariableIndex);

        branchArgs.push_back(mappedVariableIndex);
      }

      // Create a branch to the entry block of the old region.
      rewriter.create<mlir::cf::BranchOp>(
          loc, &op.getBodyRegion().front(), branchArgs);

      // Convert the return operation.
      auto returnOp =
          mlir::cast<ReturnOp>(op.getBodyRegion().back().getTerminator());

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
          op.getBodyRegion(),
          newOp.getFunctionBody(),
          newOp.getFunctionBody().end());

      return mlir::success();
    }
  };

  struct AccessFunctionOpLowering
      : public IDAOpConversionPattern<AccessFunctionOp>
  {
    using IDAOpConversionPattern<AccessFunctionOp>::IDAOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        AccessFunctionOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      mlir::SmallVector<mlir::Type, 2> argsTypes;

      argsTypes.push_back(
          mlir::LLVM::LLVMPointerType::get(rewriter.getI64Type()));

      argsTypes.push_back(
          mlir::LLVM::LLVMPointerType::get(rewriter.getI64Type()));

      auto functionType = rewriter.getFunctionType(argsTypes, llvm::None);

      auto newOp = rewriter.replaceOpWithNewOp<mlir::func::FuncOp>(
          op, op.getSymName(), functionType);

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      // The arguments to be passed to the entry block of the old operation.
      llvm::SmallVector<mlir::Value, 3> branchArgs;

      // The equation indices are passed through an array.
      mlir::Value equationIndicesPtr = newOp.getArgument(0);

      for (auto equationIndex : llvm::enumerate(op.getEquationIndices())) {
        mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(
            equationIndicesPtr.getLoc(),
            rewriter.getIntegerAttr(getIndexType(), equationIndex.index()));

        mlir::Value equationIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(
            equationIndicesPtr.getLoc(),
            equationIndicesPtr.getType(),
            equationIndicesPtr,
            index);

        mlir::Value mappedEquationIndex = rewriter.create<mlir::LLVM::LoadOp>(
            equationIndexPtr.getLoc(), equationIndexPtr);

        mappedEquationIndex = rewriter.create<mlir::arith::IndexCastOp>(
            mappedEquationIndex.getLoc(),
            rewriter.getIndexType(),
            mappedEquationIndex);

        branchArgs.push_back(mappedEquationIndex);
      }

      // Create a branch to the entry block of the old region.
      rewriter.create<mlir::cf::BranchOp>(
          loc, &op.getBodyRegion().front(), branchArgs);

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
        mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(
            variableIndicesPtr.getLoc(),
            rewriter.getIntegerAttr(getIndexType(), i));

        mlir::Value variableIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(
            variableIndicesPtr.getLoc(),
            variableIndicesPtr.getType(),
            variableIndicesPtr,
            index);

        mlir::Value variableIndex = exitBlock->getArgument(i);

        variableIndex = rewriter.create<mlir::arith::IndexCastOp>(
            variableIndex.getLoc(), rewriter.getI64Type(), variableIndex);

        rewriter.create<mlir::LLVM::StoreOp>(
            variableIndexPtr.getLoc(), variableIndex, variableIndexPtr);
      }

      rewriter.create<mlir::func::ReturnOp>(loc, llvm::None);

      // Convert the return operation.
      auto returnOp =
          mlir::cast<ReturnOp>(op.getBodyRegion().back().getTerminator());

      rewriter.setInsertionPoint(returnOp);

      rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(
          returnOp, exitBlock, returnOp.getOperands());

      // Inline the old region.
      rewriter.inlineRegionBefore(op.getBodyRegion(), exitBlock);

      return mlir::success();
    }
  };

  struct ResidualFunctionOpLowering
      : public IDAOpConversionPattern<ResidualFunctionOp>
  {
    using IDAOpConversionPattern<ResidualFunctionOp>::IDAOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        ResidualFunctionOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      mlir::SmallVector<mlir::Type, 3> argsTypes;
      argsTypes.push_back(op.getTime().getType());

      argsTypes.push_back(
          mlir::LLVM::LLVMPointerType::get(rewriter.getI64Type()));

      auto functionType = rewriter.getFunctionType(
          argsTypes,
          op.getFunctionType().getResult(0));

      auto newOp = rewriter.replaceOpWithNewOp<mlir::func::FuncOp>(
          op, op.getSymName(), functionType);

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      // The arguments to be passed to the entry block of the old operation.
      llvm::SmallVector<mlir::Value, 3> branchArgs;

      // Map the time variable.
      branchArgs.push_back(newOp.getArgument(0));

      // The equation indices are also passed through an array.
      mlir::Value equationIndicesPtr = newOp.getArgument(1);

      for (auto equationIndex : llvm::enumerate(op.getEquationIndices())) {
        mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(
            equationIndicesPtr.getLoc(),
            rewriter.getIntegerAttr(getIndexType(), equationIndex.index()));

        mlir::Value equationIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(
            equationIndicesPtr.getLoc(),
            equationIndicesPtr.getType(),
            equationIndicesPtr,
            index);

        mlir::Value mappedEquationIndex = rewriter.create<mlir::LLVM::LoadOp>(
            equationIndexPtr.getLoc(), equationIndexPtr);

        mappedEquationIndex = getTypeConverter()->materializeSourceConversion(
            rewriter,
            mappedEquationIndex.getLoc(),
            rewriter.getIndexType(),
            mappedEquationIndex);

        branchArgs.push_back(mappedEquationIndex);
      }

      // Create a branch to the entry block of the old region.
      rewriter.create<mlir::cf::BranchOp>(
          loc, &op.getBodyRegion().front(), branchArgs);

      // Convert the return operation.
      auto returnOp =
          mlir::cast<ReturnOp>(op.getBodyRegion().back().getTerminator());

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
          op.getBodyRegion(),
          newOp.getFunctionBody(),
          newOp.getFunctionBody().end());

      return mlir::success();
    }
  };

  struct JacobianFunctionOpLowering
      : public IDAOpConversionPattern<JacobianFunctionOp>
  {
    using IDAOpConversionPattern<JacobianFunctionOp>::IDAOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        JacobianFunctionOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      mlir::SmallVector<mlir::Type, 5> argsTypes;

      argsTypes.push_back(op.getTime().getType());

      argsTypes.push_back(
          mlir::LLVM::LLVMPointerType::get(rewriter.getI64Type()));

      argsTypes.push_back(
          mlir::LLVM::LLVMPointerType::get(rewriter.getI64Type()));

      argsTypes.push_back(op.getAlpha().getType());

      auto functionType = rewriter.getFunctionType(
          argsTypes,
          op.getFunctionType().getResult(0));

      auto newOp = rewriter.replaceOpWithNewOp<mlir::func::FuncOp>(
          op, op.getSymName(), functionType);

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      // The arguments to be passed to the entry block of the old operation.
      llvm::SmallVector<mlir::Value, 3> branchArgs;

      // Map the "time" variable.
      branchArgs.push_back(newOp.getArgument(0));

      // The equation indices are also passed through an array.
      mlir::Value equationIndicesPtr = newOp.getArgument(1);

      for (auto equationIndex : llvm::enumerate(op.getEquationIndices())) {
        mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(
            equationIndicesPtr.getLoc(),
            rewriter.getIntegerAttr(getIndexType(), equationIndex.index()));

        mlir::Value equationIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(
            equationIndicesPtr.getLoc(),
            equationIndicesPtr.getType(),
            equationIndicesPtr,
            index);

        mlir::Value mappedEquationIndex = rewriter.create<mlir::LLVM::LoadOp>(
            equationIndexPtr.getLoc(), equationIndexPtr);

        mappedEquationIndex = rewriter.create<mlir::arith::IndexCastOp>(
            mappedEquationIndex.getLoc(),
            rewriter.getIndexType(),
            mappedEquationIndex);

        branchArgs.push_back(mappedEquationIndex);
      }

      // The variable indices are also passed through an array.
      mlir::Value variableIndicesPtr = newOp.getArgument(2);

      for (auto variableIndex : llvm::enumerate(op.getVariableIndices())) {
        mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(
            equationIndicesPtr.getLoc(),
            rewriter.getIntegerAttr(getIndexType(), variableIndex.index()));

        mlir::Value variableIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(
            variableIndicesPtr.getLoc(),
            variableIndicesPtr.getType(),
            variableIndicesPtr,
            index);

        mlir::Value mappedVariableIndex = rewriter.create<mlir::LLVM::LoadOp>(
            variableIndexPtr.getLoc(), variableIndexPtr);

        mappedVariableIndex = getTypeConverter()->materializeSourceConversion(
            rewriter,
            mappedVariableIndex.getLoc(),
            rewriter.getIndexType(),
            mappedVariableIndex);

        branchArgs.push_back(mappedVariableIndex);
      }

      // Add the "alpha" variable.
      branchArgs.push_back(newOp.getArgument(3));

      // Create a branch to the entry block of the old region.
      rewriter.create<mlir::cf::BranchOp>(
          loc, &op.getBodyRegion().front(), branchArgs);

      // Convert the return operation.
      auto returnOp =
          mlir::cast<ReturnOp>(op.getBodyRegion().back().getTerminator());

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
          op.getBodyRegion(),
          newOp.getFunctionBody(),
          newOp.getFunctionBody().end());

      return mlir::success();
    }
  };
}

static void populateIDAToFuncPatterns(
    mlir::RewritePatternSet& patterns,
    mlir::ida::LLVMTypeConverter& typeConverter)
{
  patterns.insert<
      VariableGetterOpLowering,
      VariableSetterOpLowering,
      AccessFunctionOpLowering,
      ResidualFunctionOpLowering,
      JacobianFunctionOpLowering>(typeConverter);
}

namespace
{
  class IDAToFuncConversionPass
      : public mlir::impl::IDAToFuncConversionPassBase<IDAToFuncConversionPass>
  {
    public:
      using IDAToFuncConversionPassBase::IDAToFuncConversionPassBase;

      void runOnOperation() override
      {
        if (mlir::failed(convertOperations())) {
          mlir::emitError(getOperation().getLoc())
              << "Error in converting IDA to Func";

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
            ResidualFunctionOp,
            JacobianFunctionOp,
            ReturnOp>();

        target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
          return true;
        });

        mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
        LLVMTypeConverter typeConverter(&getContext(), llvmLoweringOptions);

        mlir::RewritePatternSet patterns(&getContext());
        populateIDAToFuncPatterns(patterns, typeConverter);

        return applyPartialConversion(moduleOp, target, std::move(patterns));
      }
  };
}

namespace mlir
{
  std::unique_ptr<mlir::Pass> createIDAToFuncConversionPass()
  {
    return std::make_unique<IDAToFuncConversionPass>();
  }

  std::unique_ptr<mlir::Pass> createIDAToFuncConversionPass(
      const IDAToFuncConversionPassOptions& options)
  {
    return std::make_unique<IDAToFuncConversionPass>(options);
  }
}
