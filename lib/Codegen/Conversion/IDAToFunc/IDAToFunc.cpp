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
      IDAOpConversionPattern(
          mlir::LLVMTypeConverter& typeConverter,
          unsigned int bitWidth)
          : mlir::ConvertOpToLLVMPattern<Op>(typeConverter),
            bitWidth(bitWidth)
      {
      }

      mlir::ida::LLVMTypeConverter& typeConverter() const
      {
        return *static_cast<mlir::ida::LLVMTypeConverter*>(
            this->getTypeConverter());
      }

      mlir::Type convertType(mlir::Type type) const
      {
        return typeConverter().convertType(type);
      }

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

    protected:
      unsigned int bitWidth;
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

      mlir::SmallVector<mlir::Type, 3> argsTypes;

      argsTypes.push_back(getVoidPtrType());

      argsTypes.push_back(mlir::LLVM::LLVMPointerType::get(
          getTypeConverter()->getIndexType()));

      auto functionType = rewriter.getFunctionType(
          argsTypes,
          getTypeConverter()->convertType(op.getFunctionType().getResult(0)));

      auto newOp = rewriter.replaceOpWithNewOp<mlir::func::FuncOp>(
          op, op.getSymName(), functionType);

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      // The arguments to be passed to the entry block of the old operation.
      llvm::SmallVector<mlir::Value, 3> branchArgs;

      // The lowered function will receive a void pointer to the array
      // descriptor of the variable.
      mlir::Value variableOpaquePtr = newOp.getArgument(0);

      mlir::Value variablePtr = rewriter.create<mlir::LLVM::BitcastOp>(
          loc, mlir::LLVM::LLVMPointerType::get(
                   op.getVariable().getType()), variableOpaquePtr);

      mlir::Value variable = rewriter.create<mlir::LLVM::LoadOp>(loc, variablePtr);
      branchArgs.push_back(variable);

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

      mlir::SmallVector<mlir::Type, 3> argsTypes;

      argsTypes.push_back(getVoidPtrType());
      argsTypes.push_back(op.getValue().getType());

      argsTypes.push_back(mlir::LLVM::LLVMPointerType::get(
          getTypeConverter()->getIndexType()));

      auto functionType = rewriter.getFunctionType(argsTypes, llvm::None);

      auto newOp = rewriter.replaceOpWithNewOp<mlir::func::FuncOp>(
          op, op.getSymName(), functionType);

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      // The arguments to be passed to the entry block of the old operation.
      llvm::SmallVector<mlir::Value, 3> branchArgs;

      // The lowered function will receive a void pointer to the array
      // descriptor of the variable
      mlir::Value variableOpaquePtr = newOp.getArgument(0);

      mlir::Value variablePtr = rewriter.create<mlir::LLVM::BitcastOp>(
          loc, mlir::LLVM::LLVMPointerType::get(
                   op.getVariable().getType()), variableOpaquePtr);

      mlir::Value variable = rewriter.create<mlir::LLVM::LoadOp>(loc, variablePtr);
      branchArgs.push_back(variable);

      // Map the value.
      branchArgs.push_back(newOp.getArgument(1));

      // The equation indices are also passed through an array.
      mlir::Value variableIndicesPtr = newOp.getArgument(2);

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
      argsTypes.push_back(getVoidPtrType());

      argsTypes.push_back(mlir::LLVM::LLVMPointerType::get(
          getTypeConverter()->getIndexType()));

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

      // The lowered function will receive a pointer to the array of variables.
      assert(!op.getVariables().empty());

      mlir::Value variablesPtr = newOp.getArgument(1);

      variablesPtr = rewriter.create<mlir::LLVM::BitcastOp>(
          variablesPtr.getLoc(),
          mlir::LLVM::LLVMPointerType::get(getVoidPtrType()), variablesPtr);

      for (auto variable : llvm::enumerate(op.getVariables())) {
        if (variable.value().getUses().empty()) {
          continue;
        }

        mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(
            variablesPtr.getLoc(),
            rewriter.getIntegerAttr(getIndexType(), variable.index()));

        mlir::Value variablePtr = rewriter.create<mlir::LLVM::GEPOp>(
            variablesPtr.getLoc(),
            variablesPtr.getType(),
            variablesPtr,
            index);

        mlir::Value mappedVariable = rewriter.create<mlir::LLVM::LoadOp>(
            variablePtr.getLoc(), variablePtr);

        mappedVariable = rewriter.create<mlir::LLVM::BitcastOp>(
            mappedVariable.getLoc(),
            mlir::LLVM::LLVMPointerType::get(
                op.getVariables()[variable.index()].getType()),
            mappedVariable);

        mappedVariable = rewriter.create<mlir::LLVM::LoadOp>(
            mappedVariable.getLoc(), mappedVariable);

        branchArgs.push_back(mappedVariable);
      }

      // The equation indices are also passed through an array.
      mlir::Value equationIndicesPtr = newOp.getArgument(2);

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
      argsTypes.push_back(getVoidPtrType());

      argsTypes.push_back(mlir::LLVM::LLVMPointerType::get(
          getTypeConverter()->getIndexType()));

      argsTypes.push_back(mlir::LLVM::LLVMPointerType::get(
          getTypeConverter()->getIndexType()));

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

      // The lowered function will receive a pointer to the array of variables.
      assert(!op.getVariables().empty());

      mlir::Value variablesPtr = newOp.getArgument(1);

      variablesPtr = rewriter.create<mlir::LLVM::BitcastOp>(
          variablesPtr.getLoc(),
          mlir::LLVM::LLVMPointerType::get(getVoidPtrType()), variablesPtr);

      for (auto variable : llvm::enumerate(op.getVariables())) {
        if (variable.value().getUses().empty()) {
          continue;
        }

        mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(
            variablesPtr.getLoc(),
            rewriter.getIntegerAttr(getIndexType(), variable.index()));

        mlir::Value variablePtr = rewriter.create<mlir::LLVM::GEPOp>(
            variablesPtr.getLoc(),
            variablesPtr.getType(),
            variablesPtr,
            index);

        mlir::Value mappedVariable = rewriter.create<mlir::LLVM::LoadOp>(
            variablePtr.getLoc(), variablePtr);

        mappedVariable = rewriter.create<mlir::LLVM::BitcastOp>(
            mappedVariable.getLoc(),
            mlir::LLVM::LLVMPointerType::get(
                op.getVariables()[variable.index()].getType()),
            mappedVariable);

        mappedVariable = rewriter.create<mlir::LLVM::LoadOp>(
            mappedVariable.getLoc(), mappedVariable);

        branchArgs.push_back(mappedVariable);
      }

      // The equation indices are also passed through an array.
      mlir::Value equationIndicesPtr = newOp.getArgument(2);

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

      // The variable indices are also passed through an array.
      mlir::Value variableIndicesPtr = newOp.getArgument(3);

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
      branchArgs.push_back(newOp.getArgument(4));

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
    mlir::ida::LLVMTypeConverter& typeConverter,
    unsigned int bitWidth)
{
  patterns.insert<
      VariableGetterOpLowering,
      VariableSetterOpLowering,
      ResidualFunctionOpLowering,
      JacobianFunctionOpLowering>(typeConverter, bitWidth);
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
          mlir::emitError(getOperation().getLoc(),
                          "Error in converting the Modelica operations");
          return signalPassFailure();
        }
      }

    private:
      mlir::LogicalResult convertOperations()
      {
        auto module = getOperation();
        mlir::ConversionTarget target(getContext());

        target.addLegalDialect<mlir::LLVM::LLVMDialect>();

        target.addIllegalOp<
            VariableGetterOp,
            VariableSetterOp,
            ResidualFunctionOp,
            JacobianFunctionOp,
            ReturnOp>();

        target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
          return true;
        });

        mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
        LLVMTypeConverter typeConverter(&getContext(), llvmLoweringOptions);

        mlir::RewritePatternSet patterns(&getContext());
        populateIDAToFuncPatterns(patterns, typeConverter, bitWidth);

        return applyPartialConversion(module, target, std::move(patterns));
      }
  };
}

namespace
{
  struct VariableGetterOpTypes
      : public mlir::OpConversionPattern<VariableGetterOp>
  {
    using mlir::OpConversionPattern<VariableGetterOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        VariableGetterOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Type resultType =
          getTypeConverter()->convertType(op.getFunctionType().getResult(0));

      mlir::Type variableType =
          getTypeConverter()->convertType(op.getVariable().getType());

      auto newOp = rewriter.replaceOpWithNewOp<VariableGetterOp>(
          op, op.getSymName(), resultType, variableType,
          op.getVariableIndices().size());

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      mlir::BlockAndValueMapping mapping;

      // Map the entry block.
      mapping.map(&op.getFunctionBody().front(), entryBlock);

      for (const auto& [original, cloned]
           : llvm::zip(op.getFunctionBody().getArguments(),
                       newOp.getFunctionBody().getArguments())) {
        mapping.map(original, cloned);
      }

      auto castArgFn = [&](mlir::Value originalArg) {
        auto castedClonedArg = getTypeConverter()->materializeSourceConversion(
            rewriter, originalArg.getLoc(), originalArg.getType(),
            mapping.lookup(originalArg));

        mapping.map(originalArg, castedClonedArg);
      };

      castArgFn(op.getVariable());

      // Create the remaining blocks and map their arguments.
      for (auto& block : llvm::enumerate(op.getBodyRegion().getBlocks())) {
        if (block.index() != 0) {
          std::vector<mlir::Location> argLocations;
          std::vector<mlir::Type> argTypes;

          for (const auto& arg : block.value().getArguments()) {
            argLocations.push_back(arg.getLoc());
            argTypes.push_back(getTypeConverter()->convertType(arg.getType()));
          }

          mlir::Block* clonedBlock = rewriter.createBlock(
              &newOp.getFunctionBody(),
              newOp.getFunctionBody().end(),
              argTypes,
              argLocations);

          mapping.map(&block.value(), clonedBlock);

          for (const auto& [original, cloned]
               : llvm::zip(block.value().getArguments(),
                           clonedBlock->getArguments())) {
            mlir::Value mapped = cloned;

            if (mapped.getType() != cloned.getType()) {
              mapped = getTypeConverter()->materializeSourceConversion(
                  rewriter, original.getLoc(), original.getType(), mapped);
            }

            mapping.map(original, mapped);
          }
        }
      }

      // Clone the operations.
      for (auto& block : op.getFunctionBody().getBlocks()) {
        mlir::Block* clonedBlock = mapping.lookup(&block);
        rewriter.setInsertionPointToEnd(clonedBlock);

        for (auto& bodyOp : block.getOperations()) {
          if (auto returnOp = mlir::dyn_cast<ReturnOp>(bodyOp)) {
            std::vector<mlir::Value> returnValues;

            for (auto returnValue : returnOp.operands()) {
              returnValues.push_back(
                  getTypeConverter()->materializeTargetConversion(
                      rewriter, returnOp.getLoc(),
                      getTypeConverter()->convertType(returnValue.getType()),
                      mapping.lookup(returnValue)));
            }

            rewriter.create<ReturnOp>(returnOp.getLoc(), returnValues);
          } else {
            rewriter.clone(bodyOp, mapping);
          }
        }
      }

      return mlir::success();
    }
  };

  struct VariableSetterOpTypes
      : public mlir::OpConversionPattern<VariableSetterOp>
  {
    using mlir::OpConversionPattern<VariableSetterOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        VariableSetterOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Type variableType =
          getTypeConverter()->convertType(op.getVariable().getType());

      mlir::Type valueType =
          getTypeConverter()->convertType(op.getValue().getType());

      auto newOp = rewriter.replaceOpWithNewOp<VariableSetterOp>(
          op, op.getSymName(), variableType, valueType,
          op.getVariableIndices().size());

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      mlir::BlockAndValueMapping mapping;

      // Map the entry block.
      mapping.map(&op.getFunctionBody().front(), entryBlock);

      for (const auto& [original, cloned]
           : llvm::zip(op.getFunctionBody().getArguments(),
                       newOp.getFunctionBody().getArguments())) {
        mapping.map(original, cloned);
      }

      auto castArgFn = [&](mlir::Value originalArg) {
        auto castedClonedArg = getTypeConverter()->materializeSourceConversion(
            rewriter, originalArg.getLoc(), originalArg.getType(),
            mapping.lookup(originalArg));

        mapping.map(originalArg, castedClonedArg);
      };

      castArgFn(op.getVariable());
      castArgFn(op.getValue());

      // Create the remaining blocks and map their arguments.
      for (auto& block : llvm::enumerate(op.getBodyRegion().getBlocks())) {
        if (block.index() != 0) {
          std::vector<mlir::Location> argLocations;
          std::vector<mlir::Type> argTypes;

          for (const auto& arg : block.value().getArguments()) {
            argLocations.push_back(arg.getLoc());
            argTypes.push_back(getTypeConverter()->convertType(arg.getType()));
          }

          mlir::Block* clonedBlock = rewriter.createBlock(
              &newOp.getFunctionBody(),
              newOp.getFunctionBody().end(),
              argTypes,
              argLocations);

          mapping.map(&block.value(), clonedBlock);

          for (const auto& [original, cloned]
               : llvm::zip(block.value().getArguments(),
                           clonedBlock->getArguments())) {
            mlir::Value mapped = cloned;

            if (mapped.getType() != cloned.getType()) {
              mapped = getTypeConverter()->materializeSourceConversion(
                  rewriter, original.getLoc(), original.getType(), mapped);
            }

            mapping.map(original, mapped);
          }
        }
      }

      // Clone the operations.
      for (auto& block : op.getFunctionBody().getBlocks()) {
        mlir::Block* clonedBlock = mapping.lookup(&block);
        rewriter.setInsertionPointToEnd(clonedBlock);

        for (auto& bodyOp : block.getOperations()) {
          rewriter.clone(bodyOp, mapping);
        }
      }

      return mlir::success();
    }
  };

  struct ResidualFunctionOpTypes
      : public mlir::OpConversionPattern<ResidualFunctionOp>
  {
    using mlir::OpConversionPattern<ResidualFunctionOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        ResidualFunctionOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Type timeType =
          getTypeConverter()->convertType(op.getTime().getType());

      llvm::SmallVector<mlir::Type> variablesTypes;

      for (auto variable : op.getVariables()) {
        variablesTypes.push_back(
            getTypeConverter()->convertType(variable.getType()));
      }

      mlir::Type differenceType =
          getTypeConverter()->convertType(op.getFunctionType().getResult(0));

      auto newOp = rewriter.replaceOpWithNewOp<ResidualFunctionOp>(
          op, op.getSymName(), timeType, variablesTypes,
          op.getEquationRank().getSExtValue(), differenceType);

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      mlir::BlockAndValueMapping mapping;

      // Map the entry block.
      mapping.map(&op.getFunctionBody().front(), entryBlock);

      for (const auto& [original, cloned]
           : llvm::zip(op.getFunctionBody().getArguments(),
                       newOp.getFunctionBody().getArguments())) {
        mapping.map(original, cloned);
      }

      auto castArgFn = [&](mlir::Value originalArg) {
        auto castedClonedArg = getTypeConverter()->materializeSourceConversion(
            rewriter, originalArg.getLoc(), originalArg.getType(),
            mapping.lookup(originalArg));

        mapping.map(originalArg, castedClonedArg);
      };

      castArgFn(op.getTime());

      for (auto variable : op.getVariables()) {
        castArgFn(variable);
      }

      // Create the remaining blocks and map their arguments.
      for (auto& block : llvm::enumerate(op.getBodyRegion().getBlocks())) {
        if (block.index() != 0) {
          std::vector<mlir::Location> argLocations;
          std::vector<mlir::Type> argTypes;

          for (const auto& arg : block.value().getArguments()) {
            argLocations.push_back(arg.getLoc());
            argTypes.push_back(getTypeConverter()->convertType(arg.getType()));
          }

          mlir::Block* clonedBlock = rewriter.createBlock(
              &newOp.getFunctionBody(),
              newOp.getFunctionBody().end(),
              argTypes,
              argLocations);

          mapping.map(&block.value(), clonedBlock);

          for (const auto& [original, cloned]
               : llvm::zip(block.value().getArguments(),
                           clonedBlock->getArguments())) {
            mlir::Value mapped = cloned;

            if (mapped.getType() != cloned.getType()) {
              mapped = getTypeConverter()->materializeSourceConversion(
                  rewriter, original.getLoc(), original.getType(), mapped);
            }

            mapping.map(original, mapped);
          }
        }
      }

      // Clone the operations.
      for (auto& block : op.getFunctionBody().getBlocks()) {
        mlir::Block* clonedBlock = mapping.lookup(&block);
        rewriter.setInsertionPointToEnd(clonedBlock);

        for (auto& bodyOp : block.getOperations()) {
          if (auto returnOp = mlir::dyn_cast<ReturnOp>(bodyOp)) {
            std::vector<mlir::Value> returnValues;

            for (auto returnValue : returnOp.operands()) {
              returnValues.push_back(
                  getTypeConverter()->materializeTargetConversion(
                      rewriter, returnOp.getLoc(),
                      getTypeConverter()->convertType(returnValue.getType()),
                      mapping.lookup(returnValue)));
            }

            rewriter.create<ReturnOp>(returnOp.getLoc(), returnValues);
          } else {
            rewriter.clone(bodyOp, mapping);
          }
        }
      }

      return mlir::success();
    }
  };

  struct JacobianFunctionOpTypes
      : public mlir::OpConversionPattern<JacobianFunctionOp>
  {
    using mlir::OpConversionPattern<JacobianFunctionOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        JacobianFunctionOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Type timeType =
          getTypeConverter()->convertType(op.getTime().getType());

      llvm::SmallVector<mlir::Type> variablesTypes;

      for (auto variable : op.getVariables()) {
        variablesTypes.push_back(getTypeConverter()->convertType(variable.getType()));
      }

      mlir::Type alphaType =
          getTypeConverter()->convertType(op.getAlpha().getType());

      mlir::Type resultType =
          getTypeConverter()->convertType(op.getFunctionType().getResult(0));

      auto newOp = rewriter.replaceOpWithNewOp<JacobianFunctionOp>(
          op, op.getSymName(), timeType, variablesTypes,
          op.getEquationRank().getSExtValue(),
          op.getVariableRank().getSExtValue(),
          alphaType, resultType);

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      mlir::BlockAndValueMapping mapping;

      // Map the entry block.
      mapping.map(&op.getFunctionBody().front(), entryBlock);

      for (const auto& [original, cloned]
           : llvm::zip(op.getFunctionBody().getArguments(),
                       newOp.getFunctionBody().getArguments())) {
        mapping.map(original, cloned);
      }

      auto castArgFn = [&](mlir::Value originalArg) {
        auto castedClonedArg = getTypeConverter()->materializeSourceConversion(
            rewriter, originalArg.getLoc(), originalArg.getType(),
            mapping.lookup(originalArg));

        mapping.map(originalArg, castedClonedArg);
      };

      castArgFn(op.getTime());

      for (auto variable : op.getVariables()) {
        castArgFn(variable);
      }

      castArgFn(op.getAlpha());

      // Create the remaining blocks and map their arguments.
      for (auto& block : llvm::enumerate(op.getBodyRegion().getBlocks())) {
        if (block.index() != 0) {
          std::vector<mlir::Location> argLocations;
          std::vector<mlir::Type> argTypes;

          for (const auto& arg : block.value().getArguments()) {
            argLocations.push_back(arg.getLoc());
            argTypes.push_back(getTypeConverter()->convertType(arg.getType()));
          }

          mlir::Block* clonedBlock = rewriter.createBlock(
              &newOp.getFunctionBody(),
              newOp.getFunctionBody().end(),
              argTypes,
              argLocations);

          mapping.map(&block.value(), clonedBlock);

          for (const auto& [original, cloned]
               : llvm::zip(block.value().getArguments(),
                           clonedBlock->getArguments())) {
            mlir::Value mapped = cloned;

            if (mapped.getType() != cloned.getType()) {
              mapped = getTypeConverter()->materializeSourceConversion(
                  rewriter, original.getLoc(), original.getType(), mapped);
            }

            mapping.map(original, mapped);
          }
        }
      }

      // Clone the operations.
      for (auto& block : op.getFunctionBody().getBlocks()) {
        mlir::Block* clonedBlock = mapping.lookup(&block);
        rewriter.setInsertionPointToEnd(clonedBlock);

        for (auto& bodyOp : block.getOperations()) {
          if (auto returnOp = mlir::dyn_cast<ReturnOp>(bodyOp)) {
            std::vector<mlir::Value> returnValues;

            for (auto returnValue : returnOp.operands()) {
              returnValues.push_back(
                  getTypeConverter()->materializeTargetConversion(
                      rewriter, returnOp.getLoc(),
                      getTypeConverter()->convertType(returnValue.getType()),
                      mapping.lookup(returnValue)));
            }

            rewriter.create<ReturnOp>(returnOp.getLoc(), returnValues);
          } else {
            rewriter.clone(bodyOp, mapping);
          }
        }
      }

      return mlir::success();
    }
  };
}

namespace mlir
{
  void populateIDAToFuncStructuralTypeConversionsAndLegality(
      mlir::LLVMTypeConverter& typeConverter,
      mlir::RewritePatternSet& patterns,
      mlir::ConversionTarget& target)
  {
    typeConverter.addConversion([&](InstanceType type) {
      return type;
    });

    typeConverter.addConversion([&](EquationType type) {
      return type;
    });

    typeConverter.addConversion([&](VariableType type) {
      return type;
    });

    patterns.add<
        VariableGetterOpTypes,
        VariableSetterOpTypes,
        ResidualFunctionOpTypes,
        JacobianFunctionOpTypes>(typeConverter, patterns.getContext());

    target.addDynamicallyLegalOp<VariableGetterOp>(
        [&](VariableGetterOp op) {
          if (!typeConverter.isLegal(op.getFunctionType().getResult(0))) {
            return false;
          }

          if (!typeConverter.isLegal(op.getVariable().getType())) {
            return false;
          }

          return true;
        });

    target.addDynamicallyLegalOp<VariableSetterOp>(
        [&](VariableSetterOp op) {
          if (!typeConverter.isLegal(op.getVariable().getType())) {
            return false;
          }

          if (!typeConverter.isLegal(op.getValue().getType())) {
            return false;
          }

          return true;
        });

    target.addDynamicallyLegalOp<ResidualFunctionOp>(
        [&](ResidualFunctionOp op) {
          if (!typeConverter.isLegal(op.getTime().getType())) {
            return false;
          }

          for (auto variable : op.getVariables()) {
            if (!typeConverter.isLegal(variable.getType())) {
              return false;
            }
          }

          if (!typeConverter.isLegal(op.getFunctionType().getResult(0))) {
            return false;
          }

          return true;
        });

    target.addDynamicallyLegalOp<JacobianFunctionOp>(
        [&](JacobianFunctionOp op) {
          if (!typeConverter.isLegal(op.getTime().getType())) {
            return false;
          }

          for (auto variable : op.getVariables()) {
            if (!typeConverter.isLegal(variable.getType())) {
              return false;
            }
          }

          if (!typeConverter.isLegal(op.getAlpha().getType())) {
            return false;
          }

          if (!typeConverter.isLegal(op.getFunctionType().getResult(0))) {
            return false;
          }

          return true;
        });
  }

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
