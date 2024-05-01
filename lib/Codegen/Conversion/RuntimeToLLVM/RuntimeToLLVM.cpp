#include "marco/Codegen/Conversion/RuntimeToLLVM/RuntimeToLLVM.h"
#include "marco/Dialect/Runtime/RuntimeDialect.h"
#include "marco/Codegen/Runtime.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

namespace mlir
{
#define GEN_PASS_DEF_RUNTIMETOLLVMCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
}

using namespace ::marco::codegen;
using namespace ::mlir::runtime;

namespace
{
  class RuntimeToLLVMConversionPass
      : public mlir::impl::RuntimeToLLVMConversionPassBase<
            RuntimeToLLVMConversionPass>
  {
    public:
      using RuntimeToLLVMConversionPassBase<RuntimeToLLVMConversionPass>
          ::RuntimeToLLVMConversionPassBase;

      void runOnOperation() override;

    private:
      mlir::LogicalResult convertOps();
  };
}

void RuntimeToLLVMConversionPass::runOnOperation()
{
  if (mlir::failed(convertOps())) {
    return signalPassFailure();
  }
}

namespace
{
  template<typename Op>
  class RuntimeOpConversion : public mlir::ConvertOpToLLVMPattern<Op>
  {
    public:
      RuntimeOpConversion(
          mlir::LLVMTypeConverter& typeConverter,
          mlir::SymbolTableCollection& symbolTableCollection)
          : mlir::ConvertOpToLLVMPattern<Op>(typeConverter),
            symbolTableCollection(&symbolTableCollection)
      {
      }

      mlir::Value getScheduler(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          llvm::StringRef name) const
      {
        auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

        mlir::Value address =
            builder.create<mlir::LLVM::AddressOfOp>(loc, ptrType, name);

        return builder.create<mlir::LLVM::LoadOp>(
            loc, this->getVoidPtrType(), address);
      }

      mlir::LLVM::GlobalOp declareRangesArray(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::ModuleOp moduleOp,
          const MultidimensionalRange& indices,
          llvm::StringRef prefix) const
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(moduleOp.getBody());

        auto arrayType = mlir::LLVM::LLVMArrayType::get(
            builder.getI64Type(), indices.rank() * 2);

        llvm::SmallVector<int64_t, 10> values;

        for (size_t dim = 0, rank = indices.rank(); dim < rank; ++dim) {
          values.push_back(indices[dim].getBegin());
          values.push_back(indices[dim].getEnd());
        }

        std::string symbolName = prefix.str() + "_range";

        auto tensorType = mlir::RankedTensorType::get(
            arrayType.getNumElements(), builder.getI64Type());

        auto globalOp = builder.create<mlir::LLVM::GlobalOp>(
            loc, arrayType, true, mlir::LLVM::Linkage::Private, symbolName,
            mlir::DenseIntElementsAttr::get(tensorType, values));

        symbolTableCollection->getSymbolTable(moduleOp).insert(globalOp);
        return globalOp;
      }

      mlir::Value declareAndGetRangesArray(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::ModuleOp moduleOp,
          std::optional<MultidimensionalRangeAttr> indices,
          llvm::StringRef instanceName) const
      {
        auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

        if (!indices) {
          return builder.create<mlir::LLVM::ZeroOp>(loc, ptrType);
        }

        auto globalOp = declareRangesArray(
            builder, loc, moduleOp, indices->getValue(), instanceName);

        mlir::Value address =
            builder.create<mlir::LLVM::AddressOfOp>(loc, globalOp);

        return address;
      }

      mlir::LLVM::LLVMFuncOp getOrDeclareFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::Location loc,
          llvm::StringRef name,
          mlir::LLVM::LLVMFunctionType functionType) const
      {
        auto funcOp =
            symbolTableCollection->lookupSymbolIn<mlir::LLVM::LLVMFuncOp>(
                moduleOp, builder.getStringAttr(name));

        if (funcOp) {
          return funcOp;
        }

        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(moduleOp.getBody());

        auto newFuncOp =
            builder.create<mlir::LLVM::LLVMFuncOp>(loc, name, functionType);

        symbolTableCollection->getSymbolTable(moduleOp).insert(newFuncOp);
        return newFuncOp;
      }

      mlir::LLVM::LLVMFuncOp getOrDeclareFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::Location loc,
          llvm::StringRef name,
          mlir::Type resultType,
          llvm::ArrayRef<mlir::Type> argTypes) const
      {
        auto functionType =
            mlir::LLVM::LLVMFunctionType::get(resultType, argTypes);

        return getOrDeclareFunction(builder, moduleOp, loc, name, functionType);
      }

      mlir::LLVM::LLVMFuncOp getOrDeclareFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp moduleOp,
          mlir::Location loc,
          llvm::StringRef name,
          mlir::Type resultType,
          llvm::ArrayRef<mlir::Value> args) const
      {
        llvm::SmallVector<mlir::Type> argTypes;

        for (mlir::Value arg : args) {
          argTypes.push_back(arg.getType());
        }

        return getOrDeclareFunction(
            builder, moduleOp, loc, name, resultType, argTypes);
      }

      mlir::Value getFunctionAddress(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::LLVM::LLVMFunctionType functionType,
          llvm::StringRef name) const
      {
        auto functionPtrType =
            mlir::LLVM::LLVMPointerType::get(builder.getContext());

        mlir::Value address = builder.create<mlir::LLVM::AddressOfOp>(
            loc, functionPtrType, name);

        return address;
      }

      mlir::Value getEquationFunctionAddress(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          llvm::StringRef name) const
      {
        mlir::Type resultType =
            mlir::LLVM::LLVMVoidType::get(builder.getContext());

        llvm::SmallVector<mlir::Type, 1> argTypes;

        argTypes.push_back(
            mlir::LLVM::LLVMPointerType::get(builder.getContext()));

        auto functionType =
            mlir::LLVM::LLVMFunctionType::get(resultType, argTypes);

        return getFunctionAddress(builder, loc, functionType, name);
      }

    protected:
      mlir::SymbolTableCollection* symbolTableCollection;
  };

  class SchedulerOpLowering : public RuntimeOpConversion<SchedulerOp>
  {
    public:
      using RuntimeOpConversion<SchedulerOp>::RuntimeOpConversion;

      mlir::LogicalResult matchAndRewrite(
          SchedulerOp op,
          OpAdaptor adaptor,
          mlir::ConversionPatternRewriter& rewriter) const override
      {
        auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

        mlir::SymbolTable& symbolTable =
            symbolTableCollection->getSymbolTable(moduleOp);

        // Create the global variable.
        rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
            op, getVoidPtrType(), false, mlir::LLVM::Linkage::Private,
            op.getSymName(), nullptr);

        symbolTable.remove(op);
        return mlir::success();
      }
  };

  struct SchedulerCreateOpLowering
      : public RuntimeOpConversion<SchedulerCreateOp>
  {
    using RuntimeOpConversion<SchedulerCreateOp>::RuntimeOpConversion;

    mlir::LogicalResult matchAndRewrite(
        SchedulerCreateOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 1> args;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      auto resultType = getVoidPtrType();
      auto mangledResultType = mangling.getVoidPointerType();

      auto functionName = mangling.getMangledFunction(
          "schedulerCreate", mangledResultType, mangledArgsTypes);

      auto funcOp = getOrDeclareFunction(
          rewriter, moduleOp, loc, functionName, resultType, args);

      auto callOp = rewriter.create<mlir::LLVM::CallOp>(loc, funcOp, args);

      auto instancePtrType =
          mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

      mlir::Value address = rewriter.create<mlir::LLVM::AddressOfOp>(
          loc, instancePtrType, op.getScheduler());

      rewriter.create<mlir::LLVM::StoreOp>(loc, callOp.getResult(), address);

      rewriter.eraseOp(op);
      return mlir::success();
    }
  };

  struct SchedulerDestroyOpLowering
      : public RuntimeOpConversion<SchedulerDestroyOp>
  {
    using RuntimeOpConversion<SchedulerDestroyOp>::RuntimeOpConversion;

    mlir::LogicalResult matchAndRewrite(
        SchedulerDestroyOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 1> args;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Scheduler instance.
      args.push_back(getScheduler(rewriter, loc, adaptor.getScheduler()));
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library.
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();

      auto functionName = mangling.getMangledFunction(
          "schedulerDestroy", mangledResultType, mangledArgsTypes);

      auto funcOp = getOrDeclareFunction(
          rewriter, moduleOp, loc, functionName, resultType, args);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, funcOp, args);
      return mlir::success();
    }
  };

  struct SchedulerAddEquationOpLowering
      : public RuntimeOpConversion<SchedulerAddEquationOp>
  {
    using RuntimeOpConversion<SchedulerAddEquationOp>
        ::RuntimeOpConversion;

    mlir::LogicalResult matchAndRewrite(
        SchedulerAddEquationOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 5> args;
      llvm::SmallVector<std::string, 5> mangledArgsTypes;

      // Scheduler instance.
      args.push_back(getScheduler(rewriter, loc, adaptor.getScheduler()));
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Equation function.
      args.push_back(getEquationFunctionAddress(
          rewriter, loc, op.getFunction()));

      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Equation rank.
      int64_t rank = 0;

      if (auto ranges = op.getRanges()) {
        rank = ranges->getValue().rank();
      }

      mlir::Value rankValue = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(rank));

      args.push_back(rankValue);

      mangledArgsTypes.push_back(mangling.getIntegerType(
          rankValue.getType().getIntOrFloatBitWidth()));

      // Equation ranges.
      mlir::Value equationRangesPtr = declareAndGetRangesArray(
          rewriter, loc, moduleOp, adaptor.getRanges(),
          adaptor.getScheduler());

      args.push_back(equationRangesPtr);

      mangledArgsTypes.push_back(
          mangling.getPointerType(mangling.getIntegerType(64)));

      // Independent indices property.
      mlir::Value independentIndices =
          rewriter.create<mlir::arith::ConstantOp>(
              loc, rewriter.getBoolAttr(op.getIndependentIndices()));

      args.push_back(independentIndices);
      mangledArgsTypes.push_back(mangling.getIntegerType(1));

      // Create the call to the runtime library.
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();

      auto functionName = mangling.getMangledFunction(
          "schedulerAddEquation", mangledResultType, mangledArgsTypes);

      auto funcOp = getOrDeclareFunction(
          rewriter, moduleOp, loc, functionName, resultType, args);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, funcOp, args);
      return mlir::success();
    }
  };

  struct SchedulerRunOpLowering : public RuntimeOpConversion<SchedulerRunOp>
  {
    using RuntimeOpConversion<SchedulerRunOp>::RuntimeOpConversion;

    mlir::LogicalResult matchAndRewrite(
        SchedulerRunOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 1> args;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Scheduler instance.
      args.push_back(getScheduler(rewriter, loc, adaptor.getScheduler()));
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library.
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();

      auto functionName = mangling.getMangledFunction(
          "schedulerRun", mangledResultType, mangledArgsTypes);

      auto funcOp = getOrDeclareFunction(
          rewriter, moduleOp, loc, functionName, resultType, args);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, funcOp, args);
      return mlir::success();
    }
  };
}

mlir::LogicalResult RuntimeToLLVMConversionPass::convertOps()
{
  mlir::ModuleOp moduleOp = getOperation();
  mlir::ConversionTarget target(getContext());

  target.addIllegalDialect<mlir::runtime::RuntimeDialect>();
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();

  mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
  llvmLoweringOptions.dataLayout.reset(dataLayout);

  mlir::LLVMTypeConverter typeConverter(&getContext(), llvmLoweringOptions);
  mlir::RewritePatternSet patterns(&getContext());
  mlir::SymbolTableCollection symbolTableCollection;

  patterns.insert<
      SchedulerOpLowering,
      SchedulerCreateOpLowering,
      SchedulerDestroyOpLowering,
      SchedulerAddEquationOpLowering,
      SchedulerRunOpLowering>(typeConverter, symbolTableCollection);

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

namespace mlir
{
  std::unique_ptr<mlir::Pass> createRuntimeToLLVMConversionPass()
  {
    return std::make_unique<RuntimeToLLVMConversionPass>();
  }

  std::unique_ptr<mlir::Pass> createRuntimeToLLVMConversionPass(
      const RuntimeToLLVMConversionPassOptions& options)
  {
    return std::make_unique<RuntimeToLLVMConversionPass>(options);
  }
}
