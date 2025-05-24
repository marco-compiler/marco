#include "marco/Codegen/Conversion/RuntimeToLLVM/RuntimeToLLVM.h"
#include "marco/Codegen/Conversion/RuntimeToLLVM/LLVMTypeConverter.h"
#include "marco/Codegen/Runtime.h"
#include "marco/Dialect/Runtime/IR/Runtime.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
#define GEN_PASS_DEF_RUNTIMETOLLVMCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
} // namespace mlir

using namespace ::marco::codegen;
using namespace ::mlir::runtime;

namespace {
class RuntimeToLLVMConversionPass
    : public mlir::impl::RuntimeToLLVMConversionPassBase<
          RuntimeToLLVMConversionPass> {
public:
  using RuntimeToLLVMConversionPassBase<
      RuntimeToLLVMConversionPass>::RuntimeToLLVMConversionPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult convertOps();
};
} // namespace

void RuntimeToLLVMConversionPass::runOnOperation() {
  if (mlir::failed(convertOps())) {
    return signalPassFailure();
  }
}

namespace {
template <typename Op>
class RuntimeOpConversion : public mlir::ConvertOpToLLVMPattern<Op> {
public:
  RuntimeOpConversion(mlir::LLVMTypeConverter &typeConverter,
                      mlir::SymbolTableCollection &symbolTableCollection)
      : mlir::ConvertOpToLLVMPattern<Op>(typeConverter),
        symbolTableCollection(&symbolTableCollection) {}

  mlir::Value getScheduler(mlir::OpBuilder &builder, mlir::Location loc,
                           llvm::StringRef name) const {
    auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

    mlir::Value address =
        builder.create<mlir::LLVM::AddressOfOp>(loc, ptrType, name);

    return builder.create<mlir::LLVM::LoadOp>(loc, this->getVoidPtrType(),
                                              address);
  }

  mlir::LLVM::GlobalOp declareRangesArray(mlir::OpBuilder &builder,
                                          mlir::Location loc,
                                          mlir::ModuleOp moduleOp,
                                          const MultidimensionalRange &indices,
                                          llvm::StringRef prefix) const {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());

    auto arrayType = mlir::LLVM::LLVMArrayType::get(builder.getI64Type(),
                                                    indices.rank() * 2);

    llvm::SmallVector<int64_t, 10> values;

    for (size_t dim = 0, rank = indices.rank(); dim < rank; ++dim) {
      values.push_back(indices[dim].getBegin());
      values.push_back(indices[dim].getEnd());
    }

    std::string symbolName = prefix.str() + "_range";

    auto tensorType = mlir::RankedTensorType::get(arrayType.getNumElements(),
                                                  builder.getI64Type());

    auto globalOp = builder.create<mlir::LLVM::GlobalOp>(
        loc, arrayType, true, mlir::LLVM::Linkage::Private, symbolName,
        mlir::DenseIntElementsAttr::get(tensorType, values));

    symbolTableCollection->getSymbolTable(moduleOp).insert(globalOp);
    return globalOp;
  }

  mlir::Value
  declareAndGetRangesArray(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::ModuleOp moduleOp,
                           std::optional<MultidimensionalRange> indices,
                           llvm::StringRef instanceName) const {
    auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

    if (!indices) {
      return builder.create<mlir::LLVM::ZeroOp>(loc, ptrType);
    }

    auto globalOp =
        declareRangesArray(builder, loc, moduleOp, *indices, instanceName);

    mlir::Value address =
        builder.create<mlir::LLVM::AddressOfOp>(loc, globalOp);

    return address;
  }

  mlir::LLVM::LLVMFuncOp
  getOrDeclareFunction(mlir::OpBuilder &builder, mlir::ModuleOp moduleOp,
                       mlir::Location loc, llvm::StringRef name,
                       mlir::LLVM::LLVMFunctionType functionType) const {
    auto funcOp = symbolTableCollection->lookupSymbolIn<mlir::LLVM::LLVMFuncOp>(
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

  mlir::LLVM::LLVMFuncOp
  getOrDeclareFunction(mlir::OpBuilder &builder, mlir::ModuleOp moduleOp,
                       mlir::Location loc, llvm::StringRef name,
                       mlir::Type resultType,
                       llvm::ArrayRef<mlir::Type> argTypes) const {
    auto functionType = mlir::LLVM::LLVMFunctionType::get(resultType, argTypes);

    return getOrDeclareFunction(builder, moduleOp, loc, name, functionType);
  }

  mlir::LLVM::LLVMFuncOp
  getOrDeclareFunction(mlir::OpBuilder &builder, mlir::ModuleOp moduleOp,
                       mlir::Location loc, llvm::StringRef name,
                       mlir::Type resultType,
                       llvm::ArrayRef<mlir::Value> args) const {
    llvm::SmallVector<mlir::Type> argTypes;

    for (mlir::Value arg : args) {
      argTypes.push_back(arg.getType());
    }

    return getOrDeclareFunction(builder, moduleOp, loc, name, resultType,
                                argTypes);
  }

  mlir::Value getFunctionAddress(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::LLVM::LLVMFunctionType functionType,
                                 llvm::StringRef name) const {
    auto functionPtrType =
        mlir::LLVM::LLVMPointerType::get(builder.getContext());

    mlir::Value address =
        builder.create<mlir::LLVM::AddressOfOp>(loc, functionPtrType, name);

    return address;
  }

  mlir::Value getEquationFunctionAddress(mlir::OpBuilder &builder,
                                         mlir::Location loc,
                                         llvm::StringRef name) const {
    mlir::Type resultType = mlir::LLVM::LLVMVoidType::get(builder.getContext());

    llvm::SmallVector<mlir::Type, 1> argTypes;

    argTypes.push_back(mlir::LLVM::LLVMPointerType::get(builder.getContext()));

    auto functionType = mlir::LLVM::LLVMFunctionType::get(resultType, argTypes);

    return getFunctionAddress(builder, loc, functionType, name);
  }

protected:
  mlir::SymbolTableCollection *symbolTableCollection;
};

class SchedulerOpLowering : public RuntimeOpConversion<SchedulerOp> {
public:
  using RuntimeOpConversion<SchedulerOp>::RuntimeOpConversion;

  mlir::LogicalResult
  matchAndRewrite(SchedulerOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

    mlir::SymbolTable &symbolTable =
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
    : public RuntimeOpConversion<SchedulerCreateOp> {
  using RuntimeOpConversion<SchedulerCreateOp>::RuntimeOpConversion;

  mlir::LogicalResult
  matchAndRewrite(SchedulerCreateOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    RuntimeFunctionsMangling mangling;

    llvm::SmallVector<mlir::Value, 1> args;
    llvm::SmallVector<std::string, 1> mangledArgsTypes;

    auto resultType = getVoidPtrType();
    auto mangledResultType = mangling.getVoidPointerType();

    auto functionName = mangling.getMangledFunction(
        "schedulerCreate", mangledResultType, mangledArgsTypes);

    auto funcOp = getOrDeclareFunction(rewriter, moduleOp, loc, functionName,
                                       resultType, args);

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
    : public RuntimeOpConversion<SchedulerDestroyOp> {
  using RuntimeOpConversion<SchedulerDestroyOp>::RuntimeOpConversion;

  mlir::LogicalResult
  matchAndRewrite(SchedulerDestroyOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
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

    auto funcOp = getOrDeclareFunction(rewriter, moduleOp, loc, functionName,
                                       resultType, args);

    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, funcOp, args);
    return mlir::success();
  }
};

struct SchedulerAddEquationOpLowering
    : public RuntimeOpConversion<SchedulerAddEquationOp> {
  using RuntimeOpConversion<SchedulerAddEquationOp>::RuntimeOpConversion;

  mlir::LogicalResult
  matchAndRewrite(SchedulerAddEquationOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    RuntimeFunctionsMangling mangling;

    llvm::SmallVector<mlir::Value, 5> args;
    llvm::SmallVector<std::string, 5> mangledArgsTypes;

    // Scheduler instance.
    args.push_back(getScheduler(rewriter, loc, adaptor.getScheduler()));
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // Equation function.
    args.push_back(getEquationFunctionAddress(rewriter, loc, op.getFunction()));

    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // Equation rank.
    const IndexSet &indices = op.getProperties().indices;

    mlir::Value rankValue = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(indices.rank()));

    args.push_back(rankValue);

    mangledArgsTypes.push_back(
        mangling.getIntegerType(rankValue.getType().getIntOrFloatBitWidth()));

    // Equation ranges.
    mlir::Value &equationRangesPtrArg = args.emplace_back();

    mangledArgsTypes.push_back(
        mangling.getPointerType(mangling.getIntegerType(64)));

    // Independent indices property.
    mlir::Value independentIndices = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, rewriter.getBoolAttr(op.getIndependentIndices()));

    args.push_back(independentIndices);
    mangledArgsTypes.push_back(mangling.getIntegerType(1));

    // Create the call to the runtime library.
    auto resultType = getVoidType();
    auto mangledResultType = mangling.getVoidType();

    auto functionName = mangling.getMangledFunction(
        "schedulerAddEquation", mangledResultType, mangledArgsTypes);

    auto funcOp = getOrDeclareFunction(rewriter, moduleOp, loc, functionName,
                                       resultType, args);

    if (indices.empty()) {
      equationRangesPtrArg = declareAndGetRangesArray(
          rewriter, loc, moduleOp, std::nullopt, adaptor.getScheduler());

      rewriter.create<mlir::LLVM::CallOp>(loc, funcOp, args);
    } else {
      for (const MultidimensionalRange &range :
           llvm::make_range(indices.rangesBegin(), indices.rangesEnd())) {
        equationRangesPtrArg = declareAndGetRangesArray(
            rewriter, loc, moduleOp, range, adaptor.getScheduler());

        rewriter.create<mlir::LLVM::CallOp>(loc, funcOp, args);
      }
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct SchedulerRunOpLowering : public RuntimeOpConversion<SchedulerRunOp> {
  using RuntimeOpConversion<SchedulerRunOp>::RuntimeOpConversion;

  mlir::LogicalResult
  matchAndRewrite(SchedulerRunOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
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

    auto funcOp = getOrDeclareFunction(rewriter, moduleOp, loc, functionName,
                                       resultType, args);

    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, funcOp, args);
    return mlir::success();
  }
};

class FunctionOpLowering : public RuntimeOpConversion<FunctionOp> {
public:
  using RuntimeOpConversion<FunctionOp>::RuntimeOpConversion;

  mlir::LogicalResult
  matchAndRewrite(FunctionOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!op.isDeclaration()) {
      return rewriter.notifyMatchFailure(op, "Not a declaration");
    }

    llvm::SmallVector<mlir::Type> argTypes;

    mlir::Type resultType =
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext());

    assert(op.getResultTypes().size() <= 1);

    if (!op.getResultTypes().empty()) {
      resultType = getTypeConverter()->convertType(op.getResultTypes()[0]);
    }

    for (mlir::Type argType : op.getArgumentTypes()) {
      if (mlir::isa<mlir::MemRefType, mlir::UnrankedMemRefType>(argType)) {
        argTypes.push_back(
            mlir::LLVM::LLVMPointerType::get(rewriter.getContext()));
      } else {
        argTypes.push_back(getTypeConverter()->convertType(argType));
      }
    }

    mlir::SymbolTable &symbolTable = symbolTableCollection->getSymbolTable(
        op->getParentOfType<mlir::ModuleOp>());

    symbolTable.remove(op);
    symbolTableCollection->invalidateSymbolTable(op);

    auto llvmFuncOp = rewriter.replaceOpWithNewOp<mlir::LLVM::LLVMFuncOp>(
        op, op.getSymName(),
        mlir::LLVM::LLVMFunctionType::get(resultType, argTypes));

    symbolTable.insert(llvmFuncOp);
    return mlir::success();
  }
};

class CallOpLowering : public mlir::ConvertOpToLLVMPattern<CallOp> {
public:
  using mlir::ConvertOpToLLVMPattern<CallOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    // Determine the result type.
    mlir::Type resultType =
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext());

    assert(op.getResultTypes().size() <= 1);

    if (!op.getResultTypes().empty()) {
      resultType = getTypeConverter()->convertType(op.getResultTypes()[0]);
    }

    // Save stack position before promoting the memref descriptors.
    auto stackSaveOp =
        rewriter.create<mlir::LLVM::StackSaveOp>(loc, getVoidPtrType());

    // Determine the arguments.
    llvm::SmallVector<mlir::Value> promotedArgs;

    for (auto [originalArg, adaptorArg] :
         llvm::zip(op.getArgs(), adaptor.getArgs())) {
      mlir::Type originalArgType = originalArg.getType();

      if (!mlir::isa<mlir::MemRefType, mlir::UnrankedMemRefType>(
              originalArgType)) {
        promotedArgs.push_back(adaptorArg);
        continue;
      }

      // The argument is a memref (ranked or unranked).
      mlir::Value memRef = adaptorArg;

      if (auto memRefType =
              mlir::dyn_cast<mlir::MemRefType>(originalArg.getType())) {
        // Promote the ranked memrefs to unranked ones.
        memRef = makeUnranked(rewriter, loc, memRef, memRefType);
      }

      // Promote the unranked descriptors to the stack.
      auto one = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, getIndexType(), rewriter.getIndexAttr(1));

      auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

      auto allocated = rewriter.create<mlir::LLVM::AllocaOp>(
          loc, ptrType, memRef.getType(), one);

      rewriter.create<mlir::LLVM::StoreOp>(loc, memRef, allocated);
      promotedArgs.push_back(allocated);
    }

    // Replace the call.
    llvm::SmallVector<mlir::Type> promotedArgTypes;

    for (mlir::Value promotedArg : promotedArgs) {
      promotedArgTypes.push_back(promotedArg.getType());
    }

    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, mlir::LLVM::LLVMFunctionType::get(resultType, promotedArgTypes),
        op.getCallee(), promotedArgs);

    // Restore the stack.
    rewriter.create<mlir::LLVM::StackRestoreOp>(loc, stackSaveOp);

    return mlir::success();
  }

  mlir::Value makeUnranked(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value ranked, mlir::MemRefType type) const {
    auto rank = builder.create<mlir::LLVM::ConstantOp>(loc, getIndexType(),
                                                       type.getRank());

    mlir::Value ptr =
        getTypeConverter()->promoteOneMemRefDescriptor(loc, ranked, builder);

    auto unrankedType = mlir::UnrankedMemRefType::get(type.getElementType(),
                                                      type.getMemorySpace());

    return mlir::UnrankedMemRefDescriptor::pack(
        builder, loc, *getTypeConverter(), unrankedType,
        mlir::ValueRange{rank, ptr});
  }
};
} // namespace

namespace {
class StringOpLowering : public RuntimeOpConversion<StringOp> {
  using RuntimeOpConversion<StringOp>::RuntimeOpConversion;

  mlir::LogicalResult
  matchAndRewrite(StringOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

    mlir::Value constantString = createGlobalString(
        rewriter, loc, moduleOp, "runtimeStr", op.getString());

    rewriter.replaceOp(op, constantString);
    return mlir::success();
  }

  mlir::Value createGlobalString(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::ModuleOp moduleOp, mlir::StringRef name,
                                 mlir::StringRef value) const {
    mlir::LLVM::GlobalOp global;

    {
      // Create the global at the entry of the module.
      mlir::OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(moduleOp.getBody());

      auto type = mlir::LLVM::LLVMArrayType::get(
          mlir::IntegerType::get(builder.getContext(), 8), value.size() + 1);

      global = builder.create<mlir::LLVM::GlobalOp>(
          loc, type, true, mlir::LLVM::Linkage::Internal, name,
          builder.getStringAttr(
              llvm::StringRef(value.data(), value.size() + 1)));

      symbolTableCollection->getSymbolTable(moduleOp).insert(global);
    }

    // Get the pointer to the first character of the global string.
    mlir::Value globalPtr =
        builder.create<mlir::LLVM::AddressOfOp>(loc, global);

    mlir::Type type = mlir::LLVM::LLVMArrayType::get(
        mlir::IntegerType::get(builder.getContext(), 8), value.size() + 1);

    return builder.create<mlir::LLVM::GEPOp>(
        loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()), type,
        globalPtr, llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0});
  }
};
} // namespace

mlir::LogicalResult RuntimeToLLVMConversionPass::convertOps() {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::ConversionTarget target(getContext());

  target.addIllegalOp<SchedulerOp, SchedulerCreateOp, SchedulerDestroyOp,
                      SchedulerAddEquationOp, SchedulerRunOp>();

  target.addDynamicallyLegalOp<FunctionOp>(
      [](FunctionOp op) { return !op.isDeclaration(); });

  target.addIllegalOp<FunctionOp, CallOp>();

  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<mlir::memref::MemRefDialect>();

  mlir::DataLayout dataLayout(moduleOp);
  mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext(), dataLayout);

  mlir::runtime::LLVMTypeConverter typeConverter(&getContext(),
                                                 llvmLoweringOptions);

  mlir::RewritePatternSet patterns(&getContext());
  mlir::SymbolTableCollection symbolTableCollection;

  populateRuntimeToLLVMPatterns(patterns, typeConverter, symbolTableCollection);

  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

namespace mlir {
void populateRuntimeToLLVMPatterns(
    mlir::RewritePatternSet &patterns, mlir::LLVMTypeConverter &typeConverter,
    mlir::SymbolTableCollection &symbolTableCollection) {
  patterns.insert<SchedulerOpLowering, SchedulerCreateOpLowering,
                  SchedulerDestroyOpLowering, SchedulerAddEquationOpLowering,
                  SchedulerRunOpLowering>(typeConverter, symbolTableCollection);

  patterns.insert<FunctionOpLowering>(typeConverter, symbolTableCollection);
  patterns.insert<CallOpLowering>(typeConverter);

  patterns.insert<StringOpLowering>(typeConverter, symbolTableCollection);
}

std::unique_ptr<mlir::Pass> createRuntimeToLLVMConversionPass() {
  return std::make_unique<RuntimeToLLVMConversionPass>();
}
} // namespace mlir
