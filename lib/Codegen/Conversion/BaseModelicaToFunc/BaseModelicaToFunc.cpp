#include "marco/Codegen/Conversion/BaseModelicaToFunc/BaseModelicaToFunc.h"
#include "marco/Codegen/Conversion/BaseModelicaCommon/TypeConverter.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_BASEMODELICATOFUNCCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

#define GEN_PASS_DEF_BASEMODELICARAWVARIABLESCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
} // namespace mlir

using namespace ::mlir::bmodelica;

namespace {
/// Generic rewrite pattern that provides some utility functions.
template <typename Op>
class ModelicaOpRewritePattern : public mlir::OpRewritePattern<Op> {
  using mlir::OpRewritePattern<Op>::OpRewritePattern;
};

/// Generic conversion pattern that provides some utility functions.
template <typename Op>
class ModelicaOpConversionPattern : public mlir::OpConversionPattern<Op> {
public:
  using mlir::OpConversionPattern<Op>::OpConversionPattern;
};

template <typename Op>
class ClassInterfaceLowering : public ModelicaOpRewritePattern<Op> {
public:
  using ModelicaOpRewritePattern<Op>::ModelicaOpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class FunctionOpLowering : public ClassInterfaceLowering<FunctionOp> {
public:
  using ClassInterfaceLowering<FunctionOp>::ClassInterfaceLowering;
};

class ModelOpLowering : public ClassInterfaceLowering<ModelOp> {
public:
  using ClassInterfaceLowering<ModelOp>::ClassInterfaceLowering;
};

class PackageOpLowering : public ClassInterfaceLowering<PackageOp> {
public:
  using ClassInterfaceLowering<PackageOp>::ClassInterfaceLowering;
};

class RecordOpLowering : public ClassInterfaceLowering<RecordOp> {
public:
  using ClassInterfaceLowering<RecordOp>::ClassInterfaceLowering;
};
} // namespace

//===---------------------------------------------------------------------===//
// Func operations
//===---------------------------------------------------------------------===//

namespace {
template <typename Op>
class FunctionLoweringPattern : public ModelicaOpConversionPattern<Op> {
  mlir::SymbolTableCollection *symbolTableCollection;

public:
  FunctionLoweringPattern(const mlir::TypeConverter &typeConverter,
                          mlir::MLIRContext *context,
                          mlir::SymbolTableCollection &symbolTableCollection)
      : ModelicaOpConversionPattern<Op>(typeConverter, context),
        symbolTableCollection(&symbolTableCollection) {}

  mlir::SymbolTableCollection &getSymbolTableCollection() const {
    assert(symbolTableCollection);
    return *symbolTableCollection;
  }
};

struct EquationFunctionOpLowering
    : public FunctionLoweringPattern<EquationFunctionOp> {
  using FunctionLoweringPattern<EquationFunctionOp>::FunctionLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(EquationFunctionOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto originalFunctionType = op.getFunctionType();
    llvm::SmallVector<mlir::Type> argsTypes;
    llvm::SmallVector<mlir::Type> resultsTypes;

    for (mlir::Type argType : originalFunctionType.getInputs()) {
      argsTypes.push_back(getTypeConverter()->convertType(argType));
    }

    for (mlir::Type resultType : originalFunctionType.getResults()) {
      resultsTypes.push_back(getTypeConverter()->convertType(resultType));
    }

    auto functionType = rewriter.getFunctionType(argsTypes, resultsTypes);

    mlir::SymbolTable &symbolTable = getSymbolTableCollection().getSymbolTable(
        op->getParentOfType<mlir::ModuleOp>());

    auto funcOp = rewriter.create<mlir::func::FuncOp>(
        op.getLoc(), op.getSymName(), functionType);

    symbolTable.remove(op);
    symbolTable.insert(funcOp);

    rewriter.inlineRegionBefore(op.getBody(), funcOp.getFunctionBody(),
                                funcOp.end());

    if (mlir::failed(rewriter.convertRegionTypes(&funcOp.getFunctionBody(),
                                                 *typeConverter))) {
      return mlir::failure();
    }

    auto yieldOp = mlir::cast<YieldOp>(funcOp.getBody().back().getTerminator());

    rewriter.setInsertionPoint(yieldOp);
    llvm::SmallVector<mlir::Value> mappedResults;

    for (mlir::Value result : yieldOp.getValues()) {
      mappedResults.push_back(getTypeConverter()->materializeTargetConversion(
          rewriter, result.getLoc(),
          getTypeConverter()->convertType(result.getType()), result));
    }

    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(yieldOp, mappedResults);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct EquationCallOpLowering
    : public ModelicaOpConversionPattern<EquationCallOp> {
  using ModelicaOpConversionPattern<
      EquationCallOp>::ModelicaOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(EquationCallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    const IndexSet &indices = op.getIndices();

    if (indices.empty()) {
      rewriter.create<mlir::func::CallOp>(loc, op.getCallee(), std::nullopt,
                                          std::nullopt);
    } else {
      for (const MultidimensionalRange &range :
           llvm::make_range(indices.rangesBegin(), indices.rangesEnd())) {
        llvm::SmallVector<mlir::Value, 10> boundaries;

        for (size_t i = 0, e = range.rank(); i < e; ++i) {
          boundaries.push_back(rewriter.create<mlir::arith::ConstantOp>(
              op.getLoc(), rewriter.getIndexAttr(range[i].getBegin())));

          boundaries.push_back(rewriter.create<mlir::arith::ConstantOp>(
              op.getLoc(), rewriter.getIndexAttr(range[i].getEnd())));
        }

        rewriter.create<mlir::func::CallOp>(loc, op.getCallee(), std::nullopt,
                                            boundaries);
      }
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct RawFunctionOpLowering : public FunctionLoweringPattern<RawFunctionOp> {
  using FunctionLoweringPattern<RawFunctionOp>::FunctionLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(RawFunctionOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto originalFunctionType = op.getFunctionType();
    llvm::SmallVector<mlir::Type> argsTypes;
    llvm::SmallVector<mlir::Type> resultsTypes;

    for (mlir::Type argType : originalFunctionType.getInputs()) {
      argsTypes.push_back(getTypeConverter()->convertType(argType));
    }

    for (mlir::Type resultType : originalFunctionType.getResults()) {
      resultsTypes.push_back(getTypeConverter()->convertType(resultType));
    }

    auto functionType = rewriter.getFunctionType(argsTypes, resultsTypes);

    mlir::SymbolTable &symbolTable = getSymbolTableCollection().getSymbolTable(
        op->getParentOfType<mlir::ModuleOp>());

    auto funcOp = rewriter.create<mlir::func::FuncOp>(
        op.getLoc(), op.getSymName(), functionType);

    symbolTable.remove(op);
    symbolTable.insert(funcOp);

    rewriter.inlineRegionBefore(op.getBody(), funcOp.getFunctionBody(),
                                funcOp.end());

    if (mlir::failed(rewriter.convertRegionTypes(&funcOp.getFunctionBody(),
                                                 *typeConverter))) {
      return mlir::failure();
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct RawReturnOpLowering : public mlir::OpConversionPattern<RawReturnOp> {
  using mlir::OpConversionPattern<RawReturnOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(RawReturnOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op,
                                                      adaptor.getOperands());

    return mlir::success();
  }
};

struct RawVariableOpTypePattern
    : public mlir::OpConversionPattern<RawVariableOp> {
  using mlir::OpConversionPattern<RawVariableOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(RawVariableOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type convertedType =
        getTypeConverter()->convertType(op.getVariable().getType());

    if (convertedType == op.getVariable().getType()) {
      return rewriter.notifyMatchFailure(op, "Type already legal");
    }

    rewriter.replaceOpWithNewOp<RawVariableOp>(
        op, convertedType, op.getName(), op.getDimensionsConstraints(),
        op.getDynamicSizes(), op.getOutput());

    return mlir::success();
  }
};

struct RawVariableGetOpTypePattern
    : public mlir::OpConversionPattern<RawVariableGetOp> {
  using mlir::OpConversionPattern<RawVariableGetOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(RawVariableGetOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type convertedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (convertedResultType == op.getResult().getType()) {
      return rewriter.notifyMatchFailure(op, "Type already legal");
    }

    rewriter.replaceOpWithNewOp<RawVariableGetOp>(op, convertedResultType,
                                                  adaptor.getVariable());

    return mlir::success();
  }
};

struct RawVariableSetOpTypePattern
    : public mlir::OpConversionPattern<RawVariableSetOp> {
  using mlir::OpConversionPattern<RawVariableSetOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(RawVariableSetOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type convertedValueType =
        getTypeConverter()->convertType(op.getValue().getType());

    if (convertedValueType == op.getValue().getType()) {
      return rewriter.notifyMatchFailure(op, "Type already legal");
    }

    rewriter.replaceOpWithNewOp<RawVariableSetOp>(op, adaptor.getVariable(),
                                                  adaptor.getValue());

    return mlir::success();
  }
};

struct CallOpLowering : public mlir::OpConversionPattern<CallOp> {
  using mlir::OpConversionPattern<CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type, 3> resultsTypes;

    if (mlir::failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultsTypes))) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, op.getCallee(), resultsTypes, adaptor.getOperands());

    return mlir::success();
  }
};
} // namespace

namespace {
class BaseModelicaToFuncConversionPass
    : public mlir::impl::BaseModelicaToFuncConversionPassBase<
          BaseModelicaToFuncConversionPass> {
public:
  using BaseModelicaToFuncConversionPassBase<
      BaseModelicaToFuncConversionPass>::BaseModelicaToFuncConversionPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult eraseObjectOrientation();

  mlir::LogicalResult convertRawFunctions();
};
} // namespace

void BaseModelicaToFuncConversionPass::runOnOperation() {
  if (mlir::failed(eraseObjectOrientation())) {
    mlir::emitError(getOperation().getLoc(),
                    "Error in erasing object-orientation");
    return signalPassFailure();
  }

  if (mlir::failed(convertRawFunctions())) {
    mlir::emitError(getOperation().getLoc(),
                    "Error in converting the Modelica raw functions");
    return signalPassFailure();
  }
}

mlir::LogicalResult BaseModelicaToFuncConversionPass::eraseObjectOrientation() {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::ConversionTarget target(getContext());

  target.addIllegalOp<FunctionOp, ModelOp, PackageOp, RecordOp>();

  mlir::RewritePatternSet patterns(&getContext());

  patterns.insert<FunctionOpLowering, ModelOpLowering, PackageOpLowering,
                  RecordOpLowering>(&getContext());

  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

mlir::LogicalResult BaseModelicaToFuncConversionPass::convertRawFunctions() {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::ConversionTarget target(getContext());

  target.addIllegalOp<EquationFunctionOp, EquationCallOp, RawFunctionOp,
                      RawReturnOp, CallOp>();

  target.markUnknownOpDynamicallyLegal(
      [](mlir::Operation *op) { return true; });

  mlir::DataLayout dataLayout(moduleOp);
  TypeConverter typeConverter(&getContext(), dataLayout);

  mlir::RewritePatternSet patterns(&getContext());
  mlir::SymbolTableCollection symbolTableCollection;

  populateBaseModelicaToFuncConversionPatterns(
      patterns, &getContext(), typeConverter, symbolTableCollection);

  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

namespace mlir {
void populateBaseModelicaToFuncConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context,
    mlir::TypeConverter &typeConverter,
    mlir::SymbolTableCollection &symbolTableCollection) {
  patterns.insert<EquationFunctionOpLowering, RawFunctionOpLowering>(
      typeConverter, context, symbolTableCollection);

  patterns.insert<EquationCallOpLowering, RawReturnOpLowering, CallOpLowering>(
      typeConverter, context);
}

std::unique_ptr<mlir::Pass> createBaseModelicaToFuncConversionPass() {
  return std::make_unique<BaseModelicaToFuncConversionPass>();
}
} // namespace mlir

namespace {
class BaseModelicaRawVariablesConversionPass
    : public mlir::impl::BaseModelicaRawVariablesConversionPassBase<
          BaseModelicaRawVariablesConversionPass> {
public:
  using BaseModelicaRawVariablesConversionPassBase<
      BaseModelicaRawVariablesConversionPass>::
      BaseModelicaRawVariablesConversionPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult convertVariables();

  mlir::LogicalResult convertVariable(RawVariableOp op);
};
} // namespace

void BaseModelicaRawVariablesConversionPass::runOnOperation() {
  if (mlir::failed(convertVariables())) {
    return signalPassFailure();
  }
}

mlir::LogicalResult BaseModelicaRawVariablesConversionPass::convertVariables() {
  llvm::SmallVector<RawVariableOp> rawVariableOps;

  getOperation()->walk([&](RawVariableOp op) {
    if (mlir::isa<mlir::MemRefType>(op.getVariable().getType())) {
      rawVariableOps.push_back(op);
    }
  });

  for (RawVariableOp varOp : rawVariableOps) {
    if (mlir::failed(convertVariable(varOp))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

namespace {
class RawVariableConverter {
public:
  RawVariableConverter(mlir::RewriterBase &rewriter) : rewriter(rewriter) {}

  virtual ~RawVariableConverter() = default;

  virtual mlir::LogicalResult convert(RawVariableOp op) const = 0;

protected:
  mlir::RewriterBase &rewriter;
};

class RawVariableScalarConverter : public RawVariableConverter {
public:
  using RawVariableConverter::RawVariableConverter;

  mlir::LogicalResult convert(RawVariableOp op) const override {
    mlir::Location loc = op.getLoc();

    auto variableMemRefType =
        mlir::dyn_cast<mlir::MemRefType>(op.getVariable().getType());

    mlir::Value reference =
        createReference(rewriter, loc, variableMemRefType, op.getHeap());

    for (auto &use : llvm::make_early_inc_range(op->getUses())) {
      mlir::Operation *user = use.getOwner();

      if (auto getOp = mlir::dyn_cast<RawVariableGetOp>(user)) {
        if (mlir::failed(convertGetOp(rewriter, getOp, reference))) {
          return mlir::failure();
        }
      } else if (auto setOp = mlir::dyn_cast<RawVariableSetOp>(user)) {
        if (mlir::failed(convertSetOp(rewriter, setOp, reference))) {
          return mlir::failure();
        }
      } else {
        use.assign(reference);
      }
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }

  mlir::Value createReference(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::MemRefType memRefType, bool heap) const {
    if (heap) {
      auto allocOp =
          builder.create<mlir::memref::AllocOp>(loc, memRefType, std::nullopt);

      return allocOp.getResult();
    }

    auto allocaOp =
        builder.create<mlir::memref::AllocaOp>(loc, memRefType, std::nullopt);

    return allocaOp.getResult();
  }

  mlir::LogicalResult convertGetOp(mlir::RewriterBase &rewriter,
                                   RawVariableGetOp op,
                                   mlir::Value reference) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    mlir::Value replacement = rewriter.create<mlir::memref::LoadOp>(
        op.getLoc(), reference, std::nullopt);

    rewriter.replaceOp(op, replacement);
    return mlir::success();
  }

  mlir::LogicalResult convertSetOp(mlir::RewriterBase &rewriter,
                                   RawVariableSetOp op,
                                   mlir::Value reference) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, op.getValue(),
                                                       reference, std::nullopt);

    return mlir::success();
  }
};

class RawVariableStaticArrayConverter : public RawVariableConverter {
public:
  using RawVariableConverter::RawVariableConverter;

  mlir::LogicalResult convert(RawVariableOp op) const override {
    mlir::Location loc = op.getLoc();

    auto variableMemRefType =
        mlir::dyn_cast<mlir::MemRefType>(op.getVariable().getType());

    mlir::Value reference =
        createReference(rewriter, loc, variableMemRefType, op.getHeap());

    for (auto &use : llvm::make_early_inc_range(op->getUses())) {
      mlir::Operation *user = use.getOwner();

      if (auto getOp = mlir::dyn_cast<RawVariableGetOp>(user)) {
        if (mlir::failed(convertGetOp(rewriter, getOp, reference))) {
          return mlir::failure();
        }
      } else if (auto setOp = mlir::dyn_cast<RawVariableSetOp>(user)) {
        if (mlir::failed(convertSetOp(rewriter, setOp, reference))) {
          return mlir::failure();
        }
      } else {
        use.assign(reference);
      }
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }

  mlir::Value createReference(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::MemRefType memRefType, bool heap) const {
    if (heap) {
      auto allocOp =
          builder.create<mlir::memref::AllocOp>(loc, memRefType, std::nullopt);

      return allocOp.getResult();
    }

    auto allocaOp =
        builder.create<mlir::memref::AllocaOp>(loc, memRefType, std::nullopt);

    return allocaOp.getResult();
  }

  mlir::LogicalResult convertGetOp(mlir::RewriterBase &rewriter,
                                   RawVariableGetOp op,
                                   mlir::Value reference) const {
    rewriter.replaceOp(op, reference);
    return mlir::success();
  }

  mlir::LogicalResult convertSetOp(mlir::RewriterBase &rewriter,
                                   RawVariableSetOp op,
                                   mlir::Value reference) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    rewriter.create<mlir::memref::CopyOp>(op.getLoc(), op.getValue(),
                                          reference);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class RawVariableDynamicArrayConverter : public RawVariableConverter {
public:
  using RawVariableConverter::RawVariableConverter;

  mlir::LogicalResult convert(RawVariableOp op) const override {
    mlir::Location loc = op.getLoc();

    auto variableMemRefType =
        mlir::cast<mlir::MemRefType>(op.getVariable().getType());

    if (variableMemRefType.getShape().empty()) {
      return rewriter.notifyMatchFailure(op, "Not an array variable");
    }

    if (op.getDynamicSizes().size() ==
        static_cast<size_t>(variableMemRefType.getNumDynamicDims())) {
      return rewriter.notifyMatchFailure(op,
                                         "Not a dynamically sized variable");
    }

    // The variable can change sizes during at runtime. Thus, we need to
    // create a pointer to the array currently in use.

    // Create the pointer to the array.
    auto memRefOfMemRefType =
        mlir::MemRefType::get(std::nullopt, variableMemRefType);

    mlir::Value reference =
        rewriter.create<mlir::memref::AllocaOp>(loc, memRefOfMemRefType);

    // We need to allocate a fake buffer in order to allow the first free
    // operation to operate on a valid memory area.

    llvm::SmallVector<int64_t> zeroedDynDimsMemRefShape;

    getZeroedDynDimsMemRefShap(zeroedDynDimsMemRefShape,
                               variableMemRefType.getShape());

    auto zeroedDynDimsMemRefType = mlir::MemRefType::get(
        zeroedDynDimsMemRefShape, variableMemRefType.getElementType());

    mlir::Value fakeArray = rewriter.create<mlir::memref::AllocOp>(
        loc, zeroedDynDimsMemRefType, std::nullopt);

    if (fakeArray.getType() != memRefOfMemRefType.getElementType()) {
      fakeArray = rewriter.create<mlir::memref::CastOp>(
          loc, memRefOfMemRefType.getElementType(), fakeArray);
    }

    rewriter.create<mlir::memref::StoreOp>(loc, fakeArray, reference);

    // Replace the users.
    for (auto &use : llvm::make_early_inc_range(op->getUses())) {
      mlir::Operation *user = use.getOwner();

      if (auto getOp = mlir::dyn_cast<RawVariableGetOp>(user)) {
        if (mlir::failed(convertGetOp(rewriter, getOp, reference))) {
          return mlir::failure();
        }
      } else if (auto setOp = mlir::dyn_cast<RawVariableSetOp>(user)) {
        if (mlir::failed(convertSetOp(rewriter, setOp, reference))) {
          return mlir::failure();
        }
      } else {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(user);

        mlir::Value memRef =
            rewriter.create<mlir::memref::LoadOp>(user->getLoc(), reference);

        if (memRef.getType() != variableMemRefType) {
          memRef = rewriter.create<mlir::memref::CastOp>(
              memRef.getLoc(), variableMemRefType, memRef);
        }

        use.assign(memRef);
      }
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }

  void getZeroedDynDimsMemRefShap(llvm::SmallVectorImpl<int64_t> &result,
                                  llvm::ArrayRef<int64_t> shape) const {
    for (int64_t size : shape) {
      if (size == mlir::ShapedType::kDynamic) {
        result.push_back(0);
      } else {
        result.push_back(size);
      }
    }
  }

  mlir::LogicalResult convertGetOp(mlir::RewriterBase &rewriter,
                                   RawVariableGetOp op,
                                   mlir::Value reference) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    auto variableMemRefType =
        mlir::cast<mlir::MemRefType>(op.getVariable().getType());

    mlir::Value memRef =
        rewriter.create<mlir::memref::LoadOp>(op.getLoc(), reference);

    if (memRef.getType() != variableMemRefType) {
      memRef = rewriter.create<mlir::memref::CastOp>(
          op.getLoc(), variableMemRefType, memRef);
    }

    rewriter.replaceOp(op, memRef);
    return mlir::success();
  }

  mlir::LogicalResult convertSetOp(mlir::RewriterBase &rewriter,
                                   RawVariableSetOp op,
                                   mlir::Value reference) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    auto variableMemRefType =
        mlir::cast<mlir::MemRefType>(op.getVariable().getType());

    // The destination array has dynamic and unknown sizes. Thus, the array
    // has not been allocated yet, and we need to create a copy of the
    // source one.

    mlir::Value value = op.getValue();
    llvm::SmallVector<mlir::Value> dynamicSizes;

    for (int64_t dim = 0, rank = variableMemRefType.getRank(); dim < rank;
         ++dim) {
      if (variableMemRefType.getDimSize(dim) == mlir::ShapedType::kDynamic) {
        mlir::Value dimensionSize =
            rewriter.create<mlir::memref::DimOp>(op.getLoc(), value, dim);

        dynamicSizes.push_back(dimensionSize);
      }
    }

    mlir::Value valueCopy = rewriter.create<mlir::memref::AllocOp>(
        op.getLoc(), variableMemRefType, dynamicSizes);

    rewriter.create<mlir::memref::CopyOp>(op.getLoc(), value, valueCopy);

    // Deallocate the previously allocated memory. This is only
    // apparently in contrast with the above statements: unknown-sized
    // arrays pointers are initialized with a pointer to a 1-element
    // sized array, so that the initial deallocation always operates on valid
    // memory.

    mlir::Value previousMemRef =
        rewriter.create<mlir::memref::LoadOp>(op.getLoc(), reference);

    rewriter.create<mlir::memref::DeallocOp>(op.getLoc(), previousMemRef);

    // Save the reference to the new copy.
    rewriter.create<mlir::memref::StoreOp>(op.getLoc(), valueCopy, reference);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};
} // namespace

mlir::LogicalResult
BaseModelicaRawVariablesConversionPass::convertVariable(RawVariableOp op) {
  mlir::IRRewriter rewriter(op);
  std::unique_ptr<RawVariableConverter> converter;

  auto variableMemRefType =
      mlir::dyn_cast<mlir::MemRefType>(op.getVariable().getType());

  if (variableMemRefType.getShape().empty()) {
    converter = std::make_unique<RawVariableScalarConverter>(rewriter);
  } else if (op.getDynamicSizes().size() ==
             variableMemRefType.getNumDynamicDims()) {
    converter = std::make_unique<RawVariableStaticArrayConverter>(rewriter);
  } else {
    converter = std::make_unique<RawVariableDynamicArrayConverter>(rewriter);
  }

  return converter->convert(op);
}

namespace mlir {
void populateBaseModelicaRawVariablesTypeLegalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context,
    mlir::TypeConverter &typeConverter) {
  patterns.insert<RawVariableOpTypePattern, RawVariableGetOpTypePattern,
                  RawVariableSetOpTypePattern>(typeConverter, context);
}

std::unique_ptr<mlir::Pass> createBaseModelicaRawVariablesConversionPass() {
  return std::make_unique<BaseModelicaRawVariablesConversionPass>();
}
} // namespace mlir
