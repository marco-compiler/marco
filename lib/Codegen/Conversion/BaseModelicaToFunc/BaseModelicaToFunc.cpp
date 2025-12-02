#include "marco/Codegen/Conversion/BaseModelicaToFunc/BaseModelicaToFunc.h"
#include "marco/Codegen/Conversion/BaseModelicaCommon/TypeConverter.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_BASEMODELICATOFUNCCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

#define GEN_PASS_DEF_BASEMODELICAEXTERNALCALLSCONVERSIONPASS
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

    // Create the new function operation.
    auto funcOp = rewriter.create<mlir::func::FuncOp>(
        op.getLoc(), op.getSymName(), functionType);

    // Update the symbol table.
    symbolTable.remove(op);
    symbolTable.insert(funcOp, rewriter.getInsertionPoint());

    // Preserve the attributes.
    funcOp->setAttrs(op->getAttrs());

    // Add an attribute marking the equation function.
    funcOp->setAttr(BaseModelicaDialect::kEquationFunctionAttrName,
                    rewriter.getUnitAttr());

    // Move the body of the original function to the new function.
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
      rewriter.create<mlir::func::CallOp>(
          loc, op.getCallee(), mlir::TypeRange(), mlir::ValueRange());
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

        rewriter.create<mlir::func::CallOp>(loc, op.getCallee(),
                                            mlir::TypeRange(), boundaries);
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
    symbolTable.insert(funcOp, rewriter.getInsertionPoint());

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

struct ExternalCallOpTypePattern
    : public mlir::OpConversionPattern<ExternalCallOp> {
  using mlir::OpConversionPattern<ExternalCallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ExternalCallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type> convertedResultTypes;

    for (mlir::Type resultType : op.getResultTypes()) {
      convertedResultTypes.push_back(
          getTypeConverter()->convertType(resultType));
    }

    rewriter.replaceOpWithNewOp<ExternalCallOp>(
        op, convertedResultTypes, adaptor.getCallee(), adaptor.getArgs());

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
    mlir::SymbolTableCollection &symbolTables) {
  patterns.insert<EquationFunctionOpLowering, RawFunctionOpLowering>(
      typeConverter, context, symbolTables);

  patterns.insert<EquationCallOpLowering, RawReturnOpLowering, CallOpLowering>(
      typeConverter, context);
}

std::unique_ptr<mlir::Pass> createBaseModelicaToFuncConversionPass() {
  return std::make_unique<BaseModelicaToFuncConversionPass>();
}
} // namespace mlir

namespace {
class ExternalCallOpCLowering : public mlir::OpRewritePattern<ExternalCallOp> {
  mlir::SymbolTableCollection &symbolTables;
  int booleanBitWidth{32};
  int integerBitWidth{32};
  int indexBitWidth{64};

public:
  ExternalCallOpCLowering(mlir::MLIRContext *context,
                          mlir::SymbolTableCollection &symbolTables,
                          int booleanBitWidth, int integerBitWidth,
                          int indexBitWidth)
      : mlir::OpRewritePattern<ExternalCallOp>(context),
        symbolTables(symbolTables), booleanBitWidth(booleanBitWidth),
        integerBitWidth(integerBitWidth), indexBitWidth(indexBitWidth) {}

  mlir::LogicalResult
  matchAndRewrite(ExternalCallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getLanguage() != "C") {
      return rewriter.notifyMatchFailure(op, "Not a C function");
    }

    llvm::SmallVector<mlir::Type> newResultTypes;

    for (mlir::Type resultType : op.getResultTypes()) {
      newResultTypes.push_back(convertType(resultType));
    }

    llvm::SmallVector<mlir::Value> newArgs;

    for (mlir::Value arg : op.getArgs()) {
      if (auto memRefType = mlir::dyn_cast<mlir::MemRefType>(arg.getType())) {
        // If needed, convert the memref to a compatible type.
        mlir::Value memRef =
            convertMemRefAndCopyData(newArgs, rewriter, op, arg);

        // Add the pointer to the memref data to the list of call arguments.
        mlir::Value pointer =
            rewriter.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(
                memRef.getLoc(), memRef);

        pointer = rewriter.create<mlir::arith::IndexCastOp>(
            pointer.getLoc(), rewriter.getI64Type(), pointer);

        pointer = rewriter.create<mlir::LLVM::IntToPtrOp>(
            pointer.getLoc(),
            mlir::LLVM::LLVMPointerType::get(rewriter.getContext()), pointer);

        newArgs.push_back(pointer);

        // Add the dimensions of the memref to the list of call arguments.
        for (unsigned int dim = 0; dim < memRefType.getRank(); ++dim) {
          mlir::Value size = rewriter.create<mlir::memref::DimOp>(
              memRef.getLoc(), memRef, dim);

          size = rewriter.create<mlir::arith::IndexCastOp>(
              size.getLoc(), convertType(size.getType()), size);

          newArgs.push_back(size);
        }

        continue;
      }

      if (arg.getType() == convertType(arg.getType())) {
        newArgs.push_back(arg);
        continue;
      }

      newArgs.push_back(rewriter.create<CastOp>(
          arg.getLoc(), convertType(arg.getType()), arg));
    }

    auto newFuncOp = getOrDeclareExternalFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(), op.getLoc(),
        op.getCallee(), newArgs, newResultTypes);

    auto callOp =
        rewriter.create<mlir::func::CallOp>(op.getLoc(), newFuncOp, newArgs);

    // Convert the results back to the original types if needed.
    llvm::SmallVector<mlir::Value> convertedResults;

    for (unsigned int i = 0, e = callOp.getNumResults(); i < e; ++i) {
      mlir::Value result = callOp.getResult(i);
      mlir::Type originalType = op.getResultTypes()[i];
      mlir::Type convertedType = convertType(originalType);

      if (convertedType == originalType) {
        convertedResults.push_back(result);
      } else {
        convertedResults.push_back(
            rewriter.create<CastOp>(result.getLoc(), originalType, result));
      }
    }

    rewriter.replaceOp(op, convertedResults);
    return mlir::success();
  }

  mlir::Type convertType(mlir::Type type) const {
    if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(type)) {
      return convertIntegerType(integerType);
    }

    if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
      return convertFloatType(floatType);
    }

    if (auto indexType = mlir::dyn_cast<mlir::IndexType>(type)) {
      return convertIndexType(indexType);
    }

    if (auto memRefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
      mlir::Type convertedElementType =
          convertType(memRefType.getElementType());

      return mlir::MemRefType::get(memRefType.getShape(), convertedElementType);
    }

    return type;
  }

  mlir::Type convertIntegerType(mlir::IntegerType type) const {
    if (type.getIntOrFloatBitWidth() == 1) {
      return mlir::IntegerType::get(type.getContext(), booleanBitWidth);
    }

    return mlir::IntegerType::get(type.getContext(), integerBitWidth);
  }

  mlir::Type convertFloatType(mlir::FloatType type) const {
    return mlir::Float64Type::get(type.getContext());
  }

  mlir::Type convertIndexType(mlir::IndexType type) const {
    return mlir::IntegerType::get(type.getContext(), indexBitWidth);
  }

  mlir::Value
  convertMemRefAndCopyData(llvm::SmallVectorImpl<mlir::Value> &newArgs,
                           mlir::RewriterBase &rewriter, mlir::Operation *op,
                           mlir::Value arg) const {
    auto memRefType = mlir::cast<mlir::MemRefType>(arg.getType());
    auto convertedElementType = convertType(memRefType.getElementType());

    if (convertedElementType == memRefType.getElementType()) {
      return arg;
    }

    mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
        arg.getLoc(), rewriter.getIndexAttr(0));

    mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(
        arg.getLoc(), rewriter.getIndexAttr(1));

    llvm::SmallVector<mlir::Value> lowerBounds(memRefType.getRank(), zero);
    llvm::SmallVector<mlir::Value> upperBounds;
    llvm::SmallVector<mlir::Value> steps(memRefType.getRank(), one);

    llvm::SmallVector<mlir::Value> dynSizes;

    for (unsigned int dim = 0, rank = memRefType.getRank(); dim < rank; ++dim) {
      mlir::Value size =
          rewriter.create<mlir::memref::DimOp>(arg.getLoc(), arg, dim);

      upperBounds.push_back(size);

      if (memRefType.isDynamicDim(dim)) {
        dynSizes.push_back(size);
      }
    }

    mlir::Value newMemRef = rewriter.create<mlir::memref::AllocOp>(
        arg.getLoc(),
        mlir::MemRefType::get(memRefType.getShape(), convertedElementType),
        dynSizes);

    // Copy from the original memref to the temporary one.
    mlir::scf::buildLoopNest(
        rewriter, arg.getLoc(), lowerBounds, upperBounds, steps,
        [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
            mlir::ValueRange ivs) {
          mlir::Value value =
              nestedBuilder.create<mlir::memref::LoadOp>(nestedLoc, arg, ivs);

          value = nestedBuilder.create<CastOp>(nestedLoc, convertedElementType,
                                               value);

          nestedBuilder.create<mlir::memref::StoreOp>(nestedLoc, value,
                                                      newMemRef, ivs);
        });

    // Copy from the temporary memref to the original one.
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);

    mlir::scf::buildLoopNest(
        rewriter, arg.getLoc(), lowerBounds, upperBounds, steps,
        [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
            mlir::ValueRange ivs) {
          mlir::Value value = nestedBuilder.create<mlir::memref::LoadOp>(
              nestedLoc, newMemRef, ivs);

          value = nestedBuilder.create<CastOp>(
              nestedLoc, memRefType.getElementType(), value);

          nestedBuilder.create<mlir::memref::StoreOp>(nestedLoc, value, arg,
                                                      ivs);
        });

    return newMemRef;
  }

  mlir::func::FuncOp
  getOrDeclareExternalFunction(mlir::OpBuilder &builder,
                               mlir::ModuleOp moduleOp, mlir::Location loc,
                               llvm::StringRef name, mlir::ValueRange args,
                               mlir::TypeRange resultTypes) const {
    return getOrDeclareExternalFunction(builder, moduleOp, loc, name,
                                        args.getTypes(), resultTypes);
  }

  mlir::func::FuncOp
  getOrDeclareExternalFunction(mlir::OpBuilder &builder,
                               mlir::ModuleOp moduleOp, mlir::Location loc,
                               llvm::StringRef name, mlir::TypeRange argTypes,
                               mlir::TypeRange resultTypes) const {
    auto funcOp = symbolTables.lookupSymbolIn<mlir::func::FuncOp>(
        moduleOp, builder.getStringAttr(name));

    if (funcOp) {
      return funcOp;
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());

    auto newFuncOp = builder.create<mlir::func::FuncOp>(
        loc, name, builder.getFunctionType(argTypes, resultTypes));

    newFuncOp.setPrivate();

    symbolTables.getSymbolTable(moduleOp).insert(newFuncOp);
    return newFuncOp;
  }
};
} // namespace

namespace mlir {
void populateBaseModelicaExternalCallsTypeLegalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context,
    mlir::TypeConverter &typeConverter) {
  patterns.insert<ExternalCallOpTypePattern, RawVariableGetOpTypePattern,
                  RawVariableSetOpTypePattern>(typeConverter, context);
}

void populateBaseModelicaExternalCallConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context,
    mlir::SymbolTableCollection &symbolTables, int booleanBitWidth,
    int integerBitWidth, int indexBitWidth) {
  patterns.insert<ExternalCallOpCLowering>(
      context, symbolTables, booleanBitWidth, integerBitWidth, indexBitWidth);
}
} // namespace mlir

namespace {
class BaseModelicaExternalCallsConversionPass
    : public mlir::impl::BaseModelicaExternalCallsConversionPassBase<
          BaseModelicaExternalCallsConversionPass> {
public:
  using BaseModelicaExternalCallsConversionPassBase::
      BaseModelicaExternalCallsConversionPassBase;

  void runOnOperation() override;
};
} // namespace

void BaseModelicaExternalCallsConversionPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::ConversionTarget target(getContext());

  target.addIllegalOp<ExternalCallOp>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });

  mlir::RewritePatternSet patterns(&getContext());
  mlir::SymbolTableCollection symbolTables;

  populateBaseModelicaExternalCallConversionPatterns(
      patterns, &getContext(), symbolTables, booleanBitWidth, integerBitWidth,
      indexBitWidth);

  if (mlir::failed(
          applyPartialConversion(moduleOp, target, std::move(patterns)))) {
    mlir::emitError(getOperation().getLoc(),
                    "Error in converting the external calls");

    return signalPassFailure();
  }
}

namespace mlir {
std::unique_ptr<mlir::Pass> createBaseModelicaExternalCallsConversionPass() {
  return std::make_unique<BaseModelicaExternalCallsConversionPass>();
}

std::unique_ptr<mlir::Pass> createBaseModelicaExternalCallsConversionPass(
    const BaseModelicaExternalCallsConversionPassOptions &options) {
  return std::make_unique<BaseModelicaExternalCallsConversionPass>(options);
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
      auto allocOp = builder.create<mlir::memref::AllocOp>(loc, memRefType);

      return allocOp.getResult();
    }

    auto allocaOp = builder.create<mlir::memref::AllocaOp>(loc, memRefType);

    return allocaOp.getResult();
  }

  mlir::LogicalResult convertGetOp(mlir::RewriterBase &rewriter,
                                   RawVariableGetOp op,
                                   mlir::Value reference) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    mlir::Value replacement =
        rewriter.create<mlir::memref::LoadOp>(op.getLoc(), reference);

    rewriter.replaceOp(op, replacement);
    return mlir::success();
  }

  mlir::LogicalResult convertSetOp(mlir::RewriterBase &rewriter,
                                   RawVariableSetOp op,
                                   mlir::Value reference) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, op.getValue(),
                                                       reference);

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
      auto allocOp = builder.create<mlir::memref::AllocOp>(loc, memRefType);

      return allocOp.getResult();
    }

    auto allocaOp = builder.create<mlir::memref::AllocaOp>(loc, memRefType);

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
    auto memRefOfMemRefType = mlir::MemRefType::get({}, variableMemRefType);

    mlir::Value reference =
        rewriter.create<mlir::memref::AllocaOp>(loc, memRefOfMemRefType);

    // We need to allocate a fake buffer in order to allow the first free
    // operation to operate on a valid memory area.

    llvm::SmallVector<int64_t> zeroedDynDimsMemRefShape;

    getZeroedDynDimsMemRefShap(zeroedDynDimsMemRefShape,
                               variableMemRefType.getShape());

    auto zeroedDynDimsMemRefType = mlir::MemRefType::get(
        zeroedDynDimsMemRefShape, variableMemRefType.getElementType());

    mlir::Value fakeArray =
        rewriter.create<mlir::memref::AllocOp>(loc, zeroedDynDimsMemRefType);

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
