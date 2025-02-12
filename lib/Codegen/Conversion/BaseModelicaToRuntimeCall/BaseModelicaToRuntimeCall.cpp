#include "marco/Codegen/Conversion/BaseModelicaToRuntimeCall/BaseModelicaToRuntimeCall.h"
#include "marco/Codegen/Conversion/BaseModelicaCommon/TypeConverter.h"
#include "marco/Codegen/Runtime.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/Runtime/IR/Runtime.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_BASEMODELICATORUNTIMECALLCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
} // namespace mlir

using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace {
class BaseModelicaToRuntimeCallConversionPass
    : public mlir::impl::BaseModelicaToRuntimeCallConversionPassBase<
          BaseModelicaToRuntimeCallConversionPass> {
public:
  using BaseModelicaToRuntimeCallConversionPassBase<
      BaseModelicaToRuntimeCallConversionPass>::
      BaseModelicaToRuntimeCallConversionPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult convertOperations();
};
} // namespace

void BaseModelicaToRuntimeCallConversionPass::runOnOperation() {
  if (mlir::failed(convertOperations())) {
    return signalPassFailure();
  }
}

mlir::LogicalResult
BaseModelicaToRuntimeCallConversionPass::convertOperations() {
  mlir::ModuleOp module = getOperation();
  mlir::ConversionTarget target(getContext());

  target.addLegalDialect<
      BaseModelicaDialect, mlir::arith::ArithDialect,
      mlir::bufferization::BufferizationDialect, mlir::memref::MemRefDialect,
      mlir::runtime::RuntimeDialect, mlir::tensor::TensorDialect>();

  target
      .addIllegalOp<AbsOp, AcosOp, AsinOp, AtanOp, Atan2Op, CeilOp, CosOp,
                    CoshOp, DiagonalOp, DivTruncOp, ExpOp, FloorOp, IdentityOp,
                    IntegerOp, LinspaceOp, LogOp, Log10Op, OnesOp, MaxOp, MinOp,
                    ModOp, ProductOp, RemOp, SignOp, SinOp, SinhOp, SqrtOp,
                    SumOp, SymmetricOp, TanOp, TanhOp, TransposeOp, ZerosOp>();

  target.addDynamicallyLegalOp<PowOp>([](PowOp op) {
    if (op.getBase().getType().isa<mlir::TensorType>()) {
      return true;
    }

    return false;
  });

  target.addIllegalOp<PrintOp>();

  mlir::RewritePatternSet patterns(&getContext());
  TypeConverter typeConverter;
  mlir::SymbolTableCollection symbolTableCollection;

  populateBaseModelicaToRuntimeCallConversionPatterns(
      patterns, &getContext(), typeConverter, symbolTableCollection);

  return applyPartialConversion(module, target, std::move(patterns));
}

/// Get or declare a function inside the module.
static mlir::runtime::FunctionOp
getOrDeclareRuntimeFunction(mlir::OpBuilder &builder,
                            mlir::SymbolTableCollection &symbolTableCollection,
                            mlir::ModuleOp moduleOp, llvm::StringRef name,
                            mlir::TypeRange results, mlir::TypeRange args) {
  mlir::SymbolTable &symbolTable =
      symbolTableCollection.getSymbolTable(moduleOp);

  if (auto funcOp = symbolTable.lookup<mlir::runtime::FunctionOp>(name)) {
    return funcOp;
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());

  auto functionType = builder.getFunctionType(args, results);

  auto funcOp = builder.create<mlir::runtime::FunctionOp>(moduleOp.getLoc(),
                                                          name, functionType);

  symbolTable.insert(funcOp);

  mlir::SymbolTable::setSymbolVisibility(
      funcOp, mlir::SymbolTable::Visibility::Private);

  return funcOp;
}

/// Get or declare a function inside the module.
static mlir::runtime::FunctionOp
getOrDeclareRuntimeFunction(mlir::OpBuilder &builder,
                            mlir::SymbolTableCollection &symbolTableCollection,
                            mlir::ModuleOp moduleOp, llvm::StringRef name,
                            mlir::TypeRange results, mlir::ValueRange args) {
  llvm::SmallVector<mlir::Type, 3> argsTypes;

  for (mlir::Value arg : args) {
    argsTypes.push_back(arg.getType());
  }

  return getOrDeclareRuntimeFunction(builder, symbolTableCollection, moduleOp,
                                     name, results, argsTypes);
}

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
class RuntimeOpConversionPattern : public ModelicaOpConversionPattern<Op> {
public:
  RuntimeOpConversionPattern(mlir::TypeConverter &typeConverter,
                             mlir::MLIRContext *context,
                             mlir::SymbolTableCollection &symbolTableCollection)
      : ModelicaOpConversionPattern<Op>(typeConverter, context),
        symbolTableCollection(&symbolTableCollection) {}

protected:
  [[nodiscard]] const RuntimeFunctionsMangling *getMangler() const {
    return &mangler;
  }

  [[nodiscard]] std::string getMangledType(mlir::Type type) const {
    if (auto indexType = type.dyn_cast<mlir::IndexType>()) {
      return getMangledType(this->getTypeConverter()->convertType(type));
    }

    if (auto integerType = type.dyn_cast<mlir::IntegerType>()) {
      return getMangler()->getIntegerType(integerType.getWidth());
    }

    if (auto floatType = type.dyn_cast<mlir::FloatType>()) {
      return getMangler()->getFloatingPointType(floatType.getWidth());
    }

    if (auto tensorType = type.dyn_cast<mlir::UnrankedTensorType>()) {
      return getMangler()->getArrayType(
          getMangledType(tensorType.getElementType()));
    }

    if (auto memRefType = type.dyn_cast<mlir::UnrankedMemRefType>()) {
      return getMangler()->getArrayType(
          getMangledType(memRefType.getElementType()));
    }

    llvm_unreachable("Unknown type for mangling");
    return "unknown";
  }

  [[nodiscard]] std::string
  getMangledFunctionName(llvm::StringRef name, mlir::TypeRange resultTypes,
                         mlir::TypeRange argTypes) const {
    llvm::SmallVector<std::string, 3> mangledArgTypes;

    for (mlir::Type type : argTypes) {
      mangledArgTypes.push_back(getMangledType(type));
    }

    assert(resultTypes.size() <= 1);

    if (resultTypes.empty()) {
      return getMangler()->getMangledFunction(name, getMangler()->getVoidType(),
                                              mangledArgTypes);
    }

    return getMangler()->getMangledFunction(
        name, getMangledType(resultTypes[0]), mangledArgTypes);
  }

  [[nodiscard]] std::string
  getMangledFunctionName(llvm::StringRef name, mlir::TypeRange resultTypes,
                         mlir::ValueRange args) const {
    return getMangledFunctionName(name, resultTypes, args.getTypes());
  }

  mlir::runtime::FunctionOp
  getOrDeclareRuntimeFunction(mlir::OpBuilder &builder, mlir::ModuleOp moduleOp,
                              llvm::StringRef name, mlir::TypeRange results,
                              mlir::ValueRange args) const {
    return ::getOrDeclareRuntimeFunction(builder, *symbolTableCollection,
                                         moduleOp, name, results, args);
  }

  mlir::LogicalResult
  addCallArgument(mlir::OpBuilder &builder, mlir::Location loc,
                  llvm::SmallVectorImpl<mlir::Value> &arguments,
                  mlir::Value argument) const {
    mlir::Type argType = argument.getType();

    if (argType.isa<mlir::IntegerType, mlir::FloatType>()) {
      arguments.push_back(argument);
      return mlir::success();
    }

    if (argType.isa<mlir::IndexType>()) {
      arguments.push_back(builder.create<CastOp>(
          argument.getLoc(), builder.getI64Type(), argument));

      return mlir::success();
    }

    if (auto tensorType = argType.dyn_cast<mlir::RankedTensorType>()) {
      mlir::Value tensor = argument;

      if (!tensor.getType().isa<mlir::UnrankedTensorType>()) {
        auto unrankedTensorType =
            mlir::UnrankedTensorType::get(tensorType.getElementType());

        tensor = builder.create<mlir::tensor::CastOp>(loc, unrankedTensorType,
                                                      tensor);
      }

      arguments.push_back(tensor);
      return mlir::success();
    }

    if (auto memRefType = argType.dyn_cast<mlir::MemRefType>()) {
      mlir::Value memRef = argument;

      if (!memRef.getType().isa<mlir::UnrankedMemRefType>()) {
        auto unrankedMemRefType = mlir::UnrankedMemRefType::get(
            memRefType.getElementType(), memRefType.getMemorySpace());

        memRef = builder.create<mlir::memref::CastOp>(loc, unrankedMemRefType,
                                                      memRef);
      }

      arguments.push_back(memRef);
      return mlir::success();
    }

    return mlir::failure();
  }

  mlir::Value convertToUnrankedMemRef(mlir::OpBuilder &builder,
                                      mlir::Value memRef) const {
    auto memRefType = memRef.getType().cast<mlir::MemRefType>();

    auto unrankedMemRefType = mlir::UnrankedMemRefType::get(
        memRefType.getElementType(), memRefType.getMemorySpace());

    return builder.create<mlir::memref::CastOp>(memRef.getLoc(),
                                                unrankedMemRefType, memRef);
  }

  mlir::Value convertToTensor(mlir::OpBuilder &builder,
                              mlir::Value memRef) const {
    auto memRefType = memRef.getType().cast<mlir::MemRefType>();

    auto tensorType = mlir::RankedTensorType::get(memRefType.getShape(),
                                                  memRefType.getElementType());

    return builder.create<mlir::bufferization::ToTensorOp>(
        memRef.getLoc(), tensorType, memRef, true, true);
  }

private:
  RuntimeFunctionsMangling mangler;
  mlir::SymbolTableCollection *symbolTableCollection;
};
} // namespace

//===---------------------------------------------------------------------===//
// Math operations
//===---------------------------------------------------------------------===//

namespace {
struct PowOpLowering : public RuntimeOpConversionPattern<PowOp> {
  using RuntimeOpConversionPattern<PowOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(PowOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value base = adaptor.getBase();

    if (!base.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible base operand");
    }

    llvm::SmallVector<mlir::Value, 2> arguments;

    // Base.
    if (mlir::failed(
            addCallArgument(rewriter, loc, arguments, adaptor.getBase()))) {
      return mlir::failure();
    }

    // Exponent.
    if (mlir::failed(
            addCallArgument(rewriter, loc, arguments, adaptor.getExponent()))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    mlir::Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());

    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("pow", resultType, arguments), resultType,
        arguments);

    rewriter.replaceOpWithNewOp<mlir::runtime::CallOp>(op, callee, arguments);
    return mlir::success();
  }
};
} // namespace

//===---------------------------------------------------------------------===//
// Built-in functions
//===---------------------------------------------------------------------===//

namespace {
struct AbsOpLowering : public RuntimeOpConversionPattern<AbsOp> {
  using RuntimeOpConversionPattern<AbsOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(AbsOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperand();

    if (!operand.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    llvm::SmallVector<mlir::Value, 1> arguments;

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, operand))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("abs", operand.getType(), arguments),
        operand.getType(), arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct AcosOpLowering : public RuntimeOpConversionPattern<AcosOp> {
  using RuntimeOpConversionPattern<AcosOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(AcosOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperand();

    if (!operand.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    llvm::SmallVector<mlir::Value, 1> arguments;

    if (!op.getType().isa<mlir::FloatType>() ||
        (operand.getType().getIntOrFloatBitWidth() != 32 &&
         operand.getType().getIntOrFloatBitWidth() != 64)) {
      operand = rewriter.create<CastOp>(loc, rewriter.getF64Type(), operand);
    }

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, operand))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("acos", operand.getType(), arguments),
        operand.getType(), arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct AsinOpLowering : public RuntimeOpConversionPattern<AsinOp> {
  using RuntimeOpConversionPattern<AsinOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(AsinOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperand();

    if (!operand.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    llvm::SmallVector<mlir::Value, 1> arguments;

    if (!op.getType().isa<mlir::FloatType>() ||
        (operand.getType().getIntOrFloatBitWidth() != 32 &&
         operand.getType().getIntOrFloatBitWidth() != 64)) {
      operand = rewriter.create<CastOp>(loc, rewriter.getF64Type(), operand);
    }

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, operand))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("asin", operand.getType(), arguments),
        operand.getType(), arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct AtanOpLowering : public RuntimeOpConversionPattern<AtanOp> {
  using RuntimeOpConversionPattern<AtanOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(AtanOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperand();

    if (!operand.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    llvm::SmallVector<mlir::Value, 1> arguments;

    if (!op.getType().isa<mlir::FloatType>() ||
        (operand.getType().getIntOrFloatBitWidth() != 32 &&
         operand.getType().getIntOrFloatBitWidth() != 64)) {
      operand = rewriter.create<CastOp>(loc, rewriter.getF64Type(), operand);
    }

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, operand))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("atan", operand.getType(), arguments),
        operand.getType(), arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct Atan2OpLowering : public RuntimeOpConversionPattern<Atan2Op> {
  using RuntimeOpConversionPattern<Atan2Op>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Atan2Op op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value yValue = adaptor.getY();
    mlir::Value xValue = adaptor.getX();

    if (!yValue.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    if (!xValue.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    mlir::Type mostGenericType = getMostGenericScalarType(yValue, xValue);

    if (!mostGenericType.isa<mlir::FloatType>() ||
        (mostGenericType.getIntOrFloatBitWidth() != 32 &&
         mostGenericType.getIntOrFloatBitWidth() != 64)) {
      mostGenericType = rewriter.getF64Type();
    }

    if (yValue.getType() != mostGenericType) {
      yValue = rewriter.create<CastOp>(loc, mostGenericType, yValue);
    }

    if (xValue.getType() != mostGenericType) {
      xValue = rewriter.create<CastOp>(loc, mostGenericType, xValue);
    }

    llvm::SmallVector<mlir::Value, 2> arguments;

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, yValue))) {
      return mlir::failure();
    }

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, xValue))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("atan2", mostGenericType, arguments),
        mostGenericType, arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct CeilOpLowering : public RuntimeOpConversionPattern<CeilOp> {
  using RuntimeOpConversionPattern<CeilOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CeilOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperand();

    if (!operand.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    llvm::SmallVector<mlir::Value, 1> arguments;

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, operand))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("ceil", operand.getType(), arguments),
        operand.getType(), arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct CosOpLowering : public RuntimeOpConversionPattern<CosOp> {
  using RuntimeOpConversionPattern<CosOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CosOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperand();

    if (!operand.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    llvm::SmallVector<mlir::Value, 1> arguments;

    if (!op.getType().isa<mlir::FloatType>() ||
        (operand.getType().getIntOrFloatBitWidth() != 32 &&
         operand.getType().getIntOrFloatBitWidth() != 64)) {
      operand = rewriter.create<CastOp>(loc, rewriter.getF64Type(), operand);
    }

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, operand))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("cos", operand.getType(), arguments),
        operand.getType(), arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct CoshOpLowering : public RuntimeOpConversionPattern<CoshOp> {
  using RuntimeOpConversionPattern<CoshOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CoshOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperand();

    if (!operand.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    llvm::SmallVector<mlir::Value, 1> arguments;

    if (!op.getType().isa<mlir::FloatType>() ||
        (operand.getType().getIntOrFloatBitWidth() != 32 &&
         operand.getType().getIntOrFloatBitWidth() != 64)) {
      operand = rewriter.create<CastOp>(loc, rewriter.getF64Type(), operand);
    }

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, operand))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("cosh", operand.getType(), arguments),
        operand.getType(), arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct DiagonalOpLowering : public RuntimeOpConversionPattern<DiagonalOp> {
  using RuntimeOpConversionPattern<DiagonalOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DiagonalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value values = adaptor.getValues();

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    auto requestedResultTensorType =
        requestedResultType.cast<mlir::TensorType>();

    // Allocate the memory for the result.
    llvm::SmallVector<mlir::Value, 2> dynamicDimensions;

    for (int64_t dimensionSize : requestedResultTensorType.getShape()) {
      if (dimensionSize == mlir::ShapedType::kDynamic) {
        if (dynamicDimensions.empty()) {
          mlir::Value zeroValue = rewriter.create<mlir::arith::ConstantOp>(
              loc, rewriter.getIndexAttr(0));

          dynamicDimensions.push_back(
              rewriter.create<DimOp>(loc, values, zeroValue));
        } else {
          dynamicDimensions.push_back(dynamicDimensions[0]);
        }
      }
    }

    auto resultMemRefType =
        mlir::MemRefType::get(requestedResultTensorType.getShape(),
                              requestedResultTensorType.getElementType());

    mlir::Value result = rewriter.create<mlir::memref::AllocOp>(
        loc, resultMemRefType, dynamicDimensions);

    result = convertToUnrankedMemRef(rewriter, result);

    // Convert the values tensor to an unranked memref.
    auto valuesTensorType = values.getType().cast<mlir::TensorType>();

    if (requestedResultTensorType.getElementType() !=
        valuesTensorType.getElementType()) {
      auto compatibleValuesTensorType =
          valuesTensorType.clone(requestedResultTensorType.getElementType());

      values = rewriter.create<CastOp>(loc, compatibleValuesTensorType, values);
    }

    // Collect the arguments for the function call.
    llvm::SmallVector<mlir::Value, 2> arguments;
    arguments.push_back(result);

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, values))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("diagonal", std::nullopt, arguments),
        std::nullopt, arguments);

    rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    result =
        rewriter.create<mlir::memref::CastOp>(loc, resultMemRefType, result);

    result = convertToTensor(rewriter, result);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct DivTruncOpLowering : public RuntimeOpConversionPattern<DivTruncOp> {
  using RuntimeOpConversionPattern<DivTruncOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DivTruncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value xValue = adaptor.getX();
    mlir::Value yValue = adaptor.getY();

    if (!xValue.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    if (!yValue.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    mlir::Type mostGenericType = getMostGenericScalarType(xValue, yValue);

    if (xValue.getType() != mostGenericType) {
      xValue = rewriter.create<CastOp>(loc, mostGenericType, xValue);
    }

    if (yValue.getType() != mostGenericType) {
      yValue = rewriter.create<CastOp>(loc, mostGenericType, yValue);
    }

    llvm::SmallVector<mlir::Value, 2> arguments;

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, xValue))) {
      return mlir::failure();
    }

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, yValue))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("div", mostGenericType, arguments),
        mostGenericType, arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct ExpOpLowering : public RuntimeOpConversionPattern<ExpOp> {
  using RuntimeOpConversionPattern<ExpOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ExpOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value exponent = adaptor.getExponent();

    if (!exponent.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    llvm::SmallVector<mlir::Value, 1> arguments;

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, exponent))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("exp", exponent.getType(), arguments),
        exponent.getType(), arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct FloorOpLowering : public RuntimeOpConversionPattern<FloorOp> {
  using RuntimeOpConversionPattern<FloorOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(FloorOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperand();

    if (!operand.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    llvm::SmallVector<mlir::Value, 1> arguments;

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, operand))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("floor", operand.getType(), arguments),
        operand.getType(), arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct IdentityOpLowering : public RuntimeOpConversionPattern<IdentityOp> {
  using RuntimeOpConversionPattern<IdentityOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(IdentityOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value size = adaptor.getSize();

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    auto requestedResultTensorType =
        requestedResultType.cast<mlir::TensorType>();

    // Allocate the memory for the result.
    llvm::SmallVector<mlir::Value, 2> dynamicDimensions;

    for (int64_t dimensionSize : requestedResultTensorType.getShape()) {
      if (dimensionSize == mlir::ShapedType::kDynamic) {
        if (dynamicDimensions.empty()) {
          mlir::Value indexSize = size;

          if (!indexSize.getType().isa<mlir::IndexType>()) {
            indexSize = rewriter.create<CastOp>(loc, rewriter.getIndexType(),
                                                indexSize);
          }

          dynamicDimensions.push_back(indexSize);
        } else {
          dynamicDimensions.push_back(dynamicDimensions[0]);
        }
      }
    }

    auto resultMemRefType =
        mlir::MemRefType::get(requestedResultTensorType.getShape(),
                              requestedResultTensorType.getElementType());

    mlir::Value result = rewriter.create<mlir::memref::AllocOp>(
        loc, resultMemRefType, dynamicDimensions);

    result = convertToUnrankedMemRef(rewriter, result);

    // Collect the arguments for the function call.
    llvm::SmallVector<mlir::Value, 1> arguments;
    arguments.push_back(result);

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("identity", std::nullopt, arguments),
        std::nullopt, arguments);

    rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    result =
        rewriter.create<mlir::memref::CastOp>(loc, resultMemRefType, result);

    result = convertToTensor(rewriter, result);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct IntegerOpLowering : public RuntimeOpConversionPattern<IntegerOp> {
  using RuntimeOpConversionPattern<IntegerOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(IntegerOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperand();

    if (!operand.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    llvm::SmallVector<mlir::Value, 1> arguments;

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, operand))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("integer", operand.getType(), arguments),
        operand.getType(), arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct LinspaceOpLowering : public RuntimeOpConversionPattern<LinspaceOp> {
  using RuntimeOpConversionPattern<LinspaceOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(LinspaceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value begin = adaptor.getBegin();
    mlir::Value end = adaptor.getEnd();

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    auto requestedResultTensorType =
        requestedResultType.cast<mlir::TensorType>();

    // Allocate the memory for the result.
    llvm::SmallVector<mlir::Value, 1> dynamicDimensions;

    if (requestedResultTensorType.getDimSize(0) == mlir::ShapedType::kDynamic) {
      mlir::Value amount = adaptor.getAmount();

      if (!amount.getType().isa<mlir::IndexType>()) {
        amount = rewriter.create<CastOp>(loc, rewriter.getIndexType(), amount);
      }

      dynamicDimensions.push_back(amount);
    }

    auto resultMemRefType =
        mlir::MemRefType::get(requestedResultTensorType.getShape(),
                              requestedResultTensorType.getElementType());

    mlir::Value result = rewriter.create<mlir::memref::AllocOp>(
        loc, resultMemRefType, dynamicDimensions);

    result = convertToUnrankedMemRef(rewriter, result);

    // Collect the arguments for the function call.
    llvm::SmallVector<mlir::Value, 3> arguments;
    arguments.push_back(result);

    mlir::Type mostGenericType = getMostGenericScalarType(begin, end);

    if (!mostGenericType.isa<mlir::FloatType>() ||
        (mostGenericType.getIntOrFloatBitWidth() != 32 &&
         mostGenericType.getIntOrFloatBitWidth() != 64)) {
      mostGenericType = rewriter.getF64Type();
    }

    if (begin.getType() != mostGenericType) {
      begin = rewriter.create<CastOp>(loc, mostGenericType, begin);
    }

    if (end.getType() != mostGenericType) {
      end = rewriter.create<CastOp>(loc, mostGenericType, end);
    }

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, begin))) {
      return mlir::failure();
    }

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, end))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("linspace", std::nullopt, arguments),
        std::nullopt, arguments);

    rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    result =
        rewriter.create<mlir::memref::CastOp>(loc, resultMemRefType, result);

    result = convertToTensor(rewriter, result);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct LogOpLowering : public RuntimeOpConversionPattern<LogOp> {
  using RuntimeOpConversionPattern<LogOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(LogOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperand();

    if (!operand.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    llvm::SmallVector<mlir::Value, 1> arguments;

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, operand))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("log", operand.getType(), arguments),
        operand.getType(), arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct Log10OpLowering : public RuntimeOpConversionPattern<Log10Op> {
  using RuntimeOpConversionPattern<Log10Op>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Log10Op op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperand();

    if (!operand.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    llvm::SmallVector<mlir::Value, 1> arguments;

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, operand))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("log10", operand.getType(), arguments),
        operand.getType(), arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct MaxOpScalarsLowering : public RuntimeOpConversionPattern<MaxOp> {
  using RuntimeOpConversionPattern<MaxOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(MaxOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    if (adaptor.getOperands().size() != 2) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    mlir::Value first = adaptor.getFirst();
    mlir::Value second = adaptor.getSecond();

    if (!first.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    if (!second.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    mlir::Type mostGenericType = getMostGenericScalarType(first, second);

    if (first.getType() != mostGenericType) {
      first = rewriter.create<CastOp>(loc, mostGenericType, first);
    }

    if (second.getType() != mostGenericType) {
      second = rewriter.create<CastOp>(loc, mostGenericType, second);
    }

    llvm::SmallVector<mlir::Value, 2> arguments;

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, first))) {
      return mlir::failure();
    }

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, second))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("maxScalars", mostGenericType, arguments),
        mostGenericType, arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct MaxOpArrayLowering : public RuntimeOpConversionPattern<MaxOp> {
  using RuntimeOpConversionPattern<MaxOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(MaxOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    if (adaptor.getOperands().size() != 1) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    mlir::Value array = adaptor.getFirst();

    // Collect the arguments for the function call.
    llvm::SmallVector<mlir::Value, 1> arguments;

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, array))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    mlir::Type elementType =
        array.getType().cast<mlir::TensorType>().getElementType();

    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("maxArray", elementType, arguments), elementType,
        arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct MinOpScalarsLowering : public RuntimeOpConversionPattern<MinOp> {
  using RuntimeOpConversionPattern<MinOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(MinOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    if (adaptor.getOperands().size() != 2) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    mlir::Value first = adaptor.getFirst();
    mlir::Value second = adaptor.getSecond();

    if (!first.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    if (!second.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    mlir::Type mostGenericType = getMostGenericScalarType(first, second);

    if (first.getType() != mostGenericType) {
      first = rewriter.create<CastOp>(loc, mostGenericType, first);
    }

    if (second.getType() != mostGenericType) {
      second = rewriter.create<CastOp>(loc, mostGenericType, second);
    }

    llvm::SmallVector<mlir::Value, 2> arguments;

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, first))) {
      return mlir::failure();
    }

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, second))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("minScalars", mostGenericType, arguments),
        mostGenericType, arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct MinOpArrayLowering : public RuntimeOpConversionPattern<MinOp> {
  using RuntimeOpConversionPattern<MinOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(MinOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    if (adaptor.getOperands().size() != 1) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    mlir::Value array = adaptor.getFirst();

    // Collect the arguments for the function call.
    llvm::SmallVector<mlir::Value, 1> arguments;

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, array))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    mlir::Type elementType =
        array.getType().cast<mlir::TensorType>().getElementType();

    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("minArray", elementType, arguments), elementType,
        arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct ModOpLowering : public RuntimeOpConversionPattern<ModOp> {
  using RuntimeOpConversionPattern<ModOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ModOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value xValue = adaptor.getX();
    mlir::Value yValue = adaptor.getY();

    if (!xValue.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    if (!yValue.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    mlir::Type mostGenericType = getMostGenericScalarType(yValue, xValue);

    if (xValue.getType() != mostGenericType) {
      xValue = rewriter.create<CastOp>(loc, mostGenericType, xValue);
    }

    if (yValue.getType() != mostGenericType) {
      yValue = rewriter.create<CastOp>(loc, mostGenericType, yValue);
    }

    llvm::SmallVector<mlir::Value, 2> arguments;

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, xValue))) {
      return mlir::failure();
    }

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, yValue))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("mod", mostGenericType, arguments),
        mostGenericType, arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct OnesOpLowering : public RuntimeOpConversionPattern<OnesOp> {
  using RuntimeOpConversionPattern<OnesOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(OnesOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto sizes = adaptor.getSizes();

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    auto requestedResultTensorType =
        requestedResultType.cast<mlir::TensorType>();

    // Allocate the memory for the result.
    llvm::SmallVector<mlir::Value, 2> dynamicDimensions;

    for (int64_t dim = 0, rank = requestedResultTensorType.getRank();
         dim < rank; ++dim) {
      if (requestedResultTensorType.getDimSize(dim) ==
          mlir::ShapedType::kDynamic) {
        mlir::Value size = sizes[dim];

        if (!size.getType().isa<mlir::IndexType>()) {
          size = rewriter.create<CastOp>(loc, rewriter.getIndexType(), size);
        }

        dynamicDimensions.push_back(size);
      }
    }

    auto resultMemRefType =
        mlir::MemRefType::get(requestedResultTensorType.getShape(),
                              requestedResultTensorType.getElementType());

    mlir::Value result = rewriter.create<mlir::memref::AllocOp>(
        loc, resultMemRefType, dynamicDimensions);

    result = convertToUnrankedMemRef(rewriter, result);

    // Collect the arguments for the function call.
    llvm::SmallVector<mlir::Value, 1> arguments;
    arguments.push_back(result);

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("ones", std::nullopt, arguments), std::nullopt,
        arguments);

    rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    result =
        rewriter.create<mlir::memref::CastOp>(loc, resultMemRefType, result);

    result = convertToTensor(rewriter, result);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct ProductOpLowering : public RuntimeOpConversionPattern<ProductOp> {
  using RuntimeOpConversionPattern<ProductOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ProductOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value array = adaptor.getArray();

    // Collect the arguments for the function call.
    llvm::SmallVector<mlir::Value, 1> arguments;

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, array))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    mlir::Type elementType =
        array.getType().cast<mlir::TensorType>().getElementType();

    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("product", elementType, arguments), elementType,
        arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct RemOpLowering : public RuntimeOpConversionPattern<RemOp> {
  using RuntimeOpConversionPattern<RemOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(RemOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value xValue = adaptor.getX();
    mlir::Value yValue = adaptor.getY();

    if (!xValue.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    if (!yValue.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    mlir::Type mostGenericType = getMostGenericScalarType(xValue, yValue);

    if (xValue.getType() != mostGenericType) {
      xValue = rewriter.create<CastOp>(loc, mostGenericType, xValue);
    }

    if (yValue.getType() != mostGenericType) {
      yValue = rewriter.create<CastOp>(loc, mostGenericType, yValue);
    }

    llvm::SmallVector<mlir::Value, 2> arguments;

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, xValue))) {
      return mlir::failure();
    }

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, yValue))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("rem", mostGenericType, arguments),
        mostGenericType, arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct SignOpLowering : public RuntimeOpConversionPattern<SignOp> {
  using RuntimeOpConversionPattern<SignOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SignOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperand();

    if (!operand.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    llvm::SmallVector<mlir::Value, 1> arguments;

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, operand))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("sign", operand.getType(), arguments),
        operand.getType(), arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct SinOpLowering : public RuntimeOpConversionPattern<SinOp> {
  using RuntimeOpConversionPattern<SinOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SinOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperand();

    if (!operand.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    llvm::SmallVector<mlir::Value, 1> arguments;

    if (!op.getType().isa<mlir::FloatType>() ||
        (operand.getType().getIntOrFloatBitWidth() != 32 &&
         operand.getType().getIntOrFloatBitWidth() != 64)) {
      operand = rewriter.create<CastOp>(loc, rewriter.getF64Type(), operand);
    }

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, operand))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("sin", operand.getType(), arguments),
        operand.getType(), arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct SinhOpLowering : public RuntimeOpConversionPattern<SinhOp> {
  using RuntimeOpConversionPattern<SinhOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SinhOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperand();

    if (!operand.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    llvm::SmallVector<mlir::Value, 1> arguments;

    if (!op.getType().isa<mlir::FloatType>() ||
        (operand.getType().getIntOrFloatBitWidth() != 32 &&
         operand.getType().getIntOrFloatBitWidth() != 64)) {
      operand = rewriter.create<CastOp>(loc, rewriter.getF64Type(), operand);
    }

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, operand))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("sinh", operand.getType(), arguments),
        operand.getType(), arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct SqrtOpLowering : public RuntimeOpConversionPattern<SqrtOp> {
  using RuntimeOpConversionPattern<SqrtOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SqrtOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperand();

    if (!operand.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    llvm::SmallVector<mlir::Value, 1> arguments;

    if (!op.getType().isa<mlir::FloatType>() ||
        (operand.getType().getIntOrFloatBitWidth() != 32 &&
         operand.getType().getIntOrFloatBitWidth() != 64)) {
      operand = rewriter.create<CastOp>(loc, rewriter.getF64Type(), operand);
    }

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, operand))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("sqrt", operand.getType(), arguments),
        operand.getType(), arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct SumOpLowering : public RuntimeOpConversionPattern<SumOp> {
  using RuntimeOpConversionPattern<SumOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SumOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value array = adaptor.getArray();

    // Collect the arguments for the function call.
    llvm::SmallVector<mlir::Value, 1> arguments;

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, array))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    mlir::Type elementType =
        array.getType().cast<mlir::TensorType>().getElementType();

    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("sum", elementType, arguments), elementType,
        arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct SymmetricOpLowering : public RuntimeOpConversionPattern<SymmetricOp> {
  using RuntimeOpConversionPattern<SymmetricOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SymmetricOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value matrix = adaptor.getMatrix();

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    auto requestedResultTensorType =
        requestedResultType.cast<mlir::TensorType>();

    // Allocate the memory for the result.
    llvm::SmallVector<mlir::Value, 2> dynamicDimensions;

    if (requestedResultTensorType.getDimSize(0) == mlir::ShapedType::kDynamic) {
      mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getIndexAttr(1));

      mlir::Value size = rewriter.create<DimOp>(loc, matrix, one);
      dynamicDimensions.push_back(size);
    }

    if (requestedResultTensorType.getDimSize(1) == mlir::ShapedType::kDynamic) {
      mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getIndexAttr(0));

      mlir::Value size = rewriter.create<DimOp>(loc, matrix, zero);
      dynamicDimensions.push_back(size);
    }

    auto resultMemRefType =
        mlir::MemRefType::get(requestedResultTensorType.getShape(),
                              requestedResultTensorType.getElementType());

    mlir::Value result = rewriter.create<mlir::memref::AllocOp>(
        loc, resultMemRefType, dynamicDimensions);

    result = convertToUnrankedMemRef(rewriter, result);

    // Collect the arguments for the function call.
    llvm::SmallVector<mlir::Value, 2> arguments;
    arguments.push_back(result);

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, matrix))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("symmetric", std::nullopt, arguments),
        std::nullopt, arguments);

    rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    result =
        rewriter.create<mlir::memref::CastOp>(loc, resultMemRefType, result);

    result = convertToTensor(rewriter, result);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct TanOpLowering : public RuntimeOpConversionPattern<TanOp> {
  using RuntimeOpConversionPattern<TanOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(TanOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperand();

    if (!operand.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    llvm::SmallVector<mlir::Value, 1> arguments;

    if (!op.getType().isa<mlir::FloatType>() ||
        (operand.getType().getIntOrFloatBitWidth() != 32 &&
         operand.getType().getIntOrFloatBitWidth() != 64)) {
      operand = rewriter.create<CastOp>(loc, rewriter.getF64Type(), operand);
    }

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, operand))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("tan", operand.getType(), arguments),
        operand.getType(), arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct TanhOpLowering : public RuntimeOpConversionPattern<TanhOp> {
  using RuntimeOpConversionPattern<TanhOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(TanhOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperand();

    if (!operand.getType()
             .isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    llvm::SmallVector<mlir::Value, 1> arguments;

    if (!op.getType().isa<mlir::FloatType>() ||
        (operand.getType().getIntOrFloatBitWidth() != 32 &&
         operand.getType().getIntOrFloatBitWidth() != 64)) {
      operand = rewriter.create<CastOp>(loc, rewriter.getF64Type(), operand);
    }

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, operand))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("tanh", operand.getType(), arguments),
        operand.getType(), arguments);

    auto callOp =
        rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    mlir::Value result = callOp.getResult(0);

    // Cast to requested result type.
    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct TransposeOpLowering : public RuntimeOpConversionPattern<TransposeOp> {
  using RuntimeOpConversionPattern<TransposeOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value matrix = adaptor.getMatrix();

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    auto requestedResultTensorType =
        requestedResultType.cast<mlir::TensorType>();

    // Allocate the memory for the result.
    llvm::SmallVector<mlir::Value, 2> dynamicDimensions;

    if (requestedResultTensorType.getDimSize(0) == mlir::ShapedType::kDynamic) {
      mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getIndexAttr(1));

      mlir::Value size = rewriter.create<DimOp>(loc, matrix, one);
      dynamicDimensions.push_back(size);
    }

    if (requestedResultTensorType.getDimSize(1) == mlir::ShapedType::kDynamic) {
      mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getIndexAttr(0));

      mlir::Value size = rewriter.create<DimOp>(loc, matrix, zero);
      dynamicDimensions.push_back(size);
    }

    auto resultMemRefType =
        mlir::MemRefType::get(requestedResultTensorType.getShape(),
                              requestedResultTensorType.getElementType());

    mlir::Value result = rewriter.create<mlir::memref::AllocOp>(
        loc, resultMemRefType, dynamicDimensions);

    result = convertToUnrankedMemRef(rewriter, result);

    // Collect the arguments for the function call.
    llvm::SmallVector<mlir::Value, 2> arguments;
    arguments.push_back(result);

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, matrix))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("transpose", std::nullopt, arguments),
        std::nullopt, arguments);

    rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    result =
        rewriter.create<mlir::memref::CastOp>(loc, resultMemRefType, result);

    result = convertToTensor(rewriter, result);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct ZerosOpLowering : public RuntimeOpConversionPattern<ZerosOp> {
  using RuntimeOpConversionPattern<ZerosOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ZerosOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto sizes = adaptor.getSizes();

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    auto requestedResultTensorType =
        requestedResultType.cast<mlir::TensorType>();

    // Allocate the memory for the result.
    llvm::SmallVector<mlir::Value, 2> dynamicDimensions;

    for (int64_t dim = 0, rank = requestedResultTensorType.getRank();
         dim < rank; ++dim) {
      if (requestedResultTensorType.getDimSize(dim) ==
          mlir::ShapedType::kDynamic) {
        mlir::Value size = sizes[dim];

        if (!size.getType().isa<mlir::IndexType>()) {
          size = rewriter.create<CastOp>(loc, rewriter.getIndexType(), size);
        }

        dynamicDimensions.push_back(size);
      }
    }

    auto resultMemRefType =
        mlir::MemRefType::get(requestedResultTensorType.getShape(),
                              requestedResultTensorType.getElementType());

    mlir::Value result = rewriter.create<mlir::memref::AllocOp>(
        loc, resultMemRefType, dynamicDimensions);

    result = convertToUnrankedMemRef(rewriter, result);

    // Collect the arguments for the function call.
    llvm::SmallVector<mlir::Value, 1> arguments;
    arguments.push_back(result);

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("zeros", std::nullopt, arguments), std::nullopt,
        arguments);

    rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    result =
        rewriter.create<mlir::memref::CastOp>(loc, resultMemRefType, result);

    result = convertToTensor(rewriter, result);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct PrintOpLowering : public RuntimeOpConversionPattern<PrintOp> {
  using RuntimeOpConversionPattern<PrintOp>::RuntimeOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(PrintOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value value = adaptor.getValue();

    // Collect the arguments for the function call.
    llvm::SmallVector<mlir::Value, 1> arguments;

    if (mlir::failed(addCallArgument(rewriter, loc, arguments, value))) {
      return mlir::failure();
    }

    // Create the call to the runtime library.
    auto callee = getOrDeclareRuntimeFunction(
        rewriter, op->getParentOfType<mlir::ModuleOp>(),
        getMangledFunctionName("print", std::nullopt, arguments), std::nullopt,
        arguments);

    rewriter.create<mlir::runtime::CallOp>(loc, callee, arguments);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};
} // namespace

namespace mlir {
void populateBaseModelicaToRuntimeCallConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context,
    mlir::TypeConverter &typeConverter,
    mlir::SymbolTableCollection &symbolTableCollection) {
  // Math operations.
  patterns.insert<PowOpLowering>(typeConverter, context, symbolTableCollection);

  // Built-in functions.
  patterns.insert<
      AbsOpLowering, AcosOpLowering, AsinOpLowering, AtanOpLowering,
      Atan2OpLowering, CeilOpLowering, CosOpLowering, CoshOpLowering,
      DiagonalOpLowering, DivTruncOpLowering, ExpOpLowering, FloorOpLowering,
      IdentityOpLowering, IntegerOpLowering, LinspaceOpLowering, LogOpLowering,
      Log10OpLowering, MaxOpScalarsLowering, MaxOpArrayLowering,
      MinOpScalarsLowering, MinOpArrayLowering, ModOpLowering, OnesOpLowering,
      ProductOpLowering, RemOpLowering, SignOpLowering, SinOpLowering,
      SinhOpLowering, SqrtOpLowering, SumOpLowering, SymmetricOpLowering,
      TanOpLowering, TanhOpLowering, TransposeOpLowering, ZerosOpLowering>(
      typeConverter, context, symbolTableCollection);

  // Utility operations.
  patterns.insert<PrintOpLowering>(typeConverter, context,
                                   symbolTableCollection);
}

std::unique_ptr<mlir::Pass> createBaseModelicaToRuntimeCallConversionPass() {
  return std::make_unique<BaseModelicaToRuntimeCallConversionPass>();
}
} // namespace mlir
