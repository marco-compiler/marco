#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/ModelicaConversion.h>
#include <modelica/mlirlowerer/passes/TypeConverter.h>
#include <modelica/mlirlowerer/CodeGen.h>
#include <numeric>

using namespace modelica::codegen;

/**
 * Generic conversion pattern that provides some utility functions.
 *
 * @tparam FromOp type of the operation to be converted
 */
template<typename FromOp>
class ModelicaOpConversion : public mlir::OpConversionPattern<FromOp> {
	protected:
	using Adaptor = typename FromOp::Adaptor;

	public:
	ModelicaOpConversion(mlir::MLIRContext* ctx, TypeConverter& typeConverter, ModelicaConversionOptions options)
			: mlir::OpConversionPattern<FromOp>(typeConverter, ctx, 1),
				options(std::move(options))
	{
	}

	[[nodiscard]] modelica::codegen::TypeConverter& typeConverter() const
	{
		return *static_cast<modelica::codegen::TypeConverter *>(this->getTypeConverter());
	}

	[[nodiscard]] mlir::Type convertType(mlir::Type type) const
	{
		return typeConverter().convertType(type);
	}

	[[nodiscard]] mlir::Value materializeTargetConversion(mlir::OpBuilder& builder, mlir::Value value) const
	{
		mlir::Type type = this->getTypeConverter()->convertType(value.getType());
		return this->getTypeConverter()->materializeTargetConversion(builder, value.getLoc(), type, value);
	}

	[[nodiscard]] bool isNumeric(mlir::Value value) const
	{
		return isNumericType(value.getType());
	}

	[[nodiscard]] bool isNumericType(mlir::Type type) const
	{
		return type.isa<mlir::IndexType, BooleanType, IntegerType, RealType>();
	}

	[[nodiscard]] llvm::SmallVector<mlir::Value, 3> getArrayDynamicDimensions(mlir::OpBuilder& builder, mlir::Location location, mlir::Value array) const
	{
		assert(array.getType().isa<PointerType>());
		auto pointerType = array.getType().cast<PointerType>();
		auto shape = pointerType.getShape();

		llvm::SmallVector<mlir::Value, 3> dimensions;

		for (size_t i = 0, e = shape.size(); i < e; ++i)
		{
			if (shape[i] == -1)
			{
				mlir::Value dim = builder.create<mlir::ConstantOp>(location, builder.getIndexAttr(i));
				dimensions.push_back(builder.create<DimOp>(location, array, dim));
			}
		}

		return dimensions;
	}

	/**
	 * Iterate over an array.
	 *
	 * @param builder   operation builder
	 * @param location  source location
	 * @param array     array to be iterated
	 * @param callback  function executed on each iteration
	 */
	void iterateArray(mlir::OpBuilder& builder, mlir::Location location, mlir::Value array, std::function<void(mlir::ValueRange)> callback) const
	{
		assert(array.getType().isa<PointerType>());
		auto pointerType = array.getType().cast<PointerType>();

		mlir::Value zero = builder.create<mlir::ConstantOp>(location, builder.getIndexAttr(0));
		mlir::Value one = builder.create<mlir::ConstantOp>(location, builder.getIndexAttr(1));

		llvm::SmallVector<mlir::Value, 3> lowerBounds(pointerType.getRank(), zero);
		llvm::SmallVector<mlir::Value, 3> upperBounds;
		llvm::SmallVector<mlir::Value, 3> steps(pointerType.getRank(), one);

		for (unsigned int i = 0, e = pointerType.getRank(); i < e; ++i)
		{
			mlir::Value dim = builder.create<mlir::ConstantOp>(location, builder.getIndexAttr(i));
			upperBounds.push_back(builder.create<DimOp>(location, array, dim));
		}

		// Create nested loops in order to iterate on each dimension of the array
		mlir::scf::buildLoopNest(
				builder, location, lowerBounds, upperBounds, steps, llvm::None,
				[&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange position, mlir::ValueRange args) -> std::vector<mlir::Value> {
					callback(position);
					return std::vector<mlir::Value>();
				});
	}

	[[nodiscard]] mlir::Value allocate(mlir::OpBuilder& builder, mlir::Location location, PointerType pointerType, mlir::ValueRange dynamicDimensions = llvm::None, bool shouldBeFreed = true) const
	{
		if (pointerType.getAllocationScope() == unknown)
			pointerType = pointerType.toMinAllowedAllocationScope();

		if (pointerType.getAllocationScope() == stack)
			return builder.create<AllocaOp>(location, pointerType.getElementType(), pointerType.getShape(), dynamicDimensions);

		return builder.create<AllocOp>(location, pointerType.getElementType(), pointerType.getShape(), dynamicDimensions, shouldBeFreed);
	}

	[[nodiscard]] mlir::FuncOp getOrDeclareFunction(mlir::OpBuilder& builder, mlir::ModuleOp module, llvm::StringRef name, mlir::TypeRange results, mlir::ValueRange args) const
	{
		return getOrDeclareFunction(builder, module, name, results, args.getTypes());
	}

	[[nodiscard]] mlir::FuncOp getOrDeclareFunction(mlir::OpBuilder& builder, mlir::ModuleOp module, llvm::StringRef name, mlir::TypeRange results, mlir::TypeRange args) const
	{
		if (auto foo = module.lookupSymbol<mlir::FuncOp>(name))
			return foo;

		mlir::PatternRewriter::InsertionGuard insertGuard(builder);
		builder.setInsertionPointToStart(module.getBody());
		auto foo = builder.create<mlir::FuncOp>(module.getLoc(), name, builder.getFunctionType(args, results));
		foo.setPrivate();
		return foo;
	}

	[[nodiscard]] std::string getMangledFunctionName(llvm::StringRef name, mlir::ValueRange args) const
	{
		return getMangledFunctionName(name, args.getTypes());
	}

	[[nodiscard]] std::string getMangledFunctionName(llvm::StringRef name, mlir::TypeRange types) const
	{
		return "_M" + name.str() + std::accumulate(
				types.begin(), types.end(), std::string(),
				[&](const std::string& result, mlir::Type type) {
					return result + "_" + getMangledType(type);
				});
	}

	private:
	[[nodiscard]] std::string getMangledType(mlir::Type type) const
	{
		if (auto booleanType = type.dyn_cast<BooleanType>())
			return "i1";

		if (auto integerType = type.dyn_cast<IntegerType>())
			return "i" + std::to_string(integerType.getBitWidth());

		if (auto realType = type.dyn_cast<RealType>())
			return "f" + std::to_string(realType.getBitWidth());

		if (auto pointerType = type.dyn_cast<UnsizedPointerType>())
			return "a" + getMangledType(pointerType.getElementType());

		if (auto indexType = type.dyn_cast<mlir::IndexType>())
			return getMangledType(this->getTypeConverter()->convertType(type));

		if (auto integerType = type.dyn_cast<mlir::IntegerType>())
			return "i" + std::to_string(integerType.getWidth());

		if (auto floatType = type.dyn_cast<mlir::FloatType>())
			return "f" + std::to_string(floatType.getWidth());

		return "unknown";
	}

	protected:
	ModelicaConversionOptions options;
};

struct ConstantOpLowering: public ModelicaOpConversion<ConstantOp>
{
	using ModelicaOpConversion<ConstantOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(ConstantOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Type resultType = getTypeConverter()->convertType(op.resultType());
		auto attribute = convertAttribute(rewriter, resultType, op.value());

		if (!attribute)
			return rewriter.notifyMatchFailure(op, "Unknown attribute type");

		rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(op, resultType, *attribute);
		return mlir::success();
	}

	private:
	llvm::Optional<mlir::Attribute> convertAttribute(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Attribute attribute) const
	{
		if (attribute.getType().isa<mlir::IndexType>())
			return attribute;

		if (auto booleanAttribute = attribute.dyn_cast<BooleanAttribute>())
			return builder.getBoolAttr(booleanAttribute.getValue());

		if (auto integerAttribute = attribute.dyn_cast<IntegerAttribute>())
			return builder.getIntegerAttr(resultType, integerAttribute.getValue());

		if (auto realAttribute = attribute.dyn_cast<RealAttribute>())
			return builder.getFloatAttr(resultType, realAttribute.getValue());

		return llvm::None;
	}
};

struct PackOpLowering: public ModelicaOpConversion<PackOp>
{
	using ModelicaOpConversion<PackOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(PackOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();
		Adaptor transformed(operands);

		StructType structType = op.resultType();
		mlir::Type descriptorType = getTypeConverter()->convertType(structType);

		mlir::Value result = rewriter.create<mlir::LLVM::UndefOp>(loc, descriptorType);

		for (auto& element : llvm::enumerate(transformed.values()))
			result = rewriter.create<mlir::LLVM::InsertValueOp>(
					loc, descriptorType, result, element.value(), rewriter.getIndexArrayAttr(element.index()));

		rewriter.replaceOp(op, result);
		return mlir::success();
	}
};

struct ExtractOpLowering: public ModelicaOpConversion<ExtractOp>
{
	using ModelicaOpConversion<ExtractOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(ExtractOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		Adaptor transformed(operands);

		mlir::Type descriptorType = getTypeConverter()->convertType(op.packedValue().getType());
		mlir::Type type = descriptorType.cast<mlir::LLVM::LLVMStructType>().getBody()[op.index()];
		mlir::Value result = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, type, transformed.packedValue(), rewriter.getIndexArrayAttr(op.index()));
		result = getTypeConverter()->materializeSourceConversion(rewriter, result.getLoc(), op.resultType(), result);
		rewriter.replaceOp(op, result);
		return mlir::success();
	}
};

/**
 * Store a scalar value.
 */
struct AssignmentOpScalarLowering: public ModelicaOpConversion<AssignmentOp>
{
	using ModelicaOpConversion<AssignmentOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(AssignmentOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		if (!isNumeric(op.source()))
			return rewriter.notifyMatchFailure(op, "Source value has not a numeric type");

		auto destinationBaseType = op.destination().getType().cast<PointerType>().getElementType();
		mlir::Value value = rewriter.create<CastOp>(loc, op.source(), destinationBaseType);
		rewriter.replaceOpWithNewOp<StoreOp>(op, value, op.destination());

		return mlir::success();
	}
};

/**
 * Store (copy) an array value.
 */
struct AssignmentOpArrayLowering: public ModelicaOpConversion<AssignmentOp>
{
	using ModelicaOpConversion<AssignmentOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(AssignmentOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		if (!op.source().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Source value is not an array");

		iterateArray(rewriter, op.getLoc(), op.source(),
								 [&](mlir::ValueRange position) {
									 mlir::Value value = rewriter.create<LoadOp>(loc, op.source(), position);
									 value = rewriter.create<CastOp>(value.getLoc(), value, op.destination().getType().cast<PointerType>().getElementType());
									 rewriter.create<StoreOp>(loc, value, op.destination(), position);
								 });

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct CallOpLowering: public ModelicaOpConversion<CallOp>
{
	using ModelicaOpConversion<CallOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(CallOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, op.callee(), op->getResultTypes(), op.args());
		return mlir::success();
	}
};

struct PrintOpLowering: public ModelicaOpConversion<PrintOp>
{
	using ModelicaOpConversion<PrintOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(PrintOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto module = op->getParentOfType<mlir::ModuleOp>();

		auto printfRef = getOrInsertPrintf(rewriter, module);
		mlir::Value semicolonCst = getOrCreateGlobalString(loc, rewriter, "semicolon", mlir::StringRef(";\0", 2), module);
		mlir::Value newLineCst = getOrCreateGlobalString(loc, rewriter, "newline", mlir::StringRef("\n\0", 2), module);

		mlir::Value printSeparator = rewriter.create<AllocaOp>(loc, BooleanType::get(op->getContext()));
		mlir::Value falseValue = rewriter.create<ConstantOp>(loc, BooleanAttribute::get(BooleanType::get(op->getContext()), false));
		rewriter.create<StoreOp>(loc, falseValue, printSeparator);

		for (auto pair : llvm::zip(op.values(), Adaptor(operands).values()))
		{
			mlir::Value source = std::get<0>(pair);
			mlir::Value transformed = std::get<1>(pair);

			if (auto pointerType = source.getType().dyn_cast<PointerType>())
			{
				if (pointerType.getRank() > 0)
				{
					iterateArray(rewriter, source.getLoc(), source,
											 [&](mlir::ValueRange position) {
												 mlir::Value value = rewriter.create<LoadOp>(source.getLoc(), source, position);
												 value = materializeTargetConversion(rewriter, value);
												 printElement(rewriter, value, printSeparator, semicolonCst, module);
											 });
				}
				else
				{
					mlir::Value value = rewriter.create<LoadOp>(source.getLoc(), source);
					value = materializeTargetConversion(rewriter, value);
					printElement(rewriter, value, printSeparator, semicolonCst, module);
				}
			}
			else
			{
				printElement(rewriter, transformed, printSeparator, semicolonCst, module);
			}
		}

		rewriter.create<mlir::LLVM::CallOp>(op.getLoc(), printfRef, newLineCst);

		rewriter.eraseOp(op);
		return mlir::success();
	}

	mlir::Value getOrCreateGlobalString(mlir::Location loc, mlir::OpBuilder& builder, mlir::StringRef name, mlir::StringRef value, mlir::ModuleOp module) const
	{
		// Create the global at the entry of the module
		mlir::LLVM::GlobalOp global;

		if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name)))
		{
			mlir::OpBuilder::InsertionGuard insertGuard(builder);
			builder.setInsertionPointToStart(module.getBody());
			auto type = mlir::LLVM::LLVMArrayType::get(mlir::IntegerType::get(builder.getContext(), 8), value.size());
			global = builder.create<mlir::LLVM::GlobalOp>(loc, type, true, mlir::LLVM::Linkage::Internal, name, builder.getStringAttr(value));
		}

		// Get the pointer to the first character in the global string
		mlir::Value globalPtr = builder.create<mlir::LLVM::AddressOfOp>(loc, global);

		mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
				loc,
				mlir::IntegerType::get(builder.getContext(), 64),
				builder.getIntegerAttr(builder.getIndexType(), 0));

		return builder.create<mlir::LLVM::GEPOp>(
				loc,
				mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(builder.getContext(), 8)),
				globalPtr, llvm::ArrayRef<mlir::Value>({cst0, cst0}));
	}

	mlir::LLVM::LLVMFuncOp getOrInsertPrintf(mlir::OpBuilder& rewriter, mlir::ModuleOp module) const
	{
		auto *context = module.getContext();

		if (auto foo = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf"))
			return foo;

		// Create a function declaration for printf, the signature is:
		//   * `i32 (i8*, ...)`
		auto llvmI32Ty = mlir::IntegerType::get(context, 32);
		auto llvmI8PtrTy = mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(context, 8));
		auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy, true);

		// Insert the printf function into the body of the parent module.
		mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
		rewriter.setInsertionPointToStart(module.getBody());
		return rewriter.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
	}

	void printElement(mlir::OpBuilder& builder, mlir::Value value, mlir::Value printSeparator, mlir::Value separator, mlir::ModuleOp module) const
	{
		auto printfRef = getOrInsertPrintf(builder, module);

		mlir::Type type = value.getType();

		// Check if the separator should be printed
		mlir::Value shouldPrintSeparator = builder.create<LoadOp>(printSeparator.getLoc(), printSeparator);
		shouldPrintSeparator = materializeTargetConversion(builder, shouldPrintSeparator);
		auto ifOp = builder.create<mlir::scf::IfOp>(value.getLoc(), shouldPrintSeparator);
		builder.setInsertionPointToStart(ifOp.getBody());
		builder.create<mlir::LLVM::CallOp>(value.getLoc(), printfRef, separator);
		builder.setInsertionPointAfter(ifOp);

		mlir::Value formatSpecifier;

		if (type.isa<mlir::IntegerType>())
			formatSpecifier = getOrCreateGlobalString(value.getLoc(), builder, "frmt_spec_int", mlir::StringRef("%ld\0", 4), module);
		else if (type.isa<mlir::FloatType>())
			formatSpecifier = getOrCreateGlobalString(value.getLoc(), builder, "frmt_spec_float", mlir::StringRef("%.12f\0", 6), module);
		else
			assert(false && "Unknown type");

		builder.create<mlir::LLVM::CallOp>(value.getLoc(), printfRef, mlir::ValueRange({ formatSpecifier, value }));

		// Set the separator as to be printed before the next value
		mlir::Value trueValue = builder.create<ConstantOp>(value.getLoc(), BooleanAttribute::get(BooleanType::get(builder.getContext()), true));
		builder.create<StoreOp>(value.getLoc(), trueValue, printSeparator);
	}
};

struct ArrayCloneOpLowering: public ModelicaOpConversion<ArrayCloneOp>
{
	using ModelicaOpConversion<ArrayCloneOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(ArrayCloneOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		Adaptor adaptor(operands);

		auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.source());
		mlir::Value result = allocate(rewriter, loc, op.resultType(), dynamicDimensions, op.shouldBeFreed());

		iterateArray(rewriter, loc, op.source(), [&](mlir::ValueRange indexes) {
			mlir::Value value = rewriter.create<LoadOp>(loc, op.source(), indexes);
			rewriter.create<StoreOp>(loc, value, result, indexes);
		});

		rewriter.replaceOp(op, result);
		return mlir::success();
	}
};

struct NotOpScalarLowering: public ModelicaOpConversion<NotOp>
{
	using ModelicaOpConversion<NotOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(NotOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();
		Adaptor transformed(operands);

		// Check if the operand is compatible
		if (!op.operand().getType().isa<BooleanType>())
			return rewriter.notifyMatchFailure(op, "Operand is not a boolean");

		// There is no native negate operation in LLVM IR, so we need to leverage
		// a property of the XOR operation: x XOR true = NOT x
		mlir::Value trueValue = rewriter.create<mlir::ConstantOp>(loc, rewriter.getBoolAttr(true));
		mlir::Value result = rewriter.create<mlir::XOrOp>(loc, trueValue, transformed.operand());
		result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(op->getContext()), result);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());

		return mlir::success();
	}
};

struct NotOpArrayLowering: public ModelicaOpConversion<NotOp>
{
	using ModelicaOpConversion<NotOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(NotOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();
		Adaptor transformed(operands);

		// Check if the operand is compatible
		if (!op.operand().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Operand is not an array");

		if (auto pointerType = op.operand().getType().cast<PointerType>(); !pointerType.getElementType().isa<BooleanType>())
			return rewriter.notifyMatchFailure(op, "Operand is not an array of booleans");

		// Allocate the result array
		auto resultType = op.resultType().cast<PointerType>();
		auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.operand());
		mlir::Value result = allocate(rewriter, loc, resultType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		// Apply the operation on each array position
		iterateArray(rewriter, loc, op.operand(),
								 [&](mlir::ValueRange position) {
									 mlir::Value value = rewriter.create<LoadOp>(loc, op.operand(), position);
									 mlir::Value negated = rewriter.create<NotOp>(loc, resultType.getElementType(), value);
									 rewriter.create<StoreOp>(loc, negated, result, position);
								 });

		return mlir::success();
	}
};

struct AndOpScalarLowering: public ModelicaOpConversion<AndOp>
{
	using ModelicaOpConversion<AndOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(AndOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();
		Adaptor transformed(operands);

		// Check if the operands are compatible
		if (!op.lhs().getType().isa<BooleanType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side operand is not a boolean");

		if (!op.rhs().getType().isa<BooleanType>())
			return rewriter.notifyMatchFailure(op, "Right-hand side operand is not a boolean");

		// Compute the result
		mlir::Value result = rewriter.create<mlir::AndOp>(loc, transformed.lhs(), transformed.rhs());
		result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(op->getContext()), result);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());

		return mlir::success();
	}
};

struct AndOpArrayLowering: public ModelicaOpConversion<AndOp>
{
	using ModelicaOpConversion<AndOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(AndOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		// Check if the operands are compatible
		if (!op.lhs().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side operand is not an array");

		if (auto pointerType = op.lhs().getType().cast<PointerType>(); !pointerType.getElementType().isa<BooleanType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side operand is not an array of booleans");

		if (!op.rhs().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Right-hand side operand is not an array");

		if (auto pointerType = op.rhs().getType().cast<PointerType>(); !pointerType.getElementType().isa<BooleanType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side operand is not an array of booleans");

		// Allocate the result array
		auto resultType = op.resultType().cast<PointerType>();
		auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.lhs());
		mlir::Value result = allocate(rewriter, loc, resultType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		// Apply the operation on each array position
		iterateArray(rewriter, op.getLoc(), result,
								 [&](mlir::ValueRange position) {
									 mlir::Value lhs = rewriter.create<LoadOp>(loc, op.lhs(), position);
									 mlir::Value rhs = rewriter.create<LoadOp>(loc, op.rhs(), position);
									 mlir::Value scalarResult = rewriter.create<AndOp>(loc, resultType.getElementType(), lhs, rhs);
									 rewriter.create<StoreOp>(loc, scalarResult, result, position);
								 });

		return mlir::success();
	}
};

struct OrOpScalarLowering: public ModelicaOpConversion<OrOp>
{
	using ModelicaOpConversion<OrOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(OrOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		Adaptor transformed(operands);

		// Check if the operands are compatible
		if (!op.lhs().getType().isa<BooleanType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side operand is not a boolean");

		if (!op.rhs().getType().isa<BooleanType>())
			return rewriter.notifyMatchFailure(op, "Right-hand side operand is not a boolean");

		// Compute the result
		mlir::Value result = rewriter.create<mlir::OrOp>(loc, transformed.lhs(), transformed.rhs());
		result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(op->getContext()), result);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());

		return mlir::success();
	}
};

struct OrOpArrayLowering: public ModelicaOpConversion<OrOp>
{
	using ModelicaOpConversion<OrOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(OrOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		// Check if the operands are compatible
		if (!op.lhs().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side operand is not an array");

		if (auto pointerType = op.lhs().getType().cast<PointerType>(); !pointerType.getElementType().isa<BooleanType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side operand is not an array of booleans");

		if (!op.rhs().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Right-hand side operand is not an array");

		if (auto pointerType = op.rhs().getType().cast<PointerType>(); !pointerType.getElementType().isa<BooleanType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side operand is not an array of booleans");

		// Allocate the result array
		auto resultType = op.resultType().cast<PointerType>();
		auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.lhs());
		mlir::Value result = allocate(rewriter, loc, resultType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		// Apply the operation on each array position
		iterateArray(rewriter, loc, result,
								 [&](mlir::ValueRange position) {
									 mlir::Value lhs = rewriter.create<LoadOp>(loc, op.lhs(), position);
									 mlir::Value rhs = rewriter.create<LoadOp>(loc, op.rhs(), position);
									 mlir::Value scalarResult = rewriter.create<OrOp>(loc, resultType.getElementType(), lhs, rhs);
									 rewriter.create<StoreOp>(loc, scalarResult, result, position);
								 });

		return mlir::success();
	}
};

struct EqOpLowering: public ModelicaOpConversion<EqOp>
{
	using ModelicaOpConversion<EqOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(EqOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		// Check if the operands are compatible
		if (!isNumeric(op.lhs()) || !isNumeric(op.rhs()))
			return rewriter.notifyMatchFailure(op, "Unsupported types");

		// Cast the operands to the most generic type, in order to avoid
		// information loss.
		auto castOp = rewriter.create<CastCommonOp>(loc, op->getOperands());
		llvm::SmallVector<mlir::Value, 3> castedOperands;

		for (const auto& operand : castOp.getResults())
			castedOperands.push_back(materializeTargetConversion(rewriter, operand));

		mlir::Type type = castOp.resultType();
		Adaptor transformed(castedOperands);

		// Compute the result
		if (type.isa<mlir::IndexType, BooleanType, IntegerType>())
		{
			mlir::Value result = rewriter.create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::eq, transformed.lhs(), transformed.rhs());
			result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(op->getContext()), result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		if (type.isa<RealType>())
		{
			mlir::Value result = rewriter.create<mlir::LLVM::FCmpOp>(loc, mlir::LLVM::FCmpPredicate::oeq, transformed.lhs(), transformed.rhs());
			result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(op->getContext()), result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		return mlir::failure();
	}
};

struct NotEqOpLowering: public ModelicaOpConversion<NotEqOp>
{
	using ModelicaOpConversion<NotEqOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(NotEqOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		// Check if the operands are compatible
		if (!isNumeric(op.lhs()) || !isNumeric(op.rhs()))
			return rewriter.notifyMatchFailure(op, "Unsupported types");

		// Cast the operands to the most generic type, in order to avoid
		// information loss.
		auto castOp = rewriter.create<CastCommonOp>(loc, op->getOperands());
		llvm::SmallVector<mlir::Value, 3> castedOperands;

		for (const auto& operand : castOp.getResults())
			castedOperands.push_back(materializeTargetConversion(rewriter, operand));

		Adaptor transformed(castedOperands);
		mlir::Type type = castOp.resultType();

		// Compute the result
		if (type.isa<mlir::IndexType, BooleanType, IntegerType>())
		{
			mlir::Value result = rewriter.create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::ne, transformed.lhs(), transformed.rhs());
			result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(op->getContext()), result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		if (type.isa<RealType>())
		{
			mlir::Value result = rewriter.create<mlir::LLVM::FCmpOp>(loc, mlir::LLVM::FCmpPredicate::one, transformed.lhs(), transformed.rhs());
			result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(op->getContext()), result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		return mlir::failure();
	}
};

struct GtOpLowering: public ModelicaOpConversion<GtOp>
{
	using ModelicaOpConversion<GtOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(GtOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		// Check if the operands are compatible
		if (!isNumeric(op.lhs()) || !isNumeric(op.rhs()))
			return rewriter.notifyMatchFailure(op, "Unsupported types");

		// Cast the operands to the most generic type, in order to avoid
		// information loss.
		auto castOp = rewriter.create<CastCommonOp>(loc, op->getOperands());
		llvm::SmallVector<mlir::Value, 3> castedOperands;

		for (const auto& operand : castOp.getResults())
			castedOperands.push_back(materializeTargetConversion(rewriter, operand));

		Adaptor transformed(castedOperands);
		mlir::Type type = castOp.resultType();

		// Compute the result
		if (type.isa<mlir::IndexType, BooleanType, IntegerType>())
		{
			mlir::Value result = rewriter.create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::sgt, transformed.lhs(), transformed.rhs());
			result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(op->getContext()), result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		if (type.isa<RealType>())
		{
			mlir::Value result = rewriter.create<mlir::LLVM::FCmpOp>(loc, mlir::LLVM::FCmpPredicate::ogt, transformed.lhs(), transformed.rhs());
			result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(op->getContext()), result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		return mlir::failure();
	}
};

struct GteOpLowering: public ModelicaOpConversion<GteOp>
{
	using ModelicaOpConversion<GteOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(GteOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		// Check if the operands are compatible
		if (!isNumeric(op.lhs()) || !isNumeric(op.rhs()))
			return rewriter.notifyMatchFailure(op, "Unsupported types");

		// Cast the operands to the most generic type, in order to avoid
		// information loss.
		auto castOp = rewriter.create<CastCommonOp>(loc, op->getOperands());
		llvm::SmallVector<mlir::Value, 3> castedOperands;

		for (const auto& operand : castOp.getResults())
			castedOperands.push_back(materializeTargetConversion(rewriter, operand));

		Adaptor transformed(castedOperands);
		mlir::Type type = castOp.resultType();

		// Compute the result
		if (type.isa<mlir::IndexType, BooleanType, IntegerType>())
		{
			mlir::Value result = rewriter.create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::sge, transformed.lhs(), transformed.rhs());
			result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(op->getContext()), result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		if (type.isa<RealType>())
		{
			mlir::Value result = rewriter.create<mlir::LLVM::FCmpOp>(loc, mlir::LLVM::FCmpPredicate::oge, transformed.lhs(), transformed.rhs());
			result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(op->getContext()), result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		return mlir::failure();
	}
};

struct LtOpLowering: public ModelicaOpConversion<LtOp>
{
	using ModelicaOpConversion<LtOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(LtOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		// Check if the operands are compatible
		if (!isNumeric(op.lhs()) || !isNumeric(op.rhs()))
			return rewriter.notifyMatchFailure(op, "Unsupported types");

		// Cast the operands to the most generic type, in order to avoid
		// information loss.
		auto castOp = rewriter.create<CastCommonOp>(loc, op->getOperands());
		llvm::SmallVector<mlir::Value, 3> castedOperands;

		for (const auto& operand : castOp.getResults())
			castedOperands.push_back(materializeTargetConversion(rewriter, operand));

		Adaptor transformed(castedOperands);
		mlir::Type type = castOp.resultType();

		// Compute the result
		if (type.isa<mlir::IndexType, BooleanType, IntegerType>())
		{
			mlir::Value result = rewriter.create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, transformed.lhs(), transformed.rhs());
			result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(op->getContext()), result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		if (type.isa<RealType>())
		{
			mlir::Value result = rewriter.create<mlir::LLVM::FCmpOp>(loc, mlir::LLVM::FCmpPredicate::olt, transformed.lhs(), transformed.rhs());
			result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(op->getContext()), result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		return mlir::failure();
	}
};

struct LteOpLowering: public ModelicaOpConversion<LteOp>
{
	using ModelicaOpConversion<LteOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(LteOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		// Check if the operands are compatible
		if (!isNumeric(op.lhs()) || !isNumeric(op.rhs()))
			return rewriter.notifyMatchFailure(op, "Unsupported types");

		// Cast the operands to the most generic type, in order to avoid
		// information loss.
		auto castOp = rewriter.create<CastCommonOp>(loc, op->getOperands());
		llvm::SmallVector<mlir::Value, 3> castedOperands;

		for (const auto& operand : castOp.getResults())
			castedOperands.push_back(materializeTargetConversion(rewriter, operand));

		Adaptor transformed(castedOperands);
		mlir::Type type = castOp.resultType();

		// Compute the result
		if (type.isa<mlir::IndexType, BooleanType, IntegerType>())
		{
			mlir::Value result = rewriter.create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::sle, transformed.lhs(), transformed.rhs());
			result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(op->getContext()), result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		if (type.isa<RealType>())
		{
			mlir::Value result = rewriter.create<mlir::LLVM::FCmpOp>(loc, mlir::LLVM::FCmpPredicate::ole, transformed.lhs(), transformed.rhs());
			result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(op->getContext()), result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		return mlir::failure();
	}
};

/**
 * Negate a scalar value.
 */
struct NegateOpScalarLowering: public ModelicaOpConversion<NegateOp>
{
	using ModelicaOpConversion<NegateOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(NegateOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		// Check if the operand is compatible
		if (!isNumeric(op.operand()))
			return rewriter.notifyMatchFailure(op, "Left-hand side value is not a scalar");

		Adaptor transformed(operands);
		mlir::Type type = op.operand().getType();

		// Compute the result
		if (type.isa<mlir::IndexType, BooleanType, IntegerType>())
		{
			mlir::Value zeroValue = rewriter.create<mlir::ConstantOp>(loc, rewriter.getZeroAttr(transformed.operand().getType()));
			mlir::Value result = rewriter.create<mlir::SubIOp>(loc, zeroValue, transformed.operand());
			result = getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		if (type.isa<RealType>())
		{
			mlir::Value result = rewriter.create<mlir::NegFOp>(loc, transformed.operand());
			result = getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		return rewriter.notifyMatchFailure(op, "Unknown type");
	}
};

/**
 * Negate an array.
 */
struct NegateOpArrayLowering: public ModelicaOpConversion<NegateOp>
{
	using ModelicaOpConversion<NegateOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(NegateOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		// Check if the operand is compatible
		if (!op.operand().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Value is not an array");

		auto pointerType = op.operand().getType().cast<PointerType>();

		if (!isNumericType(pointerType.getElementType()))
			return rewriter.notifyMatchFailure(op, "Array has not numeric elements");

		// Allocate the result array
		auto resultType = op.resultType().cast<PointerType>();
		auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.operand());
		mlir::Value result = allocate(rewriter, loc, resultType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		// Apply the operation on each array position
		iterateArray(rewriter, loc, op.operand(),
								 [&](mlir::ValueRange position) {
									 mlir::Value source = rewriter.create<LoadOp>(loc, op.operand(), position);
									 mlir::Value value = rewriter.create<NegateOp>(loc, resultType.getElementType(), source);
									 rewriter.create<StoreOp>(loc, value, result, position);
								 });

		return mlir::success();
	}
};

/**
 * Sum of two numeric scalars.
 */
struct AddOpScalarLowering: public ModelicaOpConversion<AddOp>
{
	using ModelicaOpConversion<AddOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(AddOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		// Check if the operands are compatible
		if (!isNumeric(op.lhs()))
			return rewriter.notifyMatchFailure(op, "Left-hand side value is not a scalar");

		if (!isNumeric(op.rhs()))
			return rewriter.notifyMatchFailure(op, "Right-hand side value is not a scalar");

		// Cast the operands to the most generic type
		auto castOp = rewriter.create<CastCommonOp>(loc, op->getOperands());
		llvm::SmallVector<mlir::Value, 3> castedOperands;

		for (mlir::Value operand : castOp.getResults())
			castedOperands.push_back(materializeTargetConversion(rewriter, operand));

		Adaptor transformed(castedOperands);
		mlir::Type type = castOp.resultType();

		// Compute the result
		if (type.isa<mlir::IndexType, BooleanType, IntegerType>())
		{
			mlir::Value result = rewriter.create<mlir::AddIOp>(loc, transformed.lhs(), transformed.rhs());
			result = getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		if (type.isa<RealType>())
		{
			mlir::Value result = rewriter.create<mlir::AddFOp>(loc, transformed.lhs(), transformed.rhs());
			result = getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		return rewriter.notifyMatchFailure(op, "Unknown type");
	}
};

/**
 * Sum of two numeric arrays.
 */
struct AddOpArrayLowering: public ModelicaOpConversion<AddOp>
{
	using ModelicaOpConversion<AddOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(AddOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		// Check if the operands are compatible
		if (!op.lhs().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side value is not an array");

		if (!op.rhs().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Right-hand side value is not an array");

		auto lhsPointerType = op.lhs().getType().cast<PointerType>();
		auto rhsPointerType = op.rhs().getType().cast<PointerType>();

		for (auto pair : llvm::zip(lhsPointerType.getShape(), rhsPointerType.getShape()))
		{
			auto lhsDimension = std::get<0>(pair);
			auto rhsDimension = std::get<1>(pair);

			if (lhsDimension != -1 && rhsDimension != -1 && lhsDimension != rhsDimension)
				return rewriter.notifyMatchFailure(op, "Incompatible array dimensions");
		}

		if (!isNumericType(lhsPointerType.getElementType()))
			return rewriter.notifyMatchFailure(op, "Left-hand side array has not numeric elements");

		if (!isNumericType(rhsPointerType.getElementType()))
			return rewriter.notifyMatchFailure(op, "Right-hand side array has not numeric elements");

		// Allocate the result array
		auto resultType = op.resultType().cast<PointerType>();
		auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.lhs());
		mlir::Value result = allocate(rewriter, loc, resultType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		// Apply the operation on each array position
		iterateArray(rewriter, loc, op.lhs(),
								 [&](mlir::ValueRange position) {
									 mlir::Value lhs = rewriter.create<LoadOp>(loc, op.lhs(), position);
									 mlir::Value rhs = rewriter.create<LoadOp>(loc, op.rhs(), position);
									 mlir::Value value = rewriter.create<AddOp>(loc, resultType.getElementType(), lhs, rhs);
									 rewriter.create<StoreOp>(loc, value, result, position);
								 });

		return mlir::success();
	}
};

/**
 * Subtraction of two numeric scalars.
 */
struct SubOpScalarLowering: public ModelicaOpConversion<SubOp>
{
	using ModelicaOpConversion<SubOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(SubOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		// Check if the operands are compatible
		if (!isNumeric(op.lhs()))
			return rewriter.notifyMatchFailure(op, "Left-hand side value is not a scalar");

		if (!isNumeric(op.rhs()))
			return rewriter.notifyMatchFailure(op, "Right-hand side value is not a scalar");

		// Cast the operands to the most generic type
		auto castOp = rewriter.create<CastCommonOp>(loc, op->getOperands());
		llvm::SmallVector<mlir::Value, 3> castedOperands;

		for (mlir::Value operand : castOp.getResults())
			castedOperands.push_back(materializeTargetConversion(rewriter, operand));

		Adaptor transformed(castedOperands);
		mlir::Type type = castOp.resultType();

		// Compute the result
		if (type.isa<mlir::IndexType, BooleanType, IntegerType>())
		{
			mlir::Value result = rewriter.create<mlir::SubIOp>(loc, transformed.lhs(), transformed.rhs());
			result = getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		if (type.isa<RealType>())
		{
			mlir::Value result = rewriter.create<mlir::SubFOp>(loc, transformed.lhs(), transformed.rhs());
			result = getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		return rewriter.notifyMatchFailure(op, "Unknown type");
	}
};

/**
 * Subtraction of two numeric arrays.
 */
struct SubOpArrayLowering: public ModelicaOpConversion<SubOp>
{
	using ModelicaOpConversion<SubOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(SubOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		// Check if the operands are compatible
		if (!op.lhs().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side value is not an array");

		if (!op.rhs().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Right-hand side value is not an array");

		auto lhsPointerType = op.lhs().getType().cast<PointerType>();
		auto rhsPointerType = op.rhs().getType().cast<PointerType>();

		for (auto pair : llvm::zip(lhsPointerType.getShape(), rhsPointerType.getShape()))
		{
			auto lhsDimension = std::get<0>(pair);
			auto rhsDimension = std::get<1>(pair);

			if (lhsDimension != -1 && rhsDimension != -1 && lhsDimension != rhsDimension)
				return rewriter.notifyMatchFailure(op, "Incompatible array dimensions");
		}

		if (!isNumericType(lhsPointerType.getElementType()))
			return rewriter.notifyMatchFailure(op, "Left-hand side array has not numeric elements");

		if (!isNumericType(rhsPointerType.getElementType()))
			return rewriter.notifyMatchFailure(op, "Right-hand side array has not numeric elements");

		// Allocate the result array
		auto resultType = op.resultType().cast<PointerType>();
		auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.lhs());
		mlir::Value result = allocate(rewriter, loc, resultType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		// Apply the operation on each array position
		iterateArray(rewriter, loc, op.lhs(),
								 [&](mlir::ValueRange position) {
									 mlir::Value lhs = rewriter.create<LoadOp>(loc, op.lhs(), position);
									 mlir::Value rhs = rewriter.create<LoadOp>(loc, op.rhs(), position);
									 mlir::Value value = rewriter.create<SubOp>(loc, resultType.getElementType(), lhs, rhs);
									 rewriter.create<StoreOp>(loc, value, result, position);
								 });

		return mlir::success();
	}
};

/**
 * Product between two scalar values.
 */
struct MulOpLowering: public ModelicaOpConversion<MulOp>
{
	using ModelicaOpConversion<MulOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(MulOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		// Check if the operands are compatible
		if (!isNumeric(op.lhs()))
			return rewriter.notifyMatchFailure(op, "Scalar-scalar product: left-hand side value is not a scalar");

		if (!isNumeric(op.rhs()))
			return rewriter.notifyMatchFailure(op, "Scalar-scalar product: right-hand side value is not a scalar");

		// Cast the operands to the most generic type, in order to avoid
		// information loss.
		auto castOp = rewriter.create<CastCommonOp>(loc, op->getOperands());
		llvm::SmallVector<mlir::Value, 3> castedOperands;

		for (mlir::Value operand : castOp.getResults())
			castedOperands.push_back(materializeTargetConversion(rewriter, operand));

		Adaptor transformed(castedOperands);
		mlir::Type type = castOp.resultType();

		// Compute the result
		if (type.isa<mlir::IndexType, BooleanType, IntegerType>())
		{
			mlir::Value result = rewriter.create<mlir::MulIOp>(loc, transformed.lhs(), transformed.rhs());
			result = getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		if (type.isa<RealType>())
		{
			mlir::Value result = rewriter.create<mlir::MulFOp>(loc, transformed.lhs(), transformed.rhs());
			result = getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		return rewriter.notifyMatchFailure(op, "Unknown type");
	}
};

/**
 * Product between a scalar and an array.
 */
struct MulOpScalarProductLowering: public ModelicaOpConversion<MulOp>
{
	using ModelicaOpConversion<MulOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(MulOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		// Check if the operands are compatible
		if (!isNumeric(op.lhs()) && !isNumeric(op.rhs()))
			return rewriter.notifyMatchFailure(op, "Scalar-array product: none of the operands is a scalar");

		if (isNumeric(op.lhs()))
		{
			if (!op.rhs().getType().isa<PointerType>())
				return rewriter.notifyMatchFailure(op, "Scalar-array product: right-hand size value is not an array");

			if (!isNumericType(op.rhs().getType().cast<PointerType>().getElementType()))
				return rewriter.notifyMatchFailure(op, "Scalar-array product: right-hand side array has not numeric elements");
		}

		if (isNumeric(op.rhs()))
		{
			if (!op.lhs().getType().isa<PointerType>())
				return rewriter.notifyMatchFailure(op, "Scalar-array product: right-hand size value is not an array");

			if (!isNumericType(op.lhs().getType().cast<PointerType>().getElementType()))
				return rewriter.notifyMatchFailure(op, "Scalar-array product: left-hand side array has not numeric elements");
		}

		mlir::Value scalar = isNumeric(op.lhs()) ? op.lhs() : op.rhs();
		mlir::Value array = isNumeric(op.rhs()) ? op.lhs() : op.rhs();

		// Allocate the result array
		auto resultType = op.resultType().cast<PointerType>();
		auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, array);
		mlir::Value result = allocate(rewriter, loc, resultType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		// Multiply each array element by the scalar value
		iterateArray(rewriter, loc, array,
								 [&](mlir::ValueRange position) {
									 mlir::Value arrayValue = rewriter.create<LoadOp>(loc, array, position);
									 mlir::Value value = rewriter.create<MulOp>(loc, resultType.getElementType(), scalar, arrayValue);
									 rewriter.create<StoreOp>(loc, value, result, position);
								 });

		return mlir::success();
	}
};

/**
 * Cross product of two 1-D arrays. Result is a scalar.
 *
 * [ x1, x2, x3 ] * [ y1, y2, y3 ] = x1 * y1 + x2 * y2 + x3 * y3
 */
struct MulOpCrossProductLowering: public ModelicaOpConversion<MulOp>
{
	using ModelicaOpConversion<MulOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(MulOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		// Check if the operands are compatible
		if (!op.lhs().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Cross product: left-hand side value is not an array");

		auto lhsPointerType = op.lhs().getType().cast<PointerType>();

		if (lhsPointerType.getRank() != 1)
			return rewriter.notifyMatchFailure(op, "Cross product: left-hand side arrays is not 1D");

		if (!op.rhs().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Cross product: right-hand side value is not an array");

		auto rhsPointerType = op.rhs().getType().cast<PointerType>();

		if (rhsPointerType.getRank() != 1)
			return rewriter.notifyMatchFailure(op, "Cross product: right-hand side arrays is not 1D");

		if (lhsPointerType.getShape()[0] != -1 && rhsPointerType.getShape()[0] != -1)
			if (lhsPointerType.getShape()[0] != rhsPointerType.getShape()[0])
				return rewriter.notifyMatchFailure(op, "Cross product: the two arrays have different shape");

		// Compute the result
		mlir::Type type = op.resultType();
		Adaptor transformed(operands);

		assert(lhsPointerType.getRank() == 1);
		assert(rhsPointerType.getRank() == 1);

		mlir::Value lowerBound = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
		mlir::Value upperBound = rewriter.create<DimOp>(loc, op.lhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0)));
		mlir::Value step = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));
		mlir::Value init = rewriter.create<mlir::ConstantOp>(loc, rewriter.getZeroAttr(convertType(type)));

		// Iterate on the two arrays at the same time, and propagate the
		// progressive result to the next loop iteration.
		auto loop = rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step, init);

		{
			mlir::OpBuilder::InsertionGuard guard(rewriter);
			rewriter.setInsertionPointToStart(loop.getBody());

			mlir::Value lhs = rewriter.create<LoadOp>(loc, op.lhs(), loop.getInductionVar());
			mlir::Value rhs = rewriter.create<LoadOp>(loc, op.rhs(), loop.getInductionVar());
			mlir::Value product = rewriter.create<MulOp>(loc, type, lhs, rhs);
			mlir::Value sum = getTypeConverter()->materializeSourceConversion(rewriter, loc, type, loop.getRegionIterArgs()[0]);
			sum = rewriter.create<AddOp>(loc, type, product, sum);
			sum = materializeTargetConversion(rewriter, sum);
			rewriter.create<mlir::scf::YieldOp>(loc, sum);
		}

		mlir::Value result = getTypeConverter()->materializeSourceConversion(rewriter, loc, type, loop.getResult(0));
		rewriter.replaceOp(op, result);
		return mlir::success();
	}
};

/**
 * Product of a vector (1-D array) and a matrix (2-D array).
 *
 * [ x1, x2, x3 ] * [ y11, y12 ] = [ (x1 * y11 + x2 * y21 + x3 * y31), (x1 * y12 + x2 * y22 + x3 * y32) ]
 * 									[ y21, y22 ]
 * 									[ y31, y32 ]
 */
struct MulOpVectorMatrixLowering: public ModelicaOpConversion<MulOp>
{
	using ModelicaOpConversion<MulOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(MulOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		// Check if the operands are compatible
		if (!op.lhs().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Vector-matrix product: left-hand side value is not an array");

		auto lhsPointerType = op.lhs().getType().cast<PointerType>();

		if (lhsPointerType.getRank() != 1)
			return rewriter.notifyMatchFailure(op, "Vector-matrix product: left-hand size array is not 1-D");

		if (!op.rhs().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Vector-matrix product: right-hand side value is not an array");

		auto rhsPointerType = op.rhs().getType().cast<PointerType>();

		if (rhsPointerType.getRank() != 2)
			return rewriter.notifyMatchFailure(op, "Vector-matrix product: right-hand side matrix is not 2-D");

		if (lhsPointerType.getShape()[0] != -1 && rhsPointerType.getShape()[0] != -1)
			if (lhsPointerType.getShape()[0] != rhsPointerType.getShape()[0])
				return rewriter.notifyMatchFailure(op, "Vector-matrix product: incompatible shapes");

		// Allocate the result array
		Adaptor transformed(operands);

		assert(lhsPointerType.getRank() == 1);
		assert(rhsPointerType.getRank() == 2);

		auto resultType = op.resultType().cast<PointerType>();
		auto shape = resultType.getShape();
		assert(shape.size() == 1);

		llvm::SmallVector<mlir::Value, 1> dynamicDimensions;

		if (shape[0] == -1)
			dynamicDimensions.push_back(rewriter.create<DimOp>(
					loc, op.rhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1))));

		mlir::Value result = allocate(rewriter, loc, resultType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		// Iterate on the columns
		mlir::Value columnsLowerBound = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(0));
		mlir::Value columnsUpperBound = rewriter.create<DimOp>(loc, op.rhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1)));
		mlir::Value columnsStep = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));

		auto outerLoop = rewriter.create<mlir::scf::ForOp>(loc, columnsLowerBound, columnsUpperBound, columnsStep);
		rewriter.setInsertionPointToStart(outerLoop.getBody());

		// Product between the vector and the current column
		mlir::Value lowerBound = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(0));
		mlir::Value upperBound = rewriter.create<DimOp>(loc, op.lhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0)));
		mlir::Value step = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));
		mlir::Value init = rewriter.create<mlir::ConstantOp>(loc, rewriter.getZeroAttr(convertType(resultType.getElementType())));

		auto innerLoop = rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step, init);
		rewriter.setInsertionPointToStart(innerLoop.getBody());

		mlir::Value lhs = rewriter.create<LoadOp>(loc, op.lhs(), innerLoop.getInductionVar());
		mlir::Value rhs = rewriter.create<LoadOp>(loc, op.rhs(), mlir::ValueRange({ innerLoop.getInductionVar(), outerLoop.getInductionVar() }));
		mlir::Value product = rewriter.create<MulOp>(loc, resultType.getElementType(), lhs, rhs);
		mlir::Value sum = getTypeConverter()->materializeSourceConversion(rewriter, loc, resultType.getElementType(), innerLoop.getRegionIterArgs()[0]);
		sum = rewriter.create<AddOp>(loc, resultType.getElementType(), product, sum);
		sum = materializeTargetConversion(rewriter, sum);
		rewriter.create<mlir::scf::YieldOp>(loc, sum);

		// Store the product in the result array
		rewriter.setInsertionPointAfter(innerLoop);
		mlir::Value productResult = innerLoop.getResult(0);
		productResult = getTypeConverter()->materializeSourceConversion(rewriter, loc, resultType.getElementType(), productResult);
		rewriter.create<StoreOp>(loc, productResult, result, outerLoop.getInductionVar());

		rewriter.setInsertionPointAfter(outerLoop);
		return mlir::success();
	}
};

/**
 * Product of a matrix (2-D array) and a vector (1-D array).
 *
 * [ x11, x12 ] * [ y1, y2 ] = [ x11 * y1 + x12 * y2 ]
 * [ x21, x22 ]								 [ x21 * y1 + x22 * y2 ]
 * [ x31, x32 ]								 [ x31 * y1 + x22 * y2 ]
 */
struct MulOpMatrixVectorLowering: public ModelicaOpConversion<MulOp>
{
	using ModelicaOpConversion<MulOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(MulOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		// Check if the operands are compatible
		if (!op.lhs().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Matrix-vector product: left-hand side value is not an array");

		auto lhsPointerType = op.lhs().getType().cast<PointerType>();

		if (lhsPointerType.getRank() != 2)
			return rewriter.notifyMatchFailure(op, "Matrix-vector product: left-hand size array is not 2-D");

		if (!op.rhs().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Matrix-vector product: right-hand side value is not an array");

		auto rhsPointerType = op.rhs().getType().cast<PointerType>();

		if (rhsPointerType.getRank() != 1)
			return rewriter.notifyMatchFailure(op, "Matrix-vector product: right-hand side matrix is not 1-D");

		if (lhsPointerType.getShape()[1] != -1 && rhsPointerType.getShape()[0] != -1)
			if (lhsPointerType.getShape()[1] != rhsPointerType.getShape()[0])
				return rewriter.notifyMatchFailure(op, "Matrix-vector product: incompatible shapes");

		// Allocate the result array
		Adaptor transformed(operands);

		assert(lhsPointerType.getRank() == 2);
		assert(rhsPointerType.getRank() == 1);

		auto resultType = op.resultType().cast<PointerType>();
		auto shape = resultType.getShape();

		llvm::SmallVector<mlir::Value, 1> dynamicDimensions;

		if (shape[0] == -1)
			dynamicDimensions.push_back(rewriter.create<DimOp>(
					loc, op.lhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0))));

		mlir::Value result = allocate(rewriter, loc, resultType, dynamicDimensions);

		// Iterate on the rows
		mlir::Value rowsLowerBound = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(0));
		mlir::Value rowsUpperBound = rewriter.create<DimOp>(loc, op.lhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0)));
		mlir::Value rowsStep = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));

		auto outerLoop = rewriter.create<mlir::scf::ParallelOp>(loc, rowsLowerBound, rowsUpperBound, rowsStep);
		rewriter.setInsertionPointToStart(outerLoop.getBody());

		// Product between the current row and the vector
		mlir::Value lowerBound = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(0));
		mlir::Value upperBound = rewriter.create<DimOp>(loc, op.rhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0)));
		mlir::Value step = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));
		mlir::Value init = rewriter.create<mlir::ConstantOp>(loc, rewriter.getZeroAttr(convertType(resultType.getElementType())));

		auto innerLoop = rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step, init);
		rewriter.setInsertionPointToStart(innerLoop.getBody());

		mlir::Value lhs = rewriter.create<LoadOp>(loc, op.lhs(), mlir::ValueRange({ outerLoop.getInductionVars()[0], innerLoop.getInductionVar() }));
		mlir::Value rhs = rewriter.create<LoadOp>(loc, op.rhs(), innerLoop.getInductionVar());
		mlir::Value product = rewriter.create<MulOp>(loc, resultType.getElementType(), lhs, rhs);
		mlir::Value sum = getTypeConverter()->materializeSourceConversion(rewriter, loc, resultType.getElementType(), innerLoop.getRegionIterArgs()[0]);
		sum = rewriter.create<AddOp>(loc, resultType.getElementType(), product, sum);
		sum = materializeTargetConversion(rewriter, sum);
		rewriter.create<mlir::scf::YieldOp>(loc, sum);

		// Store the product in the result array
		rewriter.setInsertionPointAfter(innerLoop);
		mlir::Value productResult = innerLoop.getResult(0);
		productResult = getTypeConverter()->materializeSourceConversion(rewriter, loc, resultType.getElementType(), productResult);
		rewriter.create<StoreOp>(loc, productResult, result, outerLoop.getInductionVars()[0]);

		rewriter.setInsertionPointAfter(outerLoop);

		rewriter.replaceOp(op, result);
		return mlir::success();
	}
};

/**
 * Product of two matrixes (2-D arrays).
 *
 * [ x11, x12, x13 ] * [ y11, y12 ] = [ x11 * y11 + x12 * y21 + x13 * y31, x11 * y12 + x12 * y22 + x13 * y32 ]
 * [ x21, x22, x23 ]   [ y21, y22 ]		[ x21 * y11 + x22 * y21 + x23 * y31, x21 * y12 + x22 * y22 + x23 * y32 ]
 * [ x31, x32, x33 ]	 [ y31, y32 ]		[ x31 * y11 + x32 * y21 + x33 * y31, x31 * y12 + x32 * y22 + x33 * y32 ]
 * [ x41, x42, x43 ]
 */
struct MulOpMatrixLowering: public ModelicaOpConversion<MulOp>
{
	using ModelicaOpConversion<MulOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(MulOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		// Check if the operands are compatible
		if (!op.lhs().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Matrix product: left-hand side value is not an array");

		auto lhsPointerType = op.lhs().getType().cast<PointerType>();

		if (lhsPointerType.getRank() != 2)
			return rewriter.notifyMatchFailure(op, "Matrix product: left-hand size array is not 2-D");

		if (!op.rhs().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Matrix product: right-hand side value is not an array");

		auto rhsPointerType = op.rhs().getType().cast<PointerType>();

		if (rhsPointerType.getRank() != 2)
			return rewriter.notifyMatchFailure(op, "Matrix product: right-hand side matrix is not 2-D");

		if (lhsPointerType.getShape()[1] != -1 && rhsPointerType.getShape()[0] != -1)
			if (lhsPointerType.getShape()[1] != rhsPointerType.getShape()[0])
				return rewriter.notifyMatchFailure(op, "Matrix-vector product: incompatible shapes");

		// Allocate the result array
		Adaptor transformed(operands);

		assert(lhsPointerType.getRank() == 2);
		assert(rhsPointerType.getRank() == 2);

		auto resultType = op.resultType().cast<PointerType>();
		auto shape = resultType.getShape();

		llvm::SmallVector<mlir::Value, 2> dynamicDimensions;

		if (shape[0] == -1)
			dynamicDimensions.push_back(rewriter.create<DimOp>(
					loc, op.lhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0))));

		if (shape[1] == -1)
			dynamicDimensions.push_back(rewriter.create<DimOp>(
					loc, op.rhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1))));

		mlir::Value result = allocate(rewriter, loc, resultType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		// Iterate on the rows
		mlir::Value rowsLowerBound = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(0));
		mlir::Value rowsUpperBound = rewriter.create<DimOp>(loc, op.lhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0)));
		mlir::Value rowsStep = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));

		auto rowsLoop = rewriter.create<mlir::scf::ForOp>(loc, rowsLowerBound, rowsUpperBound, rowsStep);
		rewriter.setInsertionPointToStart(rowsLoop.getBody());

		// Iterate on the columns
		mlir::Value columnsLowerBound = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(0));
		mlir::Value columnsUpperBound = rewriter.create<DimOp>(loc, op.rhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1)));
		mlir::Value columnsStep = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));

		auto columnsLoop = rewriter.create<mlir::scf::ForOp>(loc, columnsLowerBound, columnsUpperBound, columnsStep);
		rewriter.setInsertionPointToStart(columnsLoop.getBody());

		// Product between the current row and the current column
		mlir::Value lowerBound = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(0));
		mlir::Value upperBound = rewriter.create<DimOp>(loc, op.rhs(), rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0)));
		mlir::Value step = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));
		mlir::Value init = rewriter.create<mlir::ConstantOp>(loc, rewriter.getZeroAttr(convertType(resultType.getElementType())));

		auto innerLoop = rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step, init);
		rewriter.setInsertionPointToStart(innerLoop.getBody());

		mlir::Value lhs = rewriter.create<LoadOp>(loc, op.lhs(), mlir::ValueRange({ rowsLoop.getInductionVar(), innerLoop.getInductionVar() }));
		mlir::Value rhs = rewriter.create<LoadOp>(loc, op.rhs(), mlir::ValueRange({ innerLoop.getInductionVar(), columnsLoop.getInductionVar() }));
		mlir::Value product = rewriter.create<MulOp>(loc, resultType.getElementType(), lhs, rhs);
		mlir::Value sum = getTypeConverter()->materializeSourceConversion(rewriter, loc, resultType.getElementType(), innerLoop.getRegionIterArgs()[0]);
		sum = rewriter.create<AddOp>(loc, resultType.getElementType(), product, sum);
		sum = materializeTargetConversion(rewriter, sum);
		rewriter.create<mlir::scf::YieldOp>(loc, sum);

		// Store the product in the result array
		rewriter.setInsertionPointAfter(innerLoop);
		mlir::Value productResult = innerLoop.getResult(0);
		productResult = getTypeConverter()->materializeSourceConversion(rewriter, loc, resultType.getElementType(), productResult);
		rewriter.create<StoreOp>(loc, productResult, result, mlir::ValueRange({ rowsLoop.getInductionVar(), columnsLoop.getInductionVar() }));

		rewriter.setInsertionPointAfter(rowsLoop);

		return mlir::success();
	}
};

/**
 * Division between two scalar values.
 */
struct DivOpLowering: public ModelicaOpConversion<DivOp>
{
	using ModelicaOpConversion<DivOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(DivOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		// Check if the operands are compatible
		if (!isNumeric(op.lhs()))
			return rewriter.notifyMatchFailure(op, "Scalar-scalar division: left-hand side value is not a scalar");

		if (!isNumeric(op.rhs()))
			return rewriter.notifyMatchFailure(op, "Scalar-scalar division: right-hand side value is not a scalar");

		// Cast the operands to the most generic type
		auto castOp = rewriter.create<CastCommonOp>(loc, op->getOperands());
		llvm::SmallVector<mlir::Value, 3> castedOperands;

		for (mlir::Value operand : castOp.getResults())
			castedOperands.push_back(materializeTargetConversion(rewriter, operand));

		Adaptor transformed(castedOperands);
		mlir::Type type = castOp.resultType();

		// Compute the result
		if (type.isa<mlir::IndexType, BooleanType, IntegerType>())
		{
			mlir::Value result = rewriter.create<mlir::SignedDivIOp>(loc, transformed.lhs(), transformed.rhs());
			result = getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		if (type.isa<RealType>())
		{
			mlir::Value result = rewriter.create<mlir::DivFOp>(loc, transformed.lhs(), transformed.rhs());
			result = getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		return rewriter.notifyMatchFailure(op, "Unknown type");
	}
};

/**
 * Division between an array and a scalar value.
 */
struct DivOpArrayLowering: public ModelicaOpConversion<DivOp>
{
	using ModelicaOpConversion<DivOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(DivOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		// Check if the operands are compatible
		if (!op.lhs().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Array-scalar division: left-hand size value is not an array");

		if (!isNumericType(op.lhs().getType().cast<PointerType>().getElementType()))
			return rewriter.notifyMatchFailure(op, "Array-scalar division: right-hand side array has not numeric elements");

		if (!isNumeric(op.rhs()))
			return rewriter.notifyMatchFailure(op, "Array-scalar division: right-hand size value is not a scalar");

		// Allocate the result array
		auto resultType = op.resultType().cast<PointerType>();
		auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.lhs());
		mlir::Value result = allocate(rewriter, loc, resultType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		// Divide each array element by the scalar value
		iterateArray(rewriter, loc, op.lhs(),
								 [&](mlir::ValueRange position) {
									 mlir::Value arrayValue = rewriter.create<LoadOp>(loc, op.lhs(), position);
									 mlir::Value value = rewriter.create<DivOp>(loc, resultType.getElementType(), arrayValue, op.rhs());
									 rewriter.create<StoreOp>(loc, value, result, position);
								 });

		return mlir::success();
	}
};

/**
 * Exponentiation of a scalar value.
 */
struct PowOpLowering: public ModelicaOpConversion<PowOp>
{
	using ModelicaOpConversion<PowOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(PowOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		// Check if the operands are compatible
		if (!isNumeric(op.base()))
			return rewriter.notifyMatchFailure(op, "Base is not a scalar");

		if (!isNumeric(op.exponent()))
			return rewriter.notifyMatchFailure(op, "Base is not a scalar");

		// Compute the result
		Adaptor adaptor(operands);
		mlir::Value result = rewriter.create<mlir::math::PowFOp>(loc, adaptor.base(), adaptor.exponent());
		result = getTypeConverter()->materializeSourceConversion(rewriter, loc, op.base().getType(), result);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());

		return mlir::success();
	}
};

/**
 * Exponentiation of a square matrix.
 */
struct PowOpMatrixLowering: public ModelicaOpConversion<PowOp>
{
	using ModelicaOpConversion<PowOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(PowOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();
		Adaptor transformed(operands);

		// Check if the operands are compatible
		if (!op.base().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Base is not an array");

		auto basePointerType = op.base().getType().cast<PointerType>();

		if (basePointerType.getRank() != 2)
			return rewriter.notifyMatchFailure(op, "Base array is not 2-D");

		if (basePointerType.getShape()[0] != -1 && basePointerType.getShape()[1] != -1)
			if (basePointerType.getShape()[0] != basePointerType.getShape()[1])
				return rewriter.notifyMatchFailure(op, "Base is not a square matrix");

		if (!op.exponent().getType().isa<IntegerType>())
			return rewriter.notifyMatchFailure(op, "Exponent is not an integer");

		// Allocate the result array
		auto resultPointerType = op.resultType().cast<PointerType>();
		auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.base());
		mlir::Value result = allocate(rewriter, loc, resultPointerType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		// Compute the result
		mlir::Value exponent = rewriter.create<CastOp>(loc, op.exponent(), mlir::IndexType::get(op->getContext()));
		mlir::Value lowerBound = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));
		mlir::Value step = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));

		// The intermediate results must be allocated on the heap, in order
		// to avoid a potentially big allocation on the stack (due to the
		// iteration).
		auto intermediateResultType = op.base().getType().cast<PointerType>().toAllocationScope(BufferAllocationScope::heap);
		mlir::Value current = rewriter.create<ArrayCloneOp>(loc, op.base(), intermediateResultType);

		auto forLoop = rewriter.create<mlir::scf::ForOp>(loc, lowerBound, exponent, step);

		{
			mlir::OpBuilder::InsertionGuard guard(rewriter);
			rewriter.setInsertionPointToStart(forLoop.getBody());
			mlir::Value next = rewriter.create<MulOp>(loc, intermediateResultType, current, op.base());
			rewriter.create<AssignmentOp>(loc, next, current);
		}

		rewriter.create<AssignmentOp>(loc, current, result);
		return mlir::success();
	}
};

struct NDimsOpLowering: public ModelicaOpConversion<NDimsOp>
{
	using ModelicaOpConversion<NDimsOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(NDimsOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();
		auto pointerType = op.memory().getType().cast<PointerType>();
		mlir::Value result = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(pointerType.getRank()));
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
		return mlir::success();
	}
};

/**
 * Get the size of a specific array dimension.
 */
struct SizeOpDimensionLowering: public ModelicaOpConversion<SizeOp>
{
	using ModelicaOpConversion<SizeOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(SizeOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		if (!op.hasIndex())
			return rewriter.notifyMatchFailure(op, "No index specified");

		mlir::Value index = rewriter.create<CastOp>(loc, op.index(), rewriter.getIndexType());
		mlir::Value result = rewriter.create<DimOp>(loc, op.memory(), index);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
		return mlir::success();
	}
};

/**
 * Get the size of alla the array dimensions.
 */
struct SizeOpArrayLowering: public ModelicaOpConversion<SizeOp>
{
	using ModelicaOpConversion<SizeOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(SizeOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();
		Adaptor transformed(operands);

		if (op.hasIndex())
			return rewriter.notifyMatchFailure(op, "Index specified");

		assert(op.resultType().isa<PointerType>());
		auto resultType = op.resultType().cast<PointerType>();
		mlir::Value result = allocate(rewriter, loc, resultType, llvm::None);

		// Iterate on each dimension
		mlir::Value zeroValue = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(0));

		auto pointerType = op.memory().getType().cast<PointerType>();
		mlir::Value rank = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(pointerType.getRank()));
		mlir::Value step = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));

		auto loop = rewriter.create<mlir::scf::ForOp>(loc, zeroValue, rank, step);

		{
			mlir::OpBuilder::InsertionGuard guard(rewriter);
			rewriter.setInsertionPointToStart(loop.getBody());

			// Get the size of the current dimension
			mlir::Value dimensionSize = rewriter.create<SizeOp>(loc, resultType.getElementType(), op.memory(), loop.getInductionVar());

			// Cast it to the result base type and store it into the result array
			dimensionSize = rewriter.create<CastOp>(loc, dimensionSize, resultType.getElementType());
			rewriter.create<StoreOp>(loc, dimensionSize, result, loop.getInductionVar());
		}
		rewriter.replaceOp(op, result);
		return mlir::success();
	}
};

struct IdentityOpLowering: public ModelicaOpConversion<IdentityOp>
{
	using ModelicaOpConversion<IdentityOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(IdentityOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto pointerType = op.resultType().cast<PointerType>();

		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
		mlir::Value castedSize = nullptr;

		for (const auto& size : pointerType.getShape())
		{
			if (size == -1)
			{
				if (castedSize == nullptr)
					castedSize = rewriter.create<CastOp>(loc, op.size(), rewriter.getIndexType());

				dynamicDimensions.push_back(castedSize);
			}
		}

		mlir::Value result = allocate(rewriter, loc, pointerType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		mlir::Value arg = rewriter.create<PtrCastOp>(
				loc, result,
				result.getType().cast<PointerType>().toUnsized());

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("identity", arg),
				llvm::None,
				arg.getType());

		rewriter.create<CallOp>(loc, callee.getName(), llvm::None, arg);
		return mlir::success();
	}
};

struct DiagonalOpLowering: public ModelicaOpConversion<DiagonalOp>
{
	using ModelicaOpConversion<DiagonalOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(DiagonalOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto pointerType = op.resultType().cast<PointerType>();

		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
		mlir::Value castedSize = nullptr;

		for (const auto& size : pointerType.getShape())
		{
			if (size == -1)
			{
				if (castedSize == nullptr)
				{
					assert(op.values().getType().cast<PointerType>().getRank() == 1);
					mlir::Value zeroValue = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
					castedSize = rewriter.create<DimOp>(loc, op.values(), zeroValue);
				}

				dynamicDimensions.push_back(castedSize);
			}
		}

		mlir::Value result = allocate(rewriter, loc, pointerType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		llvm::SmallVector<mlir::Value, 3> args;

		args.push_back(rewriter.create<PtrCastOp>(
				loc, result,
				result.getType().cast<PointerType>().toUnsized()));

		args.push_back(rewriter.create<PtrCastOp>(
				loc, op.values(),
				op.values().getType().cast<PointerType>().toUnsized()));

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("diagonal", args),
				llvm::None,
				args);

		rewriter.create<CallOp>(loc, callee.getName(), llvm::None, args);
		return mlir::success();
	}
};

struct FillOpLowering: public ModelicaOpConversion<FillOp>
{
	using ModelicaOpConversion<FillOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(FillOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto pointerType = op.memory().getType().cast<PointerType>();

		if (options.useRuntimeLibrary)
		{
			llvm::SmallVector<mlir::Value, 3> args;
			args.push_back(rewriter.create<PtrCastOp>(loc, op.memory(), pointerType.toUnsized()));
			args.push_back(rewriter.create<CastOp>(loc, op.value(), pointerType.getElementType()));

			auto callee = getOrDeclareFunction(
					rewriter,
					op->getParentOfType<mlir::ModuleOp>(),
					getMangledFunctionName("fill", args),
					llvm::None,
					mlir::ValueRange(args).getTypes());

			rewriter.create<CallOp>(loc, callee.getName(), llvm::None, args);
		}
		else
		{
			mlir::Value value = rewriter.create<CastOp>(loc, op.value(), pointerType.getElementType());

			iterateArray(rewriter, loc, op.memory(),
									 [&](mlir::ValueRange position) {
										 rewriter.create<StoreOp>(loc, value, op.memory(), position);
									 });
		}

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct IfOpLowering: public ModelicaOpConversion<IfOp>
{
	using ModelicaOpConversion<IfOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(IfOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		Adaptor adaptor(operands);
		bool hasElseBlock = !op.elseRegion().empty();

		mlir::scf::IfOp ifOp;

		// In order to move the blocks into the SCF operation, we need to override
		// its blocks builders. In fact, the default ones already place the
		// SCF::YieldOp terminators, but our IR already has the Modelica::YieldOps
		// converted to SCF::YieldOps (note that although in this context the
		// Modelica::YieldOps don't carry any useful data, we can't avoid creating
		// them, or the blocks would have no terminator, which is illegal).

		assert(op.thenRegion().getBlocks().size() == 1);

		auto thenBuilder = [&](mlir::OpBuilder& builder, mlir::Location location)
		{
			rewriter.mergeBlocks(&op.thenRegion().front(), rewriter.getInsertionBlock(), llvm::None);
		};

		if (hasElseBlock)
		{
			assert(op.elseRegion().getBlocks().size() == 1);

			ifOp = rewriter.replaceOpWithNewOp<mlir::scf::IfOp>(
					op, op.resultTypes(), adaptor.condition(), thenBuilder,
					[&](mlir::OpBuilder& builder, mlir::Location location)
					{
						rewriter.mergeBlocks(&op.elseRegion().front(), builder.getInsertionBlock(), llvm::None);
					});
		}
		else
		{
			ifOp = rewriter.replaceOpWithNewOp<mlir::scf::IfOp>(op, op.resultTypes(), adaptor.condition(), thenBuilder, nullptr);
		}

		// Replace the Modelica::YieldOp terminator in the "then" branch with
		// a SCF::YieldOp.

		mlir::Block* thenBlock = &ifOp.thenRegion().front();
		auto thenTerminator = mlir::cast<YieldOp>(thenBlock->getTerminator());
		rewriter.setInsertionPointToEnd(thenBlock);
		rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(thenTerminator, thenTerminator.getOperands());

		// If the operation also has an "else" block, also replace its
		// Modelica::YieldOp terminator with a SCF::YieldOp.

		if (hasElseBlock)
		{
			mlir::Block* elseBlock = &ifOp.elseRegion().front();
			auto elseTerminator = mlir::cast<YieldOp>(elseBlock->getTerminator());
			rewriter.setInsertionPointToEnd(elseBlock);
			rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(elseTerminator, elseTerminator.getOperands());
		}

		return mlir::success();
	}
};

struct BreakableForOpLowering: public ModelicaOpConversion<BreakableForOp>
{
	using ModelicaOpConversion<BreakableForOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(BreakableForOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		Adaptor transformed(operands);

		// Split the current block
		mlir::Block* currentBlock = rewriter.getInsertionBlock();
		mlir::Block* continuation = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

		// Inline regions
		mlir::Block* conditionBlock = &op.condition().front();
		mlir::Block* bodyBlock = &op.body().front();
		mlir::Block* stepBlock = &op.step().front();

		rewriter.inlineRegionBefore(op.step(), continuation);
		rewriter.inlineRegionBefore(op.body(), stepBlock);
		rewriter.inlineRegionBefore(op.condition(), bodyBlock);

		// Start the for loop by branching to the "condition" region
		rewriter.setInsertionPointToEnd(currentBlock);
		rewriter.create<mlir::BranchOp>(loc, conditionBlock, op.args());

		// The loop is supposed to be breakable. Thus, before checking the normal
		// condition, we first need to check if the break condition variable has
		// been set to true in the previous loop execution. If it is set to true,
		// it means that a break statement has been executed and thus the loop
		// must be terminated.

		rewriter.setInsertionPointToStart(conditionBlock);

		mlir::Value breakCondition = rewriter.create<LoadOp>(loc, op.breakCondition());
		breakCondition = materializeTargetConversion(rewriter, breakCondition);

		mlir::Value returnCondition = rewriter.create<LoadOp>(loc, op.returnCondition());
		returnCondition = materializeTargetConversion(rewriter, returnCondition);

		mlir::Value stopCondition = rewriter.create<mlir::OrOp>(loc, breakCondition, returnCondition);

		mlir::Value trueValue = rewriter.create<mlir::ConstantOp>(loc, rewriter.getBoolAttr(true));
		mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, stopCondition, trueValue);

		auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, rewriter.getI1Type(), condition, true);
		mlir::Block* originalCondition = rewriter.splitBlock(conditionBlock, rewriter.getInsertionPoint());

		// If the break condition variable is set to true, return false from the
		// condition block in order to stop the loop execution.
		rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
		mlir::Value falseValue = rewriter.create<mlir::ConstantOp>(loc, rewriter.getBoolAttr(false));
		rewriter.create<mlir::scf::YieldOp>(loc, falseValue);

		// Move the original condition check in the "else" branch
		rewriter.mergeBlocks(originalCondition, &ifOp.elseRegion().front(), llvm::None);
		rewriter.setInsertionPointToEnd(&ifOp.elseRegion().front());
		auto conditionOp = mlir::cast<ConditionOp>(ifOp.elseRegion().front().getTerminator());
		rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(conditionOp, materializeTargetConversion(rewriter, conditionOp.condition()));

		// The original condition operation is converted to the SCF one and takes
		// as condition argument the result of the If operation, which is false
		// if a break must be executed or the intended condition value otherwise.
		rewriter.setInsertionPointAfter(ifOp);
		rewriter.create<mlir::CondBranchOp>(loc, ifOp.getResult(0), bodyBlock, conditionOp.args(), continuation, llvm::None);

		// Replace "body" block terminator with a branch to the "step" block
		rewriter.setInsertionPointToEnd(bodyBlock);
		auto bodyYieldOp = mlir::cast<YieldOp>(bodyBlock->getTerminator());
		rewriter.replaceOpWithNewOp<mlir::BranchOp>(bodyYieldOp, stepBlock, bodyYieldOp.args());

		// Branch to the condition check after incrementing the induction variable
		rewriter.setInsertionPointToEnd(stepBlock);
		auto stepYieldOp = mlir::cast<YieldOp>(stepBlock->getTerminator());
		rewriter.replaceOpWithNewOp<mlir::BranchOp>(stepYieldOp, conditionBlock, stepYieldOp.args());

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct ForOpLowering: public ModelicaOpConversion<ForOp>
{
	using ModelicaOpConversion<ForOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(ForOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		Adaptor transformed(operands);

		// Split the current block
		mlir::Block* currentBlock = rewriter.getInsertionBlock();
		mlir::Block* continuation = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

		// Inline regions
		mlir::Block* conditionBlock = &op.condition().front();
		mlir::Block* bodyBlock = &op.body().front();
		mlir::Block* stepBlock = &op.step().front();

		rewriter.inlineRegionBefore(op.step(), continuation);
		rewriter.inlineRegionBefore(op.body(), stepBlock);
		rewriter.inlineRegionBefore(op.condition(), bodyBlock);

		// Start the for loop by branching to the "condition" region
		rewriter.setInsertionPointToEnd(currentBlock);
		rewriter.create<mlir::BranchOp>(loc, conditionBlock, op.args());

		// Check the condition
		auto conditionOp = mlir::cast<ConditionOp>(conditionBlock->getTerminator());
		rewriter.setInsertionPoint(conditionOp);
		rewriter.replaceOpWithNewOp<mlir::CondBranchOp>(conditionOp, materializeTargetConversion(rewriter, conditionOp.condition()), bodyBlock, conditionOp.args(), continuation, llvm::None);

		// Replace "body" block terminator with a branch to the "step" block
		rewriter.setInsertionPointToEnd(bodyBlock);
		auto bodyYieldOp = mlir::cast<YieldOp>(bodyBlock->getTerminator());
		rewriter.replaceOpWithNewOp<mlir::BranchOp>(bodyYieldOp, stepBlock, bodyYieldOp.args());

		// Branch to the condition check after incrementing the induction variable
		rewriter.setInsertionPointToEnd(stepBlock);
		auto stepYieldOp = mlir::cast<YieldOp>(stepBlock->getTerminator());
		rewriter.replaceOpWithNewOp<mlir::BranchOp>(stepYieldOp, conditionBlock, stepYieldOp.args());

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct BreakableWhileOpLowering: public ModelicaOpConversion<BreakableWhileOp>
{
	using ModelicaOpConversion<BreakableWhileOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(BreakableWhileOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();
		Adaptor transformed(operands);

		auto whileOp = rewriter.create<mlir::scf::WhileOp>(loc, llvm::None, llvm::None);

		// The body block requires no modification apart from the change of the
		// terminator to the SCF dialect one.

		rewriter.createBlock(&whileOp.after());
		rewriter.mergeBlocks(&op.body().front(), &whileOp.after().front(), llvm::None);
		mlir::Block* body = &whileOp.after().front();
		auto bodyTerminator = mlir::cast<YieldOp>(body->getTerminator());
		rewriter.setInsertionPointToEnd(body);
		rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(bodyTerminator, bodyTerminator.getOperands());

		// The loop is supposed to be breakable. Thus, before checking the normal
		// condition, we first need to check if the break condition variable has
		// been set to true in the previous loop execution. If it is set to true,
		// it means that a break statement has been executed and thus the loop
		// must be terminated.

		rewriter.createBlock(&whileOp.before());
		rewriter.setInsertionPointToStart(&whileOp.before().front());

		mlir::Value breakCondition = rewriter.create<LoadOp>(loc, op.breakCondition());
		breakCondition = materializeTargetConversion(rewriter, breakCondition);

		mlir::Value returnCondition = rewriter.create<LoadOp>(loc, op.returnCondition());
		returnCondition = materializeTargetConversion(rewriter, returnCondition);

		mlir::Value stopCondition = rewriter.create<mlir::OrOp>(loc, breakCondition, returnCondition);

		mlir::Value trueValue = rewriter.create<mlir::ConstantOp>(loc, rewriter.getBoolAttr(true));
		mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, stopCondition, trueValue);

		auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, rewriter.getI1Type(), condition, true);

		// If the break condition variable is set to true, return false from the
		// condition block in order to stop the loop execution.
		rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
		mlir::Value falseValue = rewriter.create<mlir::ConstantOp>(loc, rewriter.getBoolAttr(false));
		rewriter.create<mlir::scf::YieldOp>(loc, falseValue);

		// Move the original condition check in the "else" branch
		rewriter.mergeBlocks(&op.condition().front(), &ifOp.elseRegion().front(), llvm::None);
		rewriter.setInsertionPointToEnd(&ifOp.elseRegion().front());
		auto conditionOp = mlir::cast<ConditionOp>(ifOp.elseRegion().front().getTerminator());
		rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(conditionOp, materializeTargetConversion(rewriter, conditionOp.condition()));

		// The original condition operation is converted to the SCF one and takes
		// as condition argument the result of the If operation, which is false
		// if a break must be executed or the intended condition value otherwise.
		rewriter.setInsertionPointAfter(ifOp);

		llvm::SmallVector<mlir::Value, 3> conditionOpArgs;

		for (mlir::Value arg : conditionOp.args())
			conditionOpArgs.push_back(materializeTargetConversion(rewriter, arg));

		rewriter.create<mlir::scf::ConditionOp>(loc, ifOp.getResult(0), conditionOpArgs);

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

static void populateModelicaControlFlowConversionPatterns(
		mlir::OwningRewritePatternList& patterns,
		mlir::MLIRContext* context,
		TypeConverter& typeConverter,
		ModelicaConversionOptions options)
{
	patterns.insert<
	    IfOpLowering,
			ForOpLowering,
			BreakableForOpLowering,
			BreakableWhileOpLowering>(context, typeConverter, options);
}

static void populateModelicaConversionPatterns(
		mlir::OwningRewritePatternList& patterns,
		mlir::MLIRContext* context,
		modelica::codegen::TypeConverter& typeConverter,
		ModelicaConversionOptions options)
{
	patterns.insert<
			ConstantOpLowering,
			PackOpLowering,
			ExtractOpLowering,
			AssignmentOpScalarLowering,
			AssignmentOpArrayLowering,
			CallOpLowering,
			PrintOpLowering,
			ArrayCloneOpLowering,
			NotOpScalarLowering,
			NotOpArrayLowering,
			AndOpScalarLowering,
			AndOpArrayLowering,
			OrOpScalarLowering,
			OrOpArrayLowering,
			EqOpLowering,
			NotEqOpLowering,
			GtOpLowering,
			GteOpLowering,
			LtOpLowering,
			LteOpLowering,
			NegateOpScalarLowering,
			NegateOpArrayLowering,
			AddOpScalarLowering,
			AddOpArrayLowering,
			SubOpScalarLowering,
			SubOpArrayLowering,
			MulOpLowering,
			MulOpScalarProductLowering,
			MulOpCrossProductLowering,
			MulOpVectorMatrixLowering,
			MulOpMatrixVectorLowering,
			MulOpMatrixLowering,
			DivOpLowering,
			DivOpArrayLowering,
			PowOpLowering,
			PowOpMatrixLowering,
			NDimsOpLowering,
			SizeOpDimensionLowering,
			SizeOpArrayLowering,
			IdentityOpLowering,
			DiagonalOpLowering,
			FillOpLowering>(context, typeConverter, options);
}

class ModelicaConversionPass: public mlir::PassWrapper<ModelicaConversionPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	explicit ModelicaConversionPass(ModelicaConversionOptions options)
			: options(std::move(options))
	{
	}

	void getDependentDialects(mlir::DialectRegistry &registry) const override
	{
		registry.insert<ModelicaDialect>();
		registry.insert<mlir::StandardOpsDialect>();
		registry.insert<mlir::math::MathDialect>();
		registry.insert<mlir::scf::SCFDialect>();
		registry.insert<mlir::LLVM::LLVMDialect>();
	}

	void runOnOperation() override
	{
		auto module = getOperation();

		mlir::ConversionTarget target(getContext());

		target.addIllegalOp<
		    ConstantOp, PackOp, ExtractOp,
				AssignmentOp,
				CallOp,
				FillOp,
				PrintOp,
				ArrayCloneOp,
				NotOp, AndOp, OrOp,
				EqOp, NotEqOp, GtOp, GteOp, LtOp, LteOp,
				NegateOp, AddOp, SubOp, MulOp, DivOp, PowOp,
				NDimsOp, SizeOp, IdentityOp, DiagonalOp, FillOp>();

		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });

		mlir::LowerToLLVMOptions llvmLoweringOptions;
		TypeConverter typeConverter(&getContext(), llvmLoweringOptions);

		// Provide the set of patterns that will lower the Modelica operations
		mlir::OwningRewritePatternList patterns;
		populateModelicaConversionPatterns(patterns, &getContext(), typeConverter, options);

		// With the target and rewrite patterns defined, we can now attempt the
		// conversion. The conversion will signal failure if any of our "illegal"
		// operations were not converted successfully.

		if (failed(applyPartialConversion(module, target, std::move(patterns))))
		{
			mlir::emitError(module.getLoc(), "Error in converting the Modelica operations\n");
			signalPassFailure();
		}
	}

	private:
	ModelicaConversionOptions options;
};

std::unique_ptr<mlir::Pass> modelica::codegen::createModelicaConversionPass(ModelicaConversionOptions options)
{
	return std::make_unique<ModelicaConversionPass>(options);
}

class LowerToCFGPass: public mlir::PassWrapper<LowerToCFGPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	explicit LowerToCFGPass(ModelicaConversionOptions options)
			: options(std::move(options))
	{
	}

	void runOnOperation() override
	{
		auto module = getOperation();

		mlir::ConversionTarget target(getContext());

		target.addIllegalOp<mlir::scf::ForOp, mlir::scf::IfOp, mlir::scf::ParallelOp, mlir::scf::WhileOp>();
		target.addIllegalOp<IfOp, ForOp, BreakableForOp, BreakableWhileOp>();
		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });

		mlir::LowerToLLVMOptions llvmLoweringOptions;
		TypeConverter typeConverter(&getContext(), llvmLoweringOptions);

		// Provide the set of patterns that will lower the Modelica operations
		mlir::OwningRewritePatternList patterns;
		populateModelicaControlFlowConversionPatterns(patterns, &getContext(), typeConverter, options);
		mlir::populateLoopToStdConversionPatterns(patterns, &getContext());

		// With the target and rewrite patterns defined, we can now attempt the
		// conversion. The conversion will signal failure if any of our "illegal"
		// operations were not converted successfully.

		if (failed(applyPartialConversion(module, target, std::move(patterns))))
		{
			mlir::emitError(module.getLoc(), "Error in converting the control flow operations\n");
			signalPassFailure();
		}
	}

	private:
	ModelicaConversionOptions options;
};

std::unique_ptr<mlir::Pass> modelica::codegen::createLowerToCFGPass(ModelicaConversionOptions options)
{
	return std::make_unique<LowerToCFGPass>(options);
}
