#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/ModelicaConversion.h>
#include <modelica/mlirlowerer/passes/TypeConverter.h>
#include <numeric>

using namespace modelica::codegen;

static bool isNumericType(mlir::Type type)
{
	return type.isa<mlir::IndexType, BooleanType, IntegerType, RealType>();
}

static bool isNumeric(mlir::Value value)
{
	return isNumericType(value.getType());
}

static void getArrayDynamicDimensions(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value array, llvm::SmallVectorImpl<mlir::Value>& dimensions)
{
	assert(array.getType().isa<PointerType>());
	auto pointerType = array.getType().cast<PointerType>();
	auto shape = pointerType.getShape();

	for (const auto& dimension : llvm::enumerate(pointerType.getShape()))
	{
		if (dimension.value() == -1)
		{
			mlir::Value dim = builder.create<ConstantOp>(loc, builder.getIndexAttr(dimension.index()));
			dimensions.push_back(builder.create<DimOp>(loc, array, dim));
		}
	}
}

static mlir::Value allocate(mlir::OpBuilder& builder, mlir::Location location, PointerType pointerType, mlir::ValueRange dynamicDimensions = llvm::None, bool shouldBeFreed = true)
{
	if (pointerType.getAllocationScope() == BufferAllocationScope::unknown)
		pointerType = pointerType.toMinAllowedAllocationScope();

	if (pointerType.getAllocationScope() == BufferAllocationScope::stack)
		return builder.create<AllocaOp>(location, pointerType.getElementType(), pointerType.getShape(), dynamicDimensions);

	return builder.create<AllocOp>(location, pointerType.getElementType(), pointerType.getShape(), dynamicDimensions, shouldBeFreed);
}

static mlir::FuncOp getOrDeclareFunction(mlir::OpBuilder& builder, mlir::ModuleOp module, llvm::StringRef name, mlir::TypeRange results, mlir::TypeRange args)
{
	if (auto foo = module.lookupSymbol<mlir::FuncOp>(name))
		return foo;

	mlir::PatternRewriter::InsertionGuard insertGuard(builder);
	builder.setInsertionPointToStart(module.getBody());
	auto foo = builder.create<mlir::FuncOp>(module.getLoc(), name, builder.getFunctionType(args, results));
	foo.setPrivate();
	return foo;
}

static mlir::FuncOp getOrDeclareFunction(mlir::OpBuilder& builder, mlir::ModuleOp module, llvm::StringRef name, mlir::TypeRange results, mlir::ValueRange args)
{
	return getOrDeclareFunction(builder, module, name, results, args.getTypes());
}

/**
 * Iterate over an array.
 *
 * @param builder   operation builder
 * @param location  source location
 * @param array     array to be iterated
 * @param callback  function executed on each iteration
 */
static void iterateArray(mlir::OpBuilder& builder, mlir::Location location, mlir::Value array, std::function<void(mlir::ValueRange)> callback)
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

static mlir::Type castToMostGenericType(mlir::OpBuilder& builder,
																				mlir::ValueRange values,
																				llvm::SmallVectorImpl<mlir::Value>& castedValues)
{
	mlir::Type resultType = nullptr;
	mlir::Type resultBaseType = nullptr;

	for (const auto& value : values)
	{
		mlir::Type type = value.getType();
		mlir::Type baseType = type;

		if (resultType == nullptr)
		{
			resultType = type;
			resultBaseType = type;

			while (resultBaseType.isa<PointerType>())
				resultBaseType = resultBaseType.cast<PointerType>().getElementType();

			continue;
		}

		if (type.isa<PointerType>())
		{
			while (baseType.isa<PointerType>())
				baseType = baseType.cast<PointerType>().getElementType();
		}

		if (resultBaseType.isa<mlir::IndexType>() || baseType.isa<RealType>())
		{
			resultType = type;
			resultBaseType = baseType;
		}
	}

	llvm::SmallVector<mlir::Type, 3> types;

	for (const auto& value : values)
	{
		mlir::Type type = value.getType();

		if (type.isa<PointerType>())
		{
			auto pointerType = type.cast<PointerType>();
			auto shape = pointerType.getShape();
			types.emplace_back(PointerType::get(pointerType.getContext(), pointerType.getAllocationScope(), resultBaseType, shape));
		}
		else
			types.emplace_back(resultBaseType);
	}

	for (const auto& [value, type] : llvm::zip(values, types))
	{
		mlir::Value castedValue = builder.create<CastOp>(value.getLoc(), value, type);
		castedValues.push_back(castedValue);
	}

	return types[0];
}

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
	ModelicaOpConversion(mlir::MLIRContext* ctx,
											 TypeConverter& typeConverter,
											 ModelicaConversionOptions options)
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

	void materializeTargetConversion(mlir::OpBuilder& builder, llvm::SmallVectorImpl<mlir::Value>& values) const
	{
		for (auto& value : values)
			value = materializeTargetConversion(builder, value);
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
			return "i" + std::to_string(convertType(integerType).getIntOrFloatBitWidth());

		if (auto realType = type.dyn_cast<RealType>())
			return "f" + std::to_string(convertType(realType).getIntOrFloatBitWidth());

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

struct FunctionOpLowering : public mlir::OpRewritePattern<FunctionOp>
{
	using mlir::OpRewritePattern<FunctionOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(FunctionOp op, mlir::PatternRewriter& rewriter) const override
	{
		auto function = rewriter.replaceOpWithNewOp<mlir::FuncOp>(op, op.name(), op.getType());

		{
			mlir::OpBuilder::InsertionGuard guard(rewriter);
			auto returnOp = mlir::cast<ReturnOp>(op.getBody().back().getTerminator());
			rewriter.setInsertionPoint(returnOp);
			rewriter.replaceOpWithNewOp<mlir::ReturnOp>(returnOp, returnOp.values());
		}

		auto* body = rewriter.createBlock(&function.getBody(), {}, op.getType().getInputs());
		rewriter.mergeBlocks(&op.getBody().front(), body, body->getArguments());
		return mlir::success();
	}
};

struct MemberAllocOpLowering : public mlir::OpRewritePattern<MemberCreateOp>
{
	using mlir::OpRewritePattern<MemberCreateOp>::OpRewritePattern;

	using LoadReplacer = std::function<void(MemberLoadOp)>;
	using StoreReplacer = std::function<void(MemberStoreOp)>;

	mlir::LogicalResult matchAndRewrite(MemberCreateOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		auto replacers = [&loc, &op, &rewriter]() {
			auto memberType = op.resultType().cast<MemberType>();
			auto pointerType = memberType.toPointerType();

			if (pointerType.isScalar())
			{
				assert(op.dynamicDimensions().empty());
				assert(pointerType.getAllocationScope() == BufferAllocationScope::stack);

				mlir::Value reference = rewriter.create<AllocaOp>(loc, pointerType.getElementType());
				return std::make_pair<LoadReplacer, StoreReplacer>(
						[&rewriter, reference](MemberLoadOp loadOp) -> void {
							mlir::OpBuilder::InsertionGuard guard(rewriter);
							rewriter.setInsertionPoint(loadOp);
							rewriter.replaceOpWithNewOp<LoadOp>(loadOp, reference);
						},
						[&rewriter, reference](MemberStoreOp storeOp) -> void {
							mlir::OpBuilder::InsertionGuard guard(rewriter);
							rewriter.setInsertionPoint(storeOp);
							rewriter.replaceOpWithNewOp<AssignmentOp>(storeOp, storeOp.value(), reference);
						});
			}

			// If we are in the array case, then it may be not sufficient to
			// allocate just the buffer. Instead, if the array has dynamic sizes
			// and they are not initialized, then we need to also allocate a
			// pointer to that buffer, so that we can eventually reassign it if
			// the dimensions change.

			bool hasStaticSize = op.dynamicDimensions().size() == pointerType.getDynamicDimensions();

			if (hasStaticSize)
			{
				mlir::Value reference = allocate(rewriter, loc, pointerType, op.dynamicDimensions());

				return std::make_pair<LoadReplacer, StoreReplacer>(
						[&rewriter, reference](MemberLoadOp loadOp) -> void {
							mlir::OpBuilder::InsertionGuard guard(rewriter);
							rewriter.setInsertionPoint(loadOp);
							rewriter.replaceOp(loadOp, reference);
						},
						[&rewriter, reference](MemberStoreOp storeOp) -> void {
							mlir::OpBuilder::InsertionGuard guard(rewriter);
							rewriter.setInsertionPoint(storeOp);
							rewriter.replaceOpWithNewOp<AssignmentOp>(storeOp, storeOp.value(), reference);
						});
			}

			// The array can change sizes during at runtime. Thus we need to create
			// a pointer to the array currently in use.

			assert(op.dynamicDimensions().empty());
			mlir::Value stackValue = rewriter.create<AllocaOp>(loc, pointerType);

			if (pointerType.getAllocationScope() == BufferAllocationScope::heap)
			{
				// We need to allocate a fake buffer in order to allow the first
				// free operation to operate on a valid memory area.

				PointerType::Shape shape(pointerType.getRank(), 1);
				mlir::Value var = rewriter.create<AllocOp>(loc, pointerType.getElementType(), shape, llvm::None, false);
				var = rewriter.create<PtrCastOp>(loc, var, pointerType);
				rewriter.create<StoreOp>(loc, var, stackValue);
			}

			return std::make_pair<LoadReplacer, StoreReplacer>(
					[&rewriter, stackValue](MemberLoadOp loadOp) -> void {
						mlir::OpBuilder::InsertionGuard guard(rewriter);
						rewriter.setInsertionPoint(loadOp);
						rewriter.replaceOpWithNewOp<LoadOp>(loadOp, stackValue);
					},
					[&rewriter, loc, pointerType, stackValue](MemberStoreOp storeOp) -> void {
						mlir::OpBuilder::InsertionGuard guard(rewriter);
						rewriter.setInsertionPoint(storeOp);

						// The destination array has dynamic and unknown sizes. Thus the
						// buffer has not been allocated yet and we need to create a copy
						// of the source one.

						mlir::Value copy = rewriter.create<ArrayCloneOp>(
								loc, storeOp.value(), pointerType, false);

						// Free the previously allocated memory. This is only apparently in
						// contrast with the above statements: unknown-sized arrays pointers
						// are initialized with a pointer to a 1-element sized array, so that
						// the initial free always operates on valid memory.

						if (pointerType.getAllocationScope() == BufferAllocationScope::heap)
						{
							mlir::Value buffer = rewriter.create<LoadOp>(loc, stackValue);
							rewriter.create<FreeOp>(loc, buffer);
						}

						// Save the descriptor of the new copy into the destination using StoreOp
						rewriter.replaceOpWithNewOp<StoreOp>(storeOp, copy, stackValue);
					});
		};

		LoadReplacer loadReplacer;
		StoreReplacer storeReplacer;
		std::tie(loadReplacer, storeReplacer) = replacers();

		for (auto* user : op->getUsers())
		{
			assert(mlir::isa<MemberLoadOp>(user) || mlir::isa<MemberStoreOp>(user));

			if (auto loadOp = mlir::dyn_cast<MemberLoadOp>(user))
				loadReplacer(loadOp);
			else if (auto storeOp = mlir::dyn_cast<MemberStoreOp>(user))
				storeReplacer(storeOp);
		}

		rewriter.eraseOp(op);
		return mlir::success();
	}

	private:
	static mlir::Value allocate(mlir::OpBuilder& builder, mlir::Location loc, PointerType pointerType, mlir::ValueRange dynamicDimensions)
	{
		auto scope = pointerType.getAllocationScope();
		assert(scope != BufferAllocationScope::unknown);

		if (scope == BufferAllocationScope::stack)
			return builder.create<AllocaOp>(loc, pointerType.getElementType(), pointerType.getShape(), dynamicDimensions);

		// Note that being a member, we will take care of manually freeing
		// the buffer when needed.

		return builder.create<AllocOp>(loc, pointerType.getElementType(), pointerType.getShape(), dynamicDimensions, false);
	}
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

		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

		for (auto size : llvm::enumerate(op.resultType().getShape()))
		{
			if (size.value() == -1)
			{
				mlir::Value index = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(size.index()));
				mlir::Value dim = rewriter.create<DimOp>(loc, op.source(), index);
				dynamicDimensions.push_back(dim);
			}
		}

		mlir::Value result = allocate(rewriter, loc, op.resultType(), dynamicDimensions, op.shouldBeFreed());

		iterateArray(rewriter, loc, op.source(), [&](mlir::ValueRange indexes) {
			mlir::Value value = rewriter.create<LoadOp>(loc, op.source(), indexes);
			value = rewriter.create<CastOp>(loc, value, op.resultType().cast<PointerType>().getElementType());
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
		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
		getArrayDynamicDimensions(rewriter, loc, op.operand(), dynamicDimensions);
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

		auto lhsPointerType = op.lhs().getType().cast<PointerType>();

		if (!lhsPointerType.getElementType().isa<BooleanType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side operand is not an array of booleans");

		if (!op.rhs().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Right-hand side operand is not an array");

		auto rhsPointerType = op.rhs().getType().cast<PointerType>();

		if (!rhsPointerType.getElementType().isa<BooleanType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side operand is not an array of booleans");

		assert(lhsPointerType.getRank() == rhsPointerType.getRank());

		if (options.assertions)
		{
			// Check if the dimensions are compatible
			auto lhsShape = lhsPointerType.getShape();
			auto rhsShape = rhsPointerType.getShape();

			for (size_t i = 0; i < lhsPointerType.getRank(); ++i)
			{
				if (lhsShape[i] == -1 || rhsShape[i] == -1)
				{
					mlir::Value dimensionIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i));
					mlir::Value lhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, op.lhs(), dimensionIndex));
					mlir::Value rhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, op.rhs(), dimensionIndex));
					mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
					rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
				}
			}
		}

		// Allocate the result array
		auto resultType = op.resultType().cast<PointerType>();
		llvm::SmallVector<mlir::Value, 3> lhsDynamicDimensions;
		getArrayDynamicDimensions(rewriter, loc, op.lhs(), lhsDynamicDimensions);
		mlir::Value result = allocate(rewriter, loc, resultType, lhsDynamicDimensions);
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

		auto lhsPointerType = op.lhs().getType().cast<PointerType>();

		if (!lhsPointerType.getElementType().isa<BooleanType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side operand is not an array of booleans");

		if (!op.rhs().getType().isa<PointerType>())
			return rewriter.notifyMatchFailure(op, "Right-hand side operand is not an array");

		auto rhsPointerType = op.rhs().getType().cast<PointerType>();

		if (!rhsPointerType.getElementType().isa<BooleanType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side operand is not an array of booleans");

		assert(lhsPointerType.getRank() == rhsPointerType.getRank());

		if (options.assertions)
		{
			// Check if the dimensions are compatible
			auto lhsShape = lhsPointerType.getShape();
			auto rhsShape = rhsPointerType.getShape();

			for (size_t i = 0; i < lhsPointerType.getRank(); ++i)
			{
				if (lhsShape[i] == -1 || rhsShape[i] == -1)
				{
					mlir::Value dimensionIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i));
					mlir::Value lhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, op.lhs(), dimensionIndex));
					mlir::Value rhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, op.rhs(), dimensionIndex));
					mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
					rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
				}
			}
		}

		// Allocate the result array
		auto resultType = op.resultType().cast<PointerType>();
		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
		getArrayDynamicDimensions(rewriter, loc, op.lhs(), dynamicDimensions);
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

template<typename FromOp>
struct ComparisonOpLowering : public ModelicaOpConversion<FromOp>
{
	using Adaptor = typename ModelicaOpConversion<FromOp>::Adaptor;
	using ModelicaOpConversion<FromOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(FromOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		// Check if the operands are compatible
		if (!isNumeric(op.lhs()) || !isNumeric(op.rhs()))
			return rewriter.notifyMatchFailure(op, "Unsupported types");

		// Cast the operands to the most generic type, in order to avoid
		// information loss.
		llvm::SmallVector<mlir::Value, 3> castedOperands;
		mlir::Type type = castToMostGenericType(rewriter, op->getOperands(), castedOperands);
		Adaptor adaptor(castedOperands);

		// Compute the result
		if (type.isa<mlir::IndexType, BooleanType, IntegerType>())
		{
			mlir::Value result = compareIntegers(rewriter, loc, adaptor.lhs(), adaptor.rhs());
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		if (type.isa<RealType>())
		{
			mlir::Value result = compareReals(rewriter, loc, adaptor.lhs(), adaptor.rhs());
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		return mlir::failure();
	}

	virtual mlir::Value compareIntegers(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const = 0;
	virtual mlir::Value compareReals(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const = 0;
};

struct EqOpLowering: public ComparisonOpLowering<EqOp>
{
	using ComparisonOpLowering<EqOp>::ComparisonOpLowering;

	mlir::Value compareIntegers(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::LLVM::ICmpOp>(
				loc,
				builder.getIntegerType(1),
				mlir::LLVM::ICmpPredicate::eq,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}

	mlir::Value compareReals(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::LLVM::FCmpOp>(
				loc,
				builder.getIntegerType(1),
				mlir::LLVM::FCmpPredicate::oeq,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}
};

struct NotEqOpLowering: public ComparisonOpLowering<NotEqOp>
{
	using ComparisonOpLowering<NotEqOp>::ComparisonOpLowering;

	mlir::Value compareIntegers(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::LLVM::ICmpOp>(
				loc,
				builder.getIntegerType(1),
				mlir::LLVM::ICmpPredicate::ne,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}

	mlir::Value compareReals(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::LLVM::FCmpOp>(
				loc,
				builder.getIntegerType(1),
				mlir::LLVM::FCmpPredicate::one,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}
};

struct GtOpLowering: public ComparisonOpLowering<GtOp>
{
	using ComparisonOpLowering<GtOp>::ComparisonOpLowering;

	mlir::Value compareIntegers(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::LLVM::ICmpOp>(
				loc,
				builder.getIntegerType(1),
				mlir::LLVM::ICmpPredicate::sgt,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}

	mlir::Value compareReals(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::LLVM::FCmpOp>(
				loc,
				builder.getIntegerType(1),
				mlir::LLVM::FCmpPredicate::ogt,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}
};

struct GteOpLowering: public ComparisonOpLowering<GteOp>
{
	using ComparisonOpLowering<GteOp>::ComparisonOpLowering;

	mlir::Value compareIntegers(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::LLVM::ICmpOp>(
				loc,
				builder.getIntegerType(1),
				mlir::LLVM::ICmpPredicate::sge,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}

	mlir::Value compareReals(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::LLVM::FCmpOp>(
				loc,
				builder.getIntegerType(1),
				mlir::LLVM::FCmpPredicate::oge,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}
};

struct LtOpLowering: public ComparisonOpLowering<LtOp>
{
	using ComparisonOpLowering<LtOp>::ComparisonOpLowering;

	mlir::Value compareIntegers(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::LLVM::ICmpOp>(
				loc,
				builder.getIntegerType(1),
				mlir::LLVM::ICmpPredicate::slt,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}

	mlir::Value compareReals(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::LLVM::FCmpOp>(
				loc,
				builder.getIntegerType(1),
				mlir::LLVM::FCmpPredicate::olt,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}
};

struct LteOpLowering: public ComparisonOpLowering<LteOp>
{
	using ComparisonOpLowering<LteOp>::ComparisonOpLowering;

	mlir::Value compareIntegers(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::LLVM::ICmpOp>(
				loc,
				builder.getIntegerType(1),
				mlir::LLVM::ICmpPredicate::sle,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}

	mlir::Value compareReals(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::LLVM::FCmpOp>(
				loc,
				builder.getIntegerType(1),
				mlir::LLVM::FCmpPredicate::ole,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
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
		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
		getArrayDynamicDimensions(rewriter, loc, op.operand(), dynamicDimensions);
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

		// Cast the operands to the most generic type, in order to avoid
		// information loss.
		llvm::SmallVector<mlir::Value, 3> castedOperands;
		mlir::Type type = castToMostGenericType(rewriter, op->getOperands(), castedOperands);
		materializeTargetConversion(rewriter, castedOperands);
		Adaptor transformed(castedOperands);

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

		if (options.assertions)
		{
			// Check if the dimensions are compatible
			auto lhsShape = lhsPointerType.getShape();
			auto rhsShape = rhsPointerType.getShape();

			assert(lhsPointerType.getRank() == rhsPointerType.getRank());

			for (size_t i = 0; i < lhsPointerType.getRank(); ++i)
			{
				if (lhsShape[i] == -1 || rhsShape[i] == -1)
				{
					mlir::Value dimensionIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i));
					mlir::Value lhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, op.lhs(), dimensionIndex));
					mlir::Value rhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, op.rhs(), dimensionIndex));
					mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
					rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
				}
			}
		}

		// Allocate the result array
		auto resultType = op.resultType().cast<PointerType>();
		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
		getArrayDynamicDimensions(rewriter, loc, op.lhs(), dynamicDimensions);
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

		// Cast the operands to the most generic type, in order to avoid
		// information loss.
		llvm::SmallVector<mlir::Value, 3> castedOperands;
		mlir::Type type = castToMostGenericType(rewriter, op->getOperands(), castedOperands);
		materializeTargetConversion(rewriter, castedOperands);
		Adaptor transformed(castedOperands);

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

		if (options.assertions)
		{
			// Check if the dimensions are compatible
			auto lhsShape = lhsPointerType.getShape();
			auto rhsShape = rhsPointerType.getShape();

			assert(lhsPointerType.getRank() == rhsPointerType.getRank());

			for (size_t i = 0; i < lhsPointerType.getRank(); ++i)
			{
				if (lhsShape[i] == -1 || rhsShape[i] == -1)
				{
					mlir::Value dimensionIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i));
					mlir::Value lhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, op.lhs(), dimensionIndex));
					mlir::Value rhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, op.rhs(), dimensionIndex));
					mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
					rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
				}
			}
		}

		// Allocate the result array
		auto resultType = op.resultType().cast<PointerType>();
		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
		getArrayDynamicDimensions(rewriter, loc, op.lhs(), dynamicDimensions);
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
		llvm::SmallVector<mlir::Value, 3> castedOperands;
		mlir::Type type = castToMostGenericType(rewriter, op->getOperands(), castedOperands);
		materializeTargetConversion(rewriter, castedOperands);
		Adaptor transformed(castedOperands);

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
		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
		getArrayDynamicDimensions(rewriter, loc, array, dynamicDimensions);
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

		assert(lhsPointerType.getRank() == 1);
		assert(rhsPointerType.getRank() == 1);

		if (options.assertions)
		{
			// Check if the dimensions are compatible
			auto lhsShape = lhsPointerType.getShape();
			auto rhsShape = rhsPointerType.getShape();

			if (lhsShape[0] == -1 || rhsShape[0] == -1)
			{
				mlir::Value dimensionIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
				mlir::Value lhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, op.lhs(), dimensionIndex));
				mlir::Value rhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, op.rhs(), dimensionIndex));
				mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
				rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
			}
		}

		// Compute the result
		mlir::Type type = op.resultType();
		Adaptor transformed(operands);

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
 * [ x1 ] * [ y11, y12 ] = [ (x1 * y11 + x2 * y21 + x3 * y31), (x1 * y12 + x2 * y22 + x3 * y32) ]
 * [ x2	]		[ y21, y22 ]
 * [ x3	]		[ y31, y32 ]
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

		assert(lhsPointerType.getRank() == 1);
		assert(rhsPointerType.getRank() == 2);

		if (options.assertions)
		{
			// Check if the dimensions are compatible
			auto lhsShape = lhsPointerType.getShape();
			auto rhsShape = rhsPointerType.getShape();

			if (lhsShape[0] == -1 || rhsShape[0] == -1)
			{
				mlir::Value zero = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
				mlir::Value lhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, op.lhs(), zero));
				mlir::Value rhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, op.rhs(), zero));
				mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
				rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
			}
		}

		// Allocate the result array
		Adaptor transformed(operands);
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

		assert(lhsPointerType.getRank() == 2);
		assert(rhsPointerType.getRank() == 1);

		if (options.assertions)
		{
			// Check if the dimensions are compatible
			auto lhsShape = lhsPointerType.getShape();
			auto rhsShape = rhsPointerType.getShape();

			if (lhsShape[1] == -1 || rhsShape[0] == -1)
			{
				mlir::Value zero = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
				mlir::Value one = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));
				mlir::Value lhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, op.lhs(), one));
				mlir::Value rhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, op.rhs(), zero));
				mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
				rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
			}
		}

		// Allocate the result array
		Adaptor transformed(operands);
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

		assert(lhsPointerType.getRank() == 2);
		assert(rhsPointerType.getRank() == 2);

		if (options.assertions)
		{
			// Check if the dimensions are compatible
			auto lhsShape = lhsPointerType.getShape();
			auto rhsShape = rhsPointerType.getShape();

			if (lhsShape[1] == -1 || rhsShape[0] == -1)
			{
				mlir::Value zero = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
				mlir::Value one = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));
				mlir::Value lhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, op.lhs(), one));
				mlir::Value rhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, op.rhs(), zero));
				mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
				rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
			}
		}

		// Allocate the result array
		Adaptor transformed(operands);
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
		llvm::SmallVector<mlir::Value, 3> castedOperands;
		mlir::Type type = castToMostGenericType(rewriter, op->getOperands(), castedOperands);
		materializeTargetConversion(rewriter, castedOperands);
		Adaptor transformed(castedOperands);

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
		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
		getArrayDynamicDimensions(rewriter, loc, op.lhs(), dynamicDimensions);
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

		assert(basePointerType.getRank() == 2);

		if (options.assertions)
		{
			// Check if the matrix is a square one
			auto shape = basePointerType.getShape();

			if (shape[0] == -1 || shape[1] == -1)
			{
				mlir::Value zero = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
				mlir::Value one = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));
				mlir::Value lhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, op.base(), one));
				mlir::Value rhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, op.base(), zero));
				mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
				rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Base matrix is not squared"));
			}
		}

		// Allocate the result array
		auto resultPointerType = op.resultType().cast<PointerType>();
		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
		getArrayDynamicDimensions(rewriter, loc, op.base(), dynamicDimensions);
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

struct ZerosOpLowering: public ModelicaOpConversion<ZerosOp>
{
	using ModelicaOpConversion<ZerosOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(ZerosOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto pointerType = op.resultType().cast<PointerType>();

		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

		for (auto size : llvm::enumerate(pointerType.getShape()))
			if (size.value() == -1)
				dynamicDimensions.push_back(rewriter.create<CastOp>(loc, op.sizes()[size.index()], rewriter.getIndexType()));

		mlir::Value result = allocate(rewriter, loc, pointerType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		llvm::SmallVector<mlir::Value, 3> args;

		args.push_back(rewriter.create<PtrCastOp>(
				loc, result,
				result.getType().cast<PointerType>().toUnsized()));

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("zeros", args),
				llvm::None,
				args);

		rewriter.create<CallOp>(loc, callee.getName(), llvm::None, args);
		return mlir::success();
	}
};

struct OnesOpLowering: public ModelicaOpConversion<OnesOp>
{
	using ModelicaOpConversion<OnesOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(OnesOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto pointerType = op.resultType().cast<PointerType>();

		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

		for (auto size : llvm::enumerate(pointerType.getShape()))
			if (size.value() == -1)
				dynamicDimensions.push_back(rewriter.create<CastOp>(loc, op.sizes()[size.index()], rewriter.getIndexType()));

		mlir::Value result = allocate(rewriter, loc, pointerType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		llvm::SmallVector<mlir::Value, 3> args;

		args.push_back(rewriter.create<PtrCastOp>(
				loc, result,
				result.getType().cast<PointerType>().toUnsized()));

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("ones", args),
				llvm::None,
				args);

		rewriter.create<CallOp>(loc, callee.getName(), llvm::None, args);
		return mlir::success();
	}
};

struct LinspaceOpLowering: public ModelicaOpConversion<LinspaceOp>
{
	using ModelicaOpConversion<LinspaceOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(LinspaceOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto pointerType = op.resultType().cast<PointerType>();

		assert(pointerType.getRank() == 1);
		mlir::Value size = rewriter.create<CastOp>(loc, op.steps(), rewriter.getIndexType());

		mlir::Value result = allocate(rewriter, loc, pointerType, size);
		rewriter.replaceOp(op, result);

		llvm::SmallVector<mlir::Value, 3> args;

		args.push_back(rewriter.create<PtrCastOp>(
				loc, result,
				result.getType().cast<PointerType>().toUnsized()));

		args.push_back(rewriter.create<CastOp>(
				loc, op.start(),
				RealType::get(op->getContext())));

		args.push_back(rewriter.create<CastOp>(
				loc, op.end(),
				RealType::get(op->getContext())));

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("linspace", args),
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

struct MinOpArrayLowering: public ModelicaOpConversion<MinOp>
{
	using ModelicaOpConversion<MinOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(MinOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		if (op.getNumOperands() != 1)
			return rewriter.notifyMatchFailure(op, "Operand is not an array");

		mlir::Value operand = op.values()[0];

		// If there is just one operand, then it is for sure an array, thanks
		// to the operation verification.

		assert(operand.getType().isa<PointerType>() &&
				isNumericType(operand.getType().cast<PointerType>().getElementType()));

		auto pointerType = operand.getType().cast<PointerType>();
		operand = rewriter.create<PtrCastOp>(loc, operand, pointerType.toUnsized());

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("min", operand),
				pointerType.getElementType(),
				operand);

		auto call = rewriter.create<CallOp>(loc, callee.getName(), pointerType.getElementType(), operand);
		assert(call.getNumResults() == 1);
		rewriter.replaceOpWithNewOp<CastOp>(op, call->getResult(0), op.resultType());

		return mlir::success();
	}
};

struct MinOpScalarsLowering: public ModelicaOpConversion<MinOp>
{
	using ModelicaOpConversion<MinOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(MinOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		if (op.getNumOperands() != 2)
			return rewriter.notifyMatchFailure(op, "Operands are not scalars");

		// If there are two operands then they are for sure scalars, thanks
		// to the operation verification.

		mlir::ValueRange values = op.values();
		assert(isNumeric(values[0]) && isNumeric(values[1]));

		llvm::SmallVector<mlir::Value, 3> castedOperands;
		mlir::Type type = castToMostGenericType(rewriter, values, castedOperands);
		materializeTargetConversion(rewriter, castedOperands);
		Adaptor transformed(castedOperands);

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("min", transformed.values()),
				type,
				transformed.values());

		auto call = rewriter.create<CallOp>(loc, callee.getName(), type, transformed.values());
		assert(call.getNumResults() == 1);
		rewriter.replaceOpWithNewOp<CastOp>(op, call->getResult(0), op.resultType());

		return mlir::success();
	}
};

struct MaxOpArrayLowering: public ModelicaOpConversion<MaxOp>
{
	using ModelicaOpConversion<MaxOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(MaxOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		if (op.getNumOperands() != 1)
			return rewriter.notifyMatchFailure(op, "Operand is not an array");

		mlir::Value operand = op.values()[0];

		// If there is just one operand, then it is for sure an array, thanks
		// to the operation verification.

		assert(operand.getType().isa<PointerType>() &&
					 isNumericType(operand.getType().cast<PointerType>().getElementType()));

		auto pointerType = operand.getType().cast<PointerType>();
		operand = rewriter.create<PtrCastOp>(loc, operand, pointerType.toUnsized());

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("max", operand),
				pointerType.getElementType(),
				operand);

		auto call = rewriter.create<CallOp>(loc, callee.getName(), pointerType.getElementType(), operand);
		assert(call.getNumResults() == 1);
		rewriter.replaceOpWithNewOp<CastOp>(op, call->getResult(0), op.resultType());

		return mlir::success();
	}
};

struct MaxOpScalarsLowering: public ModelicaOpConversion<MaxOp>
{
	using ModelicaOpConversion<MaxOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(MaxOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		if (op.getNumOperands() != 2)
			return rewriter.notifyMatchFailure(op, "Operands are not scalars");

		// If there are two operands then they are for sure scalars, thanks
		// to the operation verification.

		mlir::ValueRange values = op.values();
		assert(isNumeric(values[0]) && isNumeric(values[1]));

		llvm::SmallVector<mlir::Value, 3> castedOperands;
		mlir::Type type = castToMostGenericType(rewriter, values, castedOperands);
		materializeTargetConversion(rewriter, castedOperands);
		Adaptor transformed(castedOperands);

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("max", transformed.values()),
				type,
				transformed.values());

		auto call = rewriter.create<CallOp>(loc, callee.getName(), type, transformed.values());
		assert(call.getNumResults() == 1);
		rewriter.replaceOpWithNewOp<CastOp>(op, call->getResult(0), op.resultType());

		return mlir::success();
	}
};

struct SumOpLowering: public ModelicaOpConversion<SumOp>
{
	using ModelicaOpConversion<SumOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(SumOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto pointerType = op.array().getType().cast<PointerType>();

		mlir::Value arg = rewriter.create<PtrCastOp>(
				loc, op.array(),
				op.array().getType().cast<PointerType>().toUnsized());

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("sum", arg),
				pointerType.getElementType(),
				arg);

		auto call = rewriter.create<CallOp>(loc, callee.getName(), pointerType.getElementType(), arg);
		assert(call.getNumResults() == 1);
		rewriter.replaceOpWithNewOp<CastOp>(op, call->getResult(0), op.resultType());

		return mlir::success();
	}
};

struct ProductOpLowering: public ModelicaOpConversion<ProductOp>
{
	using ModelicaOpConversion<ProductOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(ProductOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto pointerType = op.array().getType().cast<PointerType>();

		mlir::Value arg = rewriter.create<PtrCastOp>(
				loc, op.array(),
				op.array().getType().cast<PointerType>().toUnsized());

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("product", arg),
				pointerType.getElementType(),
				arg);

		auto call = rewriter.create<CallOp>(loc, callee.getName(), pointerType.getElementType(), arg);
		assert(call.getNumResults() == 1);
		rewriter.replaceOpWithNewOp<CastOp>(op, call->getResult(0), op.resultType());

		return mlir::success();
	}
};

struct TransposeOpLowering: public ModelicaOpConversion<TransposeOp>
{
	using ModelicaOpConversion<TransposeOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(TransposeOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto pointerType = op.resultType().cast<PointerType>();
		auto shape = pointerType.getShape();

		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

		for (auto size : llvm::enumerate(shape))
		{
			if (size.value() == -1)
			{
				mlir::Value index = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(shape.size() - size.index() - 1));
				mlir::Value dim = rewriter.create<DimOp>(loc, op.matrix(), index);
				dynamicDimensions.push_back(dim);
			}
		}

		mlir::Value result = allocate(rewriter, loc, pointerType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		llvm::SmallVector<mlir::Value, 3> args;

		args.push_back(rewriter.create<PtrCastOp>(
				loc, result,
				result.getType().cast<PointerType>().toUnsized()));

		args.push_back(rewriter.create<PtrCastOp>(
				loc, op.matrix(),
				op.matrix().getType().cast<PointerType>().toUnsized()));

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("transpose", args),
				llvm::None,
				args);

		rewriter.create<CallOp>(loc, callee.getName(), llvm::None, args);
		return mlir::success();
	}
};

struct SymmetricOpLowering: public ModelicaOpConversion<SymmetricOp>
{
	using ModelicaOpConversion<SymmetricOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(SymmetricOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto pointerType = op.matrix().getType().cast<PointerType>();
		auto shape = pointerType.getShape();

		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

		for (auto size : llvm::enumerate(shape))
		{
			if (size.value() == -1)
			{
				mlir::Value index = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(size.index()));
				dynamicDimensions.push_back(rewriter.create<DimOp>(loc, op.matrix(), index));
			}
		}

		if (options.assertions)
		{
			// Check if the matrix is a square one
			if (shape[0] == -1 || shape[1] == -1)
			{
				mlir::Value zero = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
				mlir::Value one = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));
				mlir::Value lhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, op.matrix(), one));
				mlir::Value rhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, op.matrix(), zero));
				mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
				rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Base matrix is not squared"));
			}
		}

		mlir::Value result = allocate(rewriter, loc, pointerType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		llvm::SmallVector<mlir::Value, 3> args;

		args.push_back(rewriter.create<PtrCastOp>(
				loc, result,
				result.getType().cast<PointerType>().toUnsized()));

		args.push_back(rewriter.create<PtrCastOp>(
				loc, op.matrix(),
				op.matrix().getType().cast<PointerType>().toUnsized()));

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("symmetric", args),
				llvm::None,
				args);

		rewriter.create<CallOp>(loc, callee.getName(), llvm::None, args);
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
		rewriter.setInsertionPointAfter(thenTerminator);

		// The yielded values must be converted to their target types
		llvm::SmallVector<mlir::Value, 3> thenYieldValues;

		for (mlir::Value value : thenTerminator.values())
			thenYieldValues.push_back(value);

		rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(thenTerminator, thenYieldValues);

		// If the operation also has an "else" block, also replace its
		// Modelica::YieldOp terminator with a SCF::YieldOp.

		if (hasElseBlock)
		{
			mlir::Block* elseBlock = &ifOp.elseRegion().front();
			auto elseTerminator = mlir::cast<YieldOp>(elseBlock->getTerminator());
			rewriter.setInsertionPointAfter(elseTerminator);

			// The yielded values must be converted to their target types
			llvm::SmallVector<mlir::Value, 3> elseYieldValues;

			for (mlir::Value value : elseTerminator.values())
				elseYieldValues.push_back(value);

			rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(elseTerminator, elseYieldValues);
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
		rewriter.replaceOpWithNewOp<mlir::BranchOp>(bodyYieldOp, stepBlock, bodyYieldOp.values());

		// Branch to the condition check after incrementing the induction variable
		rewriter.setInsertionPointToEnd(stepBlock);
		auto stepYieldOp = mlir::cast<YieldOp>(stepBlock->getTerminator());
		rewriter.replaceOpWithNewOp<mlir::BranchOp>(stepYieldOp, conditionBlock, stepYieldOp.values());

		// Create stack save & restore operations
		rewriter.setInsertionPointToStart(bodyBlock);
		mlir::Value stackSave = rewriter.create<mlir::LLVM::StackSaveOp>(loc, mlir::LLVM::LLVMPointerType::get(rewriter.getIntegerType(8)));
		rewriter.setInsertionPoint(bodyBlock->getTerminator());
		rewriter.create<mlir::LLVM::StackRestoreOp>(loc, stackSave);

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
		rewriter.replaceOpWithNewOp<mlir::BranchOp>(bodyYieldOp, stepBlock, bodyYieldOp.values());

		// Branch to the condition check after incrementing the induction variable
		rewriter.setInsertionPointToEnd(stepBlock);
		auto stepYieldOp = mlir::cast<YieldOp>(stepBlock->getTerminator());
		rewriter.replaceOpWithNewOp<mlir::BranchOp>(stepYieldOp, conditionBlock, stepYieldOp.values());

		// Create stack save & restore operations
		rewriter.setInsertionPointToStart(bodyBlock);
		mlir::Value stackSave = rewriter.create<mlir::LLVM::StackSaveOp>(loc, mlir::LLVM::LLVMPointerType::get(rewriter.getIntegerType(8)));
		rewriter.setInsertionPoint(bodyBlock->getTerminator());
		rewriter.create<mlir::LLVM::StackRestoreOp>(loc, stackSave);

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

		// Create stack save & restore operations
		rewriter.setInsertionPointToStart(body);
		mlir::Value stackSave = rewriter.create<mlir::LLVM::StackSaveOp>(loc, mlir::LLVM::LLVMPointerType::get(rewriter.getIntegerType(8)));
		rewriter.setInsertionPoint(body->getTerminator());
		rewriter.create<mlir::LLVM::StackRestoreOp>(loc, stackSave);

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

class FunctionConversionPass: public mlir::PassWrapper<FunctionConversionPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	void getDependentDialects(mlir::DialectRegistry &registry) const override
	{
		registry.insert<mlir::BuiltinDialect>();
		registry.insert<mlir::StandardOpsDialect>();
	}

	void runOnOperation() override
	{
		auto module = getOperation();

		mlir::ConversionTarget target(getContext());
		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });
		target.addIllegalOp<FunctionOp>();

		// Provide the set of patterns that will lower the Modelica operations
		mlir::OwningRewritePatternList patterns(&getContext());
		patterns.insert<FunctionOpLowering>(&getContext());

		// With the target and rewrite patterns defined, we can now attempt the
		// conversion. The conversion will signal failure if any of our "illegal"
		// operations were not converted successfully.

		if (failed(applyPartialConversion(module, target, std::move(patterns))))
		{
			mlir::emitError(module.getLoc(), "Error in converting the Modelica functions\n");
			signalPassFailure();
		}
	}
};

std::unique_ptr<mlir::Pass> modelica::codegen::createFunctionConversionPass()
{
	return std::make_unique<FunctionConversionPass>();
}

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
	patterns.insert<MemberAllocOpLowering>(context);

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
			ZerosOpLowering,
			OnesOpLowering,
			LinspaceOpLowering,
			FillOpLowering,
			MinOpArrayLowering,
			MinOpScalarsLowering,
			MaxOpArrayLowering,
			MaxOpScalarsLowering,
			SumOpLowering,
			ProductOpLowering,
			TransposeOpLowering,
			SymmetricOpLowering>(context, typeConverter, options);
}

class ModelicaConversionPass: public mlir::PassWrapper<ModelicaConversionPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	explicit ModelicaConversionPass(ModelicaConversionOptions options, unsigned int bitWidth)
			: options(std::move(options)),
				bitWidth(bitWidth)
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
		    MemberCreateOp, MemberLoadOp, MemberStoreOp,
		    ConstantOp, PackOp, ExtractOp,
				AssignmentOp,
				CallOp,
				FillOp,
				PrintOp,
				ArrayCloneOp,
				NotOp, AndOp, OrOp,
				EqOp, NotEqOp, GtOp, GteOp, LtOp, LteOp,
				NegateOp, AddOp, SubOp, MulOp, DivOp, PowOp,
				NDimsOp, SizeOp, IdentityOp, DiagonalOp, ZerosOp, OnesOp, LinspaceOp, FillOp,
				MinOp, MaxOp, SumOp, ProductOp, TransposeOp, SymmetricOp>();

		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });

		mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
		TypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);

		// Provide the set of patterns that will lower the Modelica operations
		mlir::OwningRewritePatternList patterns(&getContext());
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
	unsigned int bitWidth;
};

std::unique_ptr<mlir::Pass> modelica::codegen::createModelicaConversionPass(ModelicaConversionOptions options, unsigned int bitWidth)
{
	return std::make_unique<ModelicaConversionPass>(options, bitWidth);
}

class LowerToCFGPass: public mlir::PassWrapper<LowerToCFGPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	explicit LowerToCFGPass(ModelicaConversionOptions options, unsigned int bitWidth)
			: options(std::move(options)), bitWidth(bitWidth)
	{
	}

	void getDependentDialects(mlir::DialectRegistry &registry) const override
	{
		registry.insert<mlir::StandardOpsDialect>();
		registry.insert<mlir::scf::SCFDialect>();
	}

	void runOnOperation() override
	{
		auto module = getOperation();

		mlir::ConversionTarget target(getContext());

		target.addIllegalOp<mlir::scf::ForOp, mlir::scf::IfOp, mlir::scf::ParallelOp, mlir::scf::WhileOp>();
		target.addIllegalOp<IfOp, ForOp, BreakableForOp, BreakableWhileOp>();
		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });

		mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
		TypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);

		// Provide the set of patterns that will lower the Modelica operations
		mlir::OwningRewritePatternList patterns(&getContext());
		populateModelicaControlFlowConversionPatterns(patterns, &getContext(), typeConverter, options);
		mlir::populateLoopToStdConversionPatterns(patterns);

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
	unsigned int bitWidth;
};

std::unique_ptr<mlir::Pass> modelica::codegen::createLowerToCFGPass(ModelicaConversionOptions options, unsigned int bitWidth)
{
	return std::make_unique<LowerToCFGPass>(options, bitWidth);
}
