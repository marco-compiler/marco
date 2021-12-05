#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Transforms/DialectConversion.h>
#include <marco/mlirlowerer/dialects/modelica/ModelicaDialect.h>
#include <marco/mlirlowerer/passes/ModelicaConversion.h>
#include <marco/mlirlowerer/passes/TypeConverter.h>
#include <numeric>

using namespace marco::codegen;
using namespace modelica;

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
	assert(array.getType().isa<ArrayType>());
	auto arrayType = array.getType().cast<ArrayType>();
	auto shape = arrayType.getShape();

	for (const auto& dimension : llvm::enumerate(arrayType.getShape()))
	{
		if (dimension.value() == -1)
		{
			mlir::Value dim = builder.create<ConstantOp>(loc, builder.getIndexAttr(dimension.index()));
			dimensions.push_back(builder.create<DimOp>(loc, array, dim));
		}
	}
}

static mlir::Value allocate(mlir::OpBuilder& builder, mlir::Location location, ArrayType arrayType, mlir::ValueRange dynamicDimensions = llvm::None, bool shouldBeFreed = true)
{
	if (arrayType.getAllocationScope() == BufferAllocationScope::unknown)
		arrayType = arrayType.toMinAllowedAllocationScope();

	if (arrayType.getAllocationScope() == BufferAllocationScope::stack)
		return builder.create<AllocaOp>(location, arrayType.getElementType(), arrayType.getShape(), dynamicDimensions);

	return builder.create<AllocOp>(location, arrayType.getElementType(), arrayType.getShape(), dynamicDimensions, shouldBeFreed);
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
	assert(array.getType().isa<ArrayType>());
	auto arrayType = array.getType().cast<ArrayType>();

	mlir::Value zero = builder.create<mlir::ConstantOp>(location, builder.getIndexAttr(0));
	mlir::Value one = builder.create<mlir::ConstantOp>(location, builder.getIndexAttr(1));

	llvm::SmallVector<mlir::Value, 3> lowerBounds(arrayType.getRank(), zero);
	llvm::SmallVector<mlir::Value, 3> upperBounds;
	llvm::SmallVector<mlir::Value, 3> steps(arrayType.getRank(), one);

	for (unsigned int i = 0, e = arrayType.getRank(); i < e; ++i)
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

			while (resultBaseType.isa<ArrayType>())
				resultBaseType = resultBaseType.cast<ArrayType>().getElementType();

			continue;
		}

		if (type.isa<ArrayType>())
		{
			while (baseType.isa<ArrayType>())
				baseType = baseType.cast<ArrayType>().getElementType();
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

		if (type.isa<ArrayType>())
		{
			auto arrayType = type.cast<ArrayType>();
			auto shape = arrayType.getShape();
			types.emplace_back(ArrayType::get(arrayType.getContext(), arrayType.getAllocationScope(), resultBaseType, shape));
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
class ModelicaOpConversion : public mlir::OpConversionPattern<FromOp>
{
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

	[[nodiscard]] marco::codegen::TypeConverter& typeConverter() const
	{
		return *static_cast<marco::codegen::TypeConverter *>(this->getTypeConverter());
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

	[[nodiscard]] std::string getMangledFunctionName(llvm::StringRef name, llvm::Optional<mlir::Type> returnType, mlir::ValueRange args) const
	{
		return getMangledFunctionName(name, returnType, args.getTypes());
	}

	[[nodiscard]] std::string getMangledFunctionName(llvm::StringRef name, llvm::Optional<mlir::Type> returnType, mlir::TypeRange argsTypes) const
	{
		std::string resultType = returnType.hasValue() ? getMangledType(*returnType) : "void";

		return "_M" + name.str() +
					 "_" + resultType +
					 std::accumulate(
							 argsTypes.begin(), argsTypes.end(), std::string(),
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

		if (auto arrayType = type.dyn_cast<UnsizedArrayType>())
			return "a" + getMangledType(arrayType.getElementType());

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
		mlir::Location loc = op->getLoc();
		auto function = rewriter.replaceOpWithNewOp<mlir::FuncOp>(op, op.name(), op.getType());

		rewriter.inlineRegionBefore(op.getBody(),function.getBody(), function.getBody().begin());

		//auto* body = rewriter.createBlock(&function.getBody(), {}, op.getType().getInputs());
		//rewriter.mergeBlocks(&op.getBody().front(), body, body->getArguments());

		// Map the members for faster access
		llvm::StringMap<MemberCreateOp> members;

		function->walk([&members](MemberCreateOp member) {
			members[member.name()] = member;
		});

		mlir::Block* lastBodyBlock = &function.getBody().back();
		auto functionTerminator = mlir::cast<FunctionTerminatorOp>(lastBodyBlock->getTerminator());
		mlir::Block* returnBlock = rewriter.splitBlock(lastBodyBlock, functionTerminator->getIterator());
		rewriter.setInsertionPointToEnd(lastBodyBlock);
		rewriter.replaceOpWithNewOp<mlir::BranchOp>(functionTerminator, returnBlock);

		rewriter.setInsertionPointToEnd(returnBlock);

		// TODO: free protected members

		llvm::SmallVector<mlir::Value, 1> results;

		for (const auto& name : op.resultsNames())
		{
			auto member = members.lookup(name.cast<mlir::StringAttr>().getValue()).getResult();
			auto memberType = member.getType().cast<MemberType>();
			mlir::Value value = rewriter.create<MemberLoadOp>(loc, memberType.unwrap(), member);
			results.push_back(value);
		}

		rewriter.create<mlir::ReturnOp>(loc, results);
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
			auto arrayType = memberType.toArrayType();

			if (arrayType.isScalar())
			{
				assert(op.dynamicDimensions().empty());
				assert(arrayType.getAllocationScope() == BufferAllocationScope::stack);

				mlir::Value reference = rewriter.create<AllocaOp>(loc, arrayType.getElementType());
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

			bool hasStaticSize = op.dynamicDimensions().size() == arrayType.getDynamicDimensions();

			if (hasStaticSize)
			{
				mlir::Value reference = allocate(rewriter, loc, arrayType, op.dynamicDimensions());

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
			mlir::Value stackValue = rewriter.create<AllocaOp>(loc, arrayType);

			if (arrayType.getAllocationScope() == BufferAllocationScope::heap)
			{
				// We need to allocate a fake buffer in order to allow the first
				// free operation to operate on a valid memory area.

				ArrayType::Shape shape(arrayType.getRank(), 0);
				mlir::Value var = rewriter.create<AllocOp>(loc, arrayType.getElementType(), shape, llvm::None, false);
				var = rewriter.create<ArrayCastOp>(loc, var, arrayType);
				rewriter.create<StoreOp>(loc, var, stackValue);
			}

			return std::make_pair<LoadReplacer, StoreReplacer>(
					[&rewriter, stackValue](MemberLoadOp loadOp) -> void {
						mlir::OpBuilder::InsertionGuard guard(rewriter);
						rewriter.setInsertionPoint(loadOp);
						rewriter.replaceOpWithNewOp<LoadOp>(loadOp, stackValue);
					},
					[&rewriter, loc, arrayType, stackValue](MemberStoreOp storeOp) -> void {
						mlir::OpBuilder::InsertionGuard guard(rewriter);
						rewriter.setInsertionPoint(storeOp);

						// The destination array has dynamic and unknown sizes. Thus the
						// buffer has not been allocated yet and we need to create a copy
						// of the source one.

						// The function input arguments must be cloned, in order to avoid
						// inputs modifications.
						bool canSourceBeForwarded = !storeOp.value().isa<mlir::BlockArgument>();

            // The deallocation is set as manually handled, because the value
            // will be deallocated before an eventual new value will be set into
            // the dynamic array.
            bool shouldBeDeallocated = false;

						mlir::Value copy = rewriter.create<ArrayCloneOp>(
								loc, storeOp.value(), arrayType, shouldBeDeallocated, canSourceBeForwarded);

						// Free the previously allocated memory. This is only apparently in
						// contrast with the above statements: unknown-sized arrays pointers
						// are initialized with a pointer to a 1-element sized array, so that
						// the initial free always operates on valid memory.

						if (arrayType.getAllocationScope() == BufferAllocationScope::heap)
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
	static mlir::Value allocate(mlir::OpBuilder& builder, mlir::Location loc, ArrayType arrayType, mlir::ValueRange dynamicDimensions)
	{
		auto scope = arrayType.getAllocationScope();
		assert(scope != BufferAllocationScope::unknown);

		if (scope == BufferAllocationScope::stack)
			return builder.create<AllocaOp>(loc, arrayType.getElementType(), arrayType.getShape(), dynamicDimensions);

		// Note that being a member, we will take care of manually freeing
		// the buffer when needed.

		return builder.create<AllocOp>(loc, arrayType.getElementType(), arrayType.getShape(), dynamicDimensions, false);
	}
};

struct ConstantOpLowering : public ModelicaOpConversion<ConstantOp>
{
	using ModelicaOpConversion<ConstantOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(ConstantOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		auto attribute = convertAttribute(rewriter, op.resultType(), op.value());

		//if (!attribute)
		//	return rewriter.notifyMatchFailure(op, "Unknown attribute type");

		rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, attribute);
		return mlir::success();
	}

	private:
	mlir::Attribute convertAttribute(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Attribute attribute) const
	{
		if (attribute.getType().isa<mlir::IndexType>())
			return attribute;

		resultType = getTypeConverter()->convertType(resultType);

		if (auto booleanAttribute = attribute.dyn_cast<BooleanAttribute>())
			return builder.getBoolAttr(booleanAttribute.getValue());

		if (auto integerAttribute = attribute.dyn_cast<IntegerAttribute>())
			return builder.getIntegerAttr(resultType, integerAttribute.getValue());

		assert(attribute.isa<RealAttribute>());
		return builder.getFloatAttr(resultType, attribute.cast<RealAttribute>().getValue());
	}
};

/**
 * Store a scalar value.
 */
struct AssignmentOpScalarLowering : public mlir::OpRewritePattern<AssignmentOp>
{
	using mlir::OpRewritePattern<AssignmentOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(AssignmentOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		if (!isNumeric(op.source()))
			return rewriter.notifyMatchFailure(op, "Source value has not a numeric type");

		auto destinationBaseType = op.destination().getType().cast<ArrayType>().getElementType();
		mlir::Value value = rewriter.create<CastOp>(loc, op.source(), destinationBaseType);
		rewriter.replaceOpWithNewOp<StoreOp>(op, value, op.destination());

		return mlir::success();
	}
};

/**
 * Store (copy) an array value.
 */
struct AssignmentOpArrayLowering : public mlir::OpRewritePattern<AssignmentOp>
{
	using mlir::OpRewritePattern<AssignmentOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(AssignmentOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		if (!op.source().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Source value is not an array");

		iterateArray(rewriter, op.getLoc(), op.source(),
								 [&](mlir::ValueRange position) {
									 mlir::Value value = rewriter.create<LoadOp>(loc, op.source(), position);
									 value = rewriter.create<CastOp>(value.getLoc(), value, op.destination().getType().cast<ArrayType>().getElementType());
									 rewriter.create<StoreOp>(loc, value, op.destination(), position);
								 });

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct CallOpLowering : public mlir::OpRewritePattern<CallOp>
{
	using mlir::OpRewritePattern<CallOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(CallOp op, mlir::PatternRewriter& rewriter) const override
	{
		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, op.callee(), op->getResultTypes(), op.args());
		return mlir::success();
	}
};

struct ArrayCloneOpLowering : public ModelicaOpConversion<ArrayCloneOp>
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
				mlir::Value index = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(size.index()));
				mlir::Value dim = rewriter.create<DimOp>(loc, op.source(), index);
				dynamicDimensions.push_back(dim);
			}
		}

		mlir::Value result = allocate(rewriter, loc, op.resultType(), dynamicDimensions, op.shouldBeFreed());

		if (options.useRuntimeLibrary)
		{
			llvm::SmallVector<mlir::Value, 2> args;
			args.push_back(rewriter.create<ArrayCastOp>(loc, result, result.getType().cast<ArrayType>().toUnsized()));
			args.push_back(rewriter.create<ArrayCastOp>(loc, op.source(), op.source().getType().cast<ArrayType>().toUnsized()));

			auto callee = getOrDeclareFunction(
					rewriter,
					op->getParentOfType<mlir::ModuleOp>(),
					getMangledFunctionName("clone", llvm::None, args),
					llvm::None,
					args);

			rewriter.create<mlir::CallOp>(loc, callee.getName(), llvm::None, args);
		}
		else
		{
			iterateArray(rewriter, loc, op.source(), [&](mlir::ValueRange indexes) {
				mlir::Value value = rewriter.create<LoadOp>(loc, op.source(), indexes);
				value = rewriter.create<CastOp>(loc, value, op.resultType().cast<ArrayType>().getElementType());
				rewriter.create<StoreOp>(loc, value, result, indexes);
			});
		}

		rewriter.replaceOp(op, result);
		return mlir::success();
	}
};

struct NotOpScalarLowering : public ModelicaOpConversion<NotOp>
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

struct NotOpArrayLowering : public mlir::OpRewritePattern<NotOp>
{
	using mlir::OpRewritePattern<NotOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(NotOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		// Check if the operand is compatible
		if (!op.operand().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Operand is not an array");

		if (auto arrayType = op.operand().getType().cast<ArrayType>(); !arrayType.getElementType().isa<BooleanType>())
			return rewriter.notifyMatchFailure(op, "Operand is not an array of booleans");

		// Allocate the result array
		auto resultType = op.resultType().cast<ArrayType>();
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

struct AndOpScalarLowering : public ModelicaOpConversion<AndOp>
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

struct AndOpArrayLowering : public mlir::OpRewritePattern<AndOp>
{
	AndOpArrayLowering(mlir::MLIRContext* ctx, ModelicaConversionOptions options)
			: mlir::OpRewritePattern<AndOp>(ctx),
				options(std::move(options))
	{
	}

	mlir::LogicalResult matchAndRewrite(AndOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		// Check if the operands are compatible
		if (!op.lhs().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side operand is not an array");

		auto lhsArrayType = op.lhs().getType().cast<ArrayType>();

		if (!lhsArrayType.getElementType().isa<BooleanType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side operand is not an array of booleans");

		if (!op.rhs().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Right-hand side operand is not an array");

		auto rhsArrayType = op.rhs().getType().cast<ArrayType>();

		if (!rhsArrayType.getElementType().isa<BooleanType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side operand is not an array of booleans");

		assert(lhsArrayType.getRank() == rhsArrayType.getRank());

		if (options.assertions)
		{
			// Check if the dimensions are compatible
			auto lhsShape = lhsArrayType.getShape();
			auto rhsShape = rhsArrayType.getShape();

			for (size_t i = 0; i < lhsArrayType.getRank(); ++i)
			{
				if (lhsShape[i] == -1 || rhsShape[i] == -1)
				{
					mlir::Value dimensionIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i));
					mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.lhs(), dimensionIndex);
					mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.rhs(), dimensionIndex);
					mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
					rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
				}
			}
		}

		// Allocate the result array
		auto resultType = op.resultType().cast<ArrayType>();
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

	private:
	ModelicaConversionOptions options;
};

struct OrOpScalarLowering : public ModelicaOpConversion<OrOp>
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

struct OrOpArrayLowering : public mlir::OpRewritePattern<OrOp>
{
	OrOpArrayLowering(mlir::MLIRContext* ctx, ModelicaConversionOptions options)
			: mlir::OpRewritePattern<OrOp>(ctx),
		    options(std::move(options))
	{
	}

	mlir::LogicalResult matchAndRewrite(OrOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		// Check if the operands are compatible
		if (!op.lhs().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side operand is not an array");

		auto lhsArrayType = op.lhs().getType().cast<ArrayType>();

		if (!lhsArrayType.getElementType().isa<BooleanType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side operand is not an array of booleans");

		if (!op.rhs().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Right-hand side operand is not an array");

		auto rhsArrayType = op.rhs().getType().cast<ArrayType>();

		if (!rhsArrayType.getElementType().isa<BooleanType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side operand is not an array of booleans");

		assert(lhsArrayType.getRank() == rhsArrayType.getRank());

		if (options.assertions)
		{
			// Check if the dimensions are compatible
			auto lhsShape = lhsArrayType.getShape();
			auto rhsShape = rhsArrayType.getShape();

			for (size_t i = 0; i < lhsArrayType.getRank(); ++i)
			{
				if (lhsShape[i] == -1 || rhsShape[i] == -1)
				{
					mlir::Value dimensionIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i));
					mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.lhs(), dimensionIndex);
					mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.rhs(), dimensionIndex);
					mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
					rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
				}
			}
		}

		// Allocate the result array
		auto resultType = op.resultType().cast<ArrayType>();
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

	private:
	ModelicaConversionOptions options;
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

struct EqOpLowering : public ComparisonOpLowering<EqOp>
{
	using ComparisonOpLowering<EqOp>::ComparisonOpLowering;

	mlir::Value compareIntegers(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::CmpIOp>(
				loc,
				builder.getIntegerType(1),
				mlir::CmpIPredicate::eq,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}

	mlir::Value compareReals(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::CmpFOp>(
				loc,
				builder.getIntegerType(1),
				mlir::CmpFPredicate::OEQ,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}
};

struct NotEqOpLowering : public ComparisonOpLowering<NotEqOp>
{
	using ComparisonOpLowering<NotEqOp>::ComparisonOpLowering;

	mlir::Value compareIntegers(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::CmpIOp>(
				loc,
				builder.getIntegerType(1),
				mlir::CmpIPredicate::ne,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}

	mlir::Value compareReals(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::CmpFOp>(
				loc,
				builder.getIntegerType(1),
				mlir::CmpFPredicate::ONE,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}
};

struct GtOpLowering : public ComparisonOpLowering<GtOp>
{
	using ComparisonOpLowering<GtOp>::ComparisonOpLowering;

	mlir::Value compareIntegers(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::CmpIOp>(
				loc,
				builder.getIntegerType(1),
				mlir::CmpIPredicate::sgt,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}

	mlir::Value compareReals(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::CmpFOp>(
				loc,
				builder.getIntegerType(1),
				mlir::CmpFPredicate::OGT,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}
};

struct GteOpLowering : public ComparisonOpLowering<GteOp>
{
	using ComparisonOpLowering<GteOp>::ComparisonOpLowering;

	mlir::Value compareIntegers(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::CmpIOp>(
				loc,
				builder.getIntegerType(1),
				mlir::CmpIPredicate::sge,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}

	mlir::Value compareReals(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::CmpFOp>(
				loc,
				builder.getIntegerType(1),
				mlir::CmpFPredicate::OGE,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}
};

struct LtOpLowering : public ComparisonOpLowering<LtOp>
{
	using ComparisonOpLowering<LtOp>::ComparisonOpLowering;

	mlir::Value compareIntegers(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::CmpIOp>(
				loc,
				builder.getIntegerType(1),
				mlir::CmpIPredicate::slt,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}

	mlir::Value compareReals(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::CmpFOp>(
				loc,
				builder.getIntegerType(1),
				mlir::CmpFPredicate::OLT,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}
};

struct LteOpLowering : public ComparisonOpLowering<LteOp>
{
	using ComparisonOpLowering<LteOp>::ComparisonOpLowering;

	mlir::Value compareIntegers(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::CmpIOp>(
				loc,
				builder.getIntegerType(1),
				mlir::CmpIPredicate::sle,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}

	mlir::Value compareReals(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
	{
		mlir::Value result = builder.create<mlir::CmpFOp>(
				loc,
				builder.getIntegerType(1),
				mlir::CmpFPredicate::OLE,
				materializeTargetConversion(builder, lhs),
				materializeTargetConversion(builder, rhs));

		return getTypeConverter()->materializeSourceConversion(
				builder, loc, BooleanType::get(result.getContext()), result);
	}
};

/**
 * Negate a scalar value.
 */
struct NegateOpScalarLowering : public ModelicaOpConversion<NegateOp>
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
struct NegateOpArrayLowering : public mlir::OpRewritePattern<NegateOp>
{
	using mlir::OpRewritePattern<NegateOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(NegateOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		// Check if the operand is compatible
		if (!op.operand().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Value is not an array");

		auto arrayType = op.operand().getType().cast<ArrayType>();

		if (!isNumericType(arrayType.getElementType()))
			return rewriter.notifyMatchFailure(op, "Array has not numeric elements");

		// Allocate the result array
		auto resultType = op.resultType().cast<ArrayType>();
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
template<typename FromOp>
struct AddOpLikeScalarsLowering : public ModelicaOpConversion<FromOp>
{
	using ModelicaOpConversion<FromOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(FromOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		// Check if the operands are compatible
		if (!isNumeric(op.lhs()))
			return rewriter.notifyMatchFailure(op, "Left-hand side value is not a scalar");

		if (!isNumeric(op.rhs()))
			return rewriter.notifyMatchFailure(op, "Right-hand side value is not a scalar");

		// Cast the operands to the most generic type, in order to avoid
		// information loss.
		llvm::SmallVector<mlir::Value, 3> castedOperands;
		mlir::Type type = castToMostGenericType(rewriter, op->getOperands(), castedOperands);
		this->materializeTargetConversion(rewriter, castedOperands);
		typename ModelicaOpConversion<FromOp>::Adaptor transformed(castedOperands);

		// Compute the result
		if (type.isa<mlir::IndexType, BooleanType, IntegerType>())
		{
			mlir::Value result = rewriter.create<mlir::AddIOp>(loc, transformed.lhs(), transformed.rhs());
			result = this->getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		if (type.isa<RealType>())
		{
			mlir::Value result = rewriter.create<mlir::AddFOp>(loc, transformed.lhs(), transformed.rhs());
			result = this->getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		return rewriter.notifyMatchFailure(op, "Unknown type");
	}
};

/**
 * Sum of two numeric arrays.
 */
template<typename FromOp>
struct AddOpLikeArraysLowering : public mlir::OpRewritePattern<FromOp>
{
	AddOpLikeArraysLowering(mlir::MLIRContext* ctx, ModelicaConversionOptions options)
			: mlir::OpRewritePattern<FromOp>(ctx),
				options(std::move(options))
	{
	}

	mlir::LogicalResult matchAndRewrite(FromOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		// Check if the operands are compatible
		if (!op.lhs().getType().template isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side value is not an array");

		if (!op.rhs().getType().template isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Right-hand side value is not an array");

		auto lhsArrayType = op.lhs().getType().template cast<ArrayType>();
		auto rhsArrayType = op.rhs().getType().template cast<ArrayType>();

		for (auto pair : llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape()))
		{
			auto lhsDimension = std::get<0>(pair);
			auto rhsDimension = std::get<1>(pair);

			if (lhsDimension != -1 && rhsDimension != -1 && lhsDimension != rhsDimension)
				return rewriter.notifyMatchFailure(op, "Incompatible array dimensions");
		}

		if (!isNumericType(lhsArrayType.getElementType()))
			return rewriter.notifyMatchFailure(op, "Left-hand side array has not numeric elements");

		if (!isNumericType(rhsArrayType.getElementType()))
			return rewriter.notifyMatchFailure(op, "Right-hand side array has not numeric elements");

		if (options.assertions)
		{
			// Check if the dimensions are compatible
			auto lhsShape = lhsArrayType.getShape();
			auto rhsShape = rhsArrayType.getShape();

			assert(lhsArrayType.getRank() == rhsArrayType.getRank());

			for (size_t i = 0; i < lhsArrayType.getRank(); ++i)
			{
				if (lhsShape[i] == -1 || rhsShape[i] == -1)
				{
					mlir::Value dimensionIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i));
					mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.lhs(), dimensionIndex);
					mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.rhs(), dimensionIndex);
					mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
					rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
				}
			}
		}

		// Allocate the result array
		auto resultType = op.resultType().template cast<ArrayType>();
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

	private:
	ModelicaConversionOptions options;
};

struct AddOpScalarsLowering : public AddOpLikeScalarsLowering<AddOp>
{
	using AddOpLikeScalarsLowering<AddOp>::AddOpLikeScalarsLowering;
};

struct AddOpArraysLowering : public AddOpLikeArraysLowering<AddOp>
{
	using AddOpLikeArraysLowering<AddOp>::AddOpLikeArraysLowering;
};

struct AddElementWiseOpScalarsLowering : public AddOpLikeScalarsLowering<AddElementWiseOp>
{
	using AddOpLikeScalarsLowering<AddElementWiseOp>::AddOpLikeScalarsLowering;
};

struct AddElementWiseOpArraysLowering : public AddOpLikeArraysLowering<AddElementWiseOp>
{
	using AddOpLikeArraysLowering<AddElementWiseOp>::AddOpLikeArraysLowering;
};

struct AddElementWiseOpMixedLowering : public mlir::OpRewritePattern<AddElementWiseOp>
{
	using mlir::OpRewritePattern<AddElementWiseOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(AddElementWiseOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		if (!op.lhs().getType().isa<ArrayType>() && !op.rhs().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "None of the operands is an array");

		if (op.lhs().getType().isa<ArrayType>() && isNumeric(op.rhs()))
			return rewriter.notifyMatchFailure(op, "Right-hand side operand is not a scalar");

		if (op.rhs().getType().isa<ArrayType>() && isNumeric(op.lhs()))
			return rewriter.notifyMatchFailure(op, "Left-hand side operand is not a scalar");

		mlir::Value array = op.lhs().getType().isa<ArrayType>() ? op.lhs() : op.rhs();

		// Allocate the result array
		auto resultType = op.resultType().cast<ArrayType>();
		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
		getArrayDynamicDimensions(rewriter, loc, array, dynamicDimensions);
		mlir::Value result = allocate(rewriter, loc, resultType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		// Apply the operation on each array position
		iterateArray(rewriter, loc, array,
								 [&](mlir::ValueRange position) {
									 mlir::Value lhs = op.lhs().getType().isa<ArrayType>() ?
																		 rewriter.create<LoadOp>(loc, op.lhs(), position) : op.lhs();

									 mlir::Value rhs = op.lhs().getType().isa<ArrayType>() ?
																		 op.rhs() : rewriter.create<LoadOp>(loc, op.rhs(), position);

									 mlir::Value value = rewriter.create<AddOp>(loc, resultType.getElementType(), lhs, rhs);
									 rewriter.create<StoreOp>(loc, value, result, position);
								 });

		return mlir::success();
	}
};

/**
 * Subtraction of two numeric scalars.
 */
template<typename FromOp>
struct SubOpLikeScalarsLowering : public ModelicaOpConversion<FromOp>
{
	using ModelicaOpConversion<FromOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(FromOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		// Check if the operands are compatible
		if (!isNumeric(op.lhs()))
			return rewriter.notifyMatchFailure(op, "Left-hand side value is not a scalar");

		if (!isNumeric(op.rhs()))
			return rewriter.notifyMatchFailure(op, "Right-hand side value is not a scalar");

		// Cast the operands to the most generic type, in order to avoid
		// information loss.
		llvm::SmallVector<mlir::Value, 3> castedOperands;
		mlir::Type type = castToMostGenericType(rewriter, op->getOperands(), castedOperands);
		this->materializeTargetConversion(rewriter, castedOperands);
		typename ModelicaOpConversion<FromOp>::Adaptor transformed(castedOperands);

		// Compute the result
		if (type.isa<mlir::IndexType, BooleanType, IntegerType>())
		{
			mlir::Value result = rewriter.create<mlir::SubIOp>(loc, transformed.lhs(), transformed.rhs());
			result = this->getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		if (type.isa<RealType>())
		{
			mlir::Value result = rewriter.create<mlir::SubFOp>(loc, transformed.lhs(), transformed.rhs());
			result = this->getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		return rewriter.notifyMatchFailure(op, "Unknown type");
	}
};

/**
 * Subtraction of two numeric arrays.
 */
template<typename FromOp>
struct SubOpLikeArraysLowering : public mlir::OpRewritePattern<FromOp>
{
	SubOpLikeArraysLowering(mlir::MLIRContext* ctx, ModelicaConversionOptions options)
			: mlir::OpRewritePattern<FromOp>(ctx),
				options(std::move(options))
	{
	}

	mlir::LogicalResult matchAndRewrite(FromOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		// Check if the operands are compatible
		if (!op.lhs().getType().template isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side value is not an array");

		if (!op.rhs().getType().template isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Right-hand side value is not an array");

		auto lhsArrayType = op.lhs().getType().template cast<ArrayType>();
		auto rhsArrayType = op.rhs().getType().template cast<ArrayType>();

		for (auto pair : llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape()))
		{
			auto lhsDimension = std::get<0>(pair);
			auto rhsDimension = std::get<1>(pair);

			if (lhsDimension != -1 && rhsDimension != -1 && lhsDimension != rhsDimension)
				return rewriter.notifyMatchFailure(op, "Incompatible array dimensions");
		}

		if (!isNumericType(lhsArrayType.getElementType()))
			return rewriter.notifyMatchFailure(op, "Left-hand side array has not numeric elements");

		if (!isNumericType(rhsArrayType.getElementType()))
			return rewriter.notifyMatchFailure(op, "Right-hand side array has not numeric elements");

		if (options.assertions)
		{
			// Check if the dimensions are compatible
			auto lhsShape = lhsArrayType.getShape();
			auto rhsShape = rhsArrayType.getShape();

			assert(lhsArrayType.getRank() == rhsArrayType.getRank());

			for (size_t i = 0; i < lhsArrayType.getRank(); ++i)
			{
				if (lhsShape[i] == -1 || rhsShape[i] == -1)
				{
					mlir::Value dimensionIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i));
					mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.lhs(), dimensionIndex);
					mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.rhs(), dimensionIndex);
					mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
					rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
				}
			}
		}

		// Allocate the result array
		auto resultType = op.resultType().template cast<ArrayType>();
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

	private:
	ModelicaConversionOptions options;
};

struct SubOpScalarsLowering : public SubOpLikeScalarsLowering<SubOp>
{
	using SubOpLikeScalarsLowering<SubOp>::SubOpLikeScalarsLowering;
};

struct SubOpArraysLowering : public SubOpLikeArraysLowering<SubOp>
{
	using SubOpLikeArraysLowering<SubOp>::SubOpLikeArraysLowering;
};

struct SubElementWiseOpScalarsLowering : public SubOpLikeScalarsLowering<SubElementWiseOp>
{
	using SubOpLikeScalarsLowering<SubElementWiseOp>::SubOpLikeScalarsLowering;
};

struct SubElementWiseOpArraysLowering : public SubOpLikeArraysLowering<SubElementWiseOp>
{
	using SubOpLikeArraysLowering<SubElementWiseOp>::SubOpLikeArraysLowering;
};

struct SubElementWiseOpMixedLowering : public mlir::OpRewritePattern<SubElementWiseOp>
{
	using mlir::OpRewritePattern<SubElementWiseOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(SubElementWiseOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		if (!op.lhs().getType().isa<ArrayType>() && !op.rhs().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "None of the operands is an array");

		if (op.lhs().getType().isa<ArrayType>() && isNumeric(op.rhs()))
			return rewriter.notifyMatchFailure(op, "Right-hand side operand is not a scalar");

		if (op.rhs().getType().isa<ArrayType>() && isNumeric(op.lhs()))
			return rewriter.notifyMatchFailure(op, "Left-hand side operand is not a scalar");

		mlir::Value array = op.lhs().getType().isa<ArrayType>() ? op.lhs() : op.rhs();

		// Allocate the result array
		auto resultType = op.resultType().cast<ArrayType>();
		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
		getArrayDynamicDimensions(rewriter, loc, array, dynamicDimensions);
		mlir::Value result = allocate(rewriter, loc, resultType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		// Apply the operation on each array position
		iterateArray(rewriter, loc, array,
								 [&](mlir::ValueRange position) {
									 mlir::Value lhs = op.lhs().getType().isa<ArrayType>() ?
									     rewriter.create<LoadOp>(loc, op.lhs(), position) : op.lhs();

									 mlir::Value rhs = op.lhs().getType().isa<ArrayType>() ?
																		 op.rhs() : rewriter.create<LoadOp>(loc, op.rhs(), position);

									 mlir::Value value = rewriter.create<SubOp>(loc, resultType.getElementType(), lhs, rhs);
									 rewriter.create<StoreOp>(loc, value, result, position);
								 });

		return mlir::success();
	}
};

/**
 * Product between two scalar values.
 */
template<typename FromOp>
struct MulOpLikeScalarsLowering : public ModelicaOpConversion<FromOp>
{
	using ModelicaOpConversion<FromOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(FromOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
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
		this->materializeTargetConversion(rewriter, castedOperands);
		typename ModelicaOpConversion<FromOp>::Adaptor transformed(castedOperands);

		// Compute the result
		if (type.isa<mlir::IndexType, BooleanType, IntegerType>())
		{
			mlir::Value result = rewriter.create<mlir::MulIOp>(loc, transformed.lhs(), transformed.rhs());
			result = this->getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		if (type.isa<RealType>())
		{
			mlir::Value result = rewriter.create<mlir::MulFOp>(loc, transformed.lhs(), transformed.rhs());
			result = this->getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		return rewriter.notifyMatchFailure(op, "Unknown type");
	}
};

/**
 * Product between a scalar and an array.
 */
template<typename FromOp>
struct MulOpLikeScalarProductLowering : public mlir::OpRewritePattern<FromOp>
{
	using mlir::OpRewritePattern<FromOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(FromOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		// Check if the operands are compatible
		if (!isNumeric(op.lhs()) && !isNumeric(op.rhs()))
			return rewriter.notifyMatchFailure(op, "Scalar-array product: none of the operands is a scalar");

		if (isNumeric(op.lhs()))
		{
			if (!op.rhs().getType().template isa<ArrayType>())
				return rewriter.notifyMatchFailure(op, "Scalar-array product: right-hand size value is not an array");

			if (!isNumericType(op.rhs().getType().template cast<ArrayType>().getElementType()))
				return rewriter.notifyMatchFailure(op, "Scalar-array product: right-hand side array has not numeric elements");
		}

		if (isNumeric(op.rhs()))
		{
			if (!op.lhs().getType().template isa<ArrayType>())
				return rewriter.notifyMatchFailure(op, "Scalar-array product: right-hand size value is not an array");

			if (!isNumericType(op.lhs().getType().template cast<ArrayType>().getElementType()))
				return rewriter.notifyMatchFailure(op, "Scalar-array product: left-hand side array has not numeric elements");
		}

		mlir::Value scalar = isNumeric(op.lhs()) ? op.lhs() : op.rhs();
		mlir::Value array = isNumeric(op.rhs()) ? op.lhs() : op.rhs();

		// Allocate the result array
		auto resultType = op.resultType().template cast<ArrayType>();
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

struct MulOpScalarsLowering : public MulOpLikeScalarsLowering<MulOp>
{
	using MulOpLikeScalarsLowering<MulOp>::MulOpLikeScalarsLowering;
};

struct MulOpScalarProductLowering : public MulOpLikeScalarProductLowering<MulOp>
{
	using MulOpLikeScalarProductLowering<MulOp>::MulOpLikeScalarProductLowering;
};

/**
 * Cross product of two 1-D arrays. Result is a scalar.
 *
 * [ x1, x2, x3 ] * [ y1, y2, y3 ] = x1 * y1 + x2 * y2 + x3 * y3
 */
struct MulOpCrossProductLowering : public ModelicaOpConversion<MulOp>
{
	using ModelicaOpConversion<MulOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(MulOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		// Check if the operands are compatible
		if (!op.lhs().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Cross product: left-hand side value is not an array");

		auto lhsArrayType = op.lhs().getType().cast<ArrayType>();

		if (lhsArrayType.getRank() != 1)
			return rewriter.notifyMatchFailure(op, "Cross product: left-hand side arrays is not 1D");

		if (!op.rhs().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Cross product: right-hand side value is not an array");

		auto rhsArrayType = op.rhs().getType().cast<ArrayType>();

		if (rhsArrayType.getRank() != 1)
			return rewriter.notifyMatchFailure(op, "Cross product: right-hand side arrays is not 1D");

		if (lhsArrayType.getShape()[0] != -1 && rhsArrayType.getShape()[0] != -1)
			if (lhsArrayType.getShape()[0] != rhsArrayType.getShape()[0])
				return rewriter.notifyMatchFailure(op, "Cross product: the two arrays have different shape");

		assert(lhsArrayType.getRank() == 1);
		assert(rhsArrayType.getRank() == 1);

		if (options.assertions)
		{
			// Check if the dimensions are compatible
			auto lhsShape = lhsArrayType.getShape();
			auto rhsShape = rhsArrayType.getShape();

			if (lhsShape[0] == -1 || rhsShape[0] == -1)
			{
				mlir::Value dimensionIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
				mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.lhs(), dimensionIndex);
				mlir::Value rhsDimensionSize =  rewriter.create<DimOp>(loc, op.rhs(), dimensionIndex);
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
struct MulOpVectorMatrixLowering : public ModelicaOpConversion<MulOp>
{
	using ModelicaOpConversion<MulOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(MulOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		// Check if the operands are compatible
		if (!op.lhs().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Vector-matrix product: left-hand side value is not an array");

		auto lhsArrayType = op.lhs().getType().cast<ArrayType>();

		if (lhsArrayType.getRank() != 1)
			return rewriter.notifyMatchFailure(op, "Vector-matrix product: left-hand size array is not 1-D");

		if (!op.rhs().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Vector-matrix product: right-hand side value is not an array");

		auto rhsArrayType = op.rhs().getType().cast<ArrayType>();

		if (rhsArrayType.getRank() != 2)
			return rewriter.notifyMatchFailure(op, "Vector-matrix product: right-hand side matrix is not 2-D");

		if (lhsArrayType.getShape()[0] != -1 && rhsArrayType.getShape()[0] != -1)
			if (lhsArrayType.getShape()[0] != rhsArrayType.getShape()[0])
				return rewriter.notifyMatchFailure(op, "Vector-matrix product: incompatible shapes");

		assert(lhsArrayType.getRank() == 1);
		assert(rhsArrayType.getRank() == 2);

		if (options.assertions)
		{
			// Check if the dimensions are compatible
			auto lhsShape = lhsArrayType.getShape();
			auto rhsShape = rhsArrayType.getShape();

			if (lhsShape[0] == -1 || rhsShape[0] == -1)
			{
				mlir::Value zero = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
				mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.lhs(), zero);
				mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.rhs(), zero);
				mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
				rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
			}
		}

		// Allocate the result array
		Adaptor transformed(operands);
		auto resultType = op.resultType().cast<ArrayType>();
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
struct MulOpMatrixVectorLowering : public ModelicaOpConversion<MulOp>
{
	using ModelicaOpConversion<MulOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(MulOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		// Check if the operands are compatible
		if (!op.lhs().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Matrix-vector product: left-hand side value is not an array");

		auto lhsArrayType = op.lhs().getType().cast<ArrayType>();

		if (lhsArrayType.getRank() != 2)
			return rewriter.notifyMatchFailure(op, "Matrix-vector product: left-hand size array is not 2-D");

		if (!op.rhs().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Matrix-vector product: right-hand side value is not an array");

		auto rhsArrayType = op.rhs().getType().cast<ArrayType>();

		if (rhsArrayType.getRank() != 1)
			return rewriter.notifyMatchFailure(op, "Matrix-vector product: right-hand side matrix is not 1-D");

		if (lhsArrayType.getShape()[1] != -1 && rhsArrayType.getShape()[0] != -1)
			if (lhsArrayType.getShape()[1] != rhsArrayType.getShape()[0])
				return rewriter.notifyMatchFailure(op, "Matrix-vector product: incompatible shapes");

		assert(lhsArrayType.getRank() == 2);
		assert(rhsArrayType.getRank() == 1);

		if (options.assertions)
		{
			// Check if the dimensions are compatible
			auto lhsShape = lhsArrayType.getShape();
			auto rhsShape = rhsArrayType.getShape();

			if (lhsShape[1] == -1 || rhsShape[0] == -1)
			{
				mlir::Value zero = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
				mlir::Value one = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));
				mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.lhs(), one);
				mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.rhs(), zero);
				mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
				rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
			}
		}

		// Allocate the result array
		Adaptor transformed(operands);
		auto resultType = op.resultType().cast<ArrayType>();
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
struct MulOpMatrixLowering : public ModelicaOpConversion<MulOp>
{
	using ModelicaOpConversion<MulOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(MulOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		// Check if the operands are compatible
		if (!op.lhs().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Matrix product: left-hand side value is not an array");

		auto lhsArrayType = op.lhs().getType().cast<ArrayType>();

		if (lhsArrayType.getRank() != 2)
			return rewriter.notifyMatchFailure(op, "Matrix product: left-hand size array is not 2-D");

		if (!op.rhs().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Matrix product: right-hand side value is not an array");

		auto rhsArrayType = op.rhs().getType().cast<ArrayType>();

		if (rhsArrayType.getRank() != 2)
			return rewriter.notifyMatchFailure(op, "Matrix product: right-hand side matrix is not 2-D");

		if (lhsArrayType.getShape()[1] != -1 && rhsArrayType.getShape()[0] != -1)
			if (lhsArrayType.getShape()[1] != rhsArrayType.getShape()[0])
				return rewriter.notifyMatchFailure(op, "Matrix-vector product: incompatible shapes");

		assert(lhsArrayType.getRank() == 2);
		assert(rhsArrayType.getRank() == 2);

		if (options.assertions)
		{
			// Check if the dimensions are compatible
			auto lhsShape = lhsArrayType.getShape();
			auto rhsShape = rhsArrayType.getShape();

			if (lhsShape[1] == -1 || rhsShape[0] == -1)
			{
				mlir::Value zero = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
				mlir::Value one = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));
				mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.lhs(), one);
				mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.rhs(), zero);
				mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
				rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
			}
		}

		// Allocate the result array
		Adaptor transformed(operands);
		auto resultType = op.resultType().cast<ArrayType>();
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

struct MulElementWiseOpScalarsLowering : public MulOpLikeScalarsLowering<MulElementWiseOp>
{
	using MulOpLikeScalarsLowering<MulElementWiseOp>::MulOpLikeScalarsLowering;
};

struct MulElementWiseOpScalarProductLowering : public MulOpLikeScalarProductLowering<MulElementWiseOp>
{
	using MulOpLikeScalarProductLowering<MulElementWiseOp>::MulOpLikeScalarProductLowering;
};

struct MulElementWiseOpArraysLowering : public mlir::OpRewritePattern<MulElementWiseOp>
{
	MulElementWiseOpArraysLowering(mlir::MLIRContext* ctx, ModelicaConversionOptions options)
			: mlir::OpRewritePattern<MulElementWiseOp>(ctx),
				options(std::move(options))
	{
	}

	mlir::LogicalResult matchAndRewrite(MulElementWiseOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		// Check if the operands are compatible
		if (!op.lhs().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side value is not an array");

		if (!op.rhs().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Right-hand side value is not an array");

		auto lhsArrayType = op.lhs().getType().cast<ArrayType>();
		auto rhsArrayType = op.rhs().getType().cast<ArrayType>();

		for (auto pair : llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape()))
		{
			auto lhsDimension = std::get<0>(pair);
			auto rhsDimension = std::get<1>(pair);

			if (lhsDimension != -1 && rhsDimension != -1 && lhsDimension != rhsDimension)
				return rewriter.notifyMatchFailure(op, "Incompatible array dimensions");
		}

		if (!isNumericType(lhsArrayType.getElementType()))
			return rewriter.notifyMatchFailure(op, "Left-hand side array has not numeric elements");

		if (!isNumericType(rhsArrayType.getElementType()))
			return rewriter.notifyMatchFailure(op, "Right-hand side array has not numeric elements");

		if (options.assertions)
		{
			// Check if the dimensions are compatible
			auto lhsShape = lhsArrayType.getShape();
			auto rhsShape = rhsArrayType.getShape();

			assert(lhsArrayType.getRank() == rhsArrayType.getRank());

			for (size_t i = 0; i < lhsArrayType.getRank(); ++i)
			{
				if (lhsShape[i] == -1 || rhsShape[i] == -1)
				{
					mlir::Value dimensionIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i));
					mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.lhs(), dimensionIndex);
					mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.rhs(), dimensionIndex);
					mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
					rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
				}
			}
		}

		// Allocate the result array
		auto resultType = op.resultType().cast<ArrayType>();
		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
		getArrayDynamicDimensions(rewriter, loc, op.lhs(), dynamicDimensions);
		mlir::Value result = allocate(rewriter, loc, resultType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		// Apply the operation on each array position
		iterateArray(rewriter, loc, op.lhs(),
								 [&](mlir::ValueRange position) {
									 mlir::Value lhs = rewriter.create<LoadOp>(loc, op.lhs(), position);
									 mlir::Value rhs = rewriter.create<LoadOp>(loc, op.rhs(), position);
									 mlir::Value value = rewriter.create<MulOp>(loc, resultType.getElementType(), lhs, rhs);
									 rewriter.create<StoreOp>(loc, value, result, position);
								 });

		return mlir::success();
	}

	private:
	ModelicaConversionOptions options;
};

/**
 * Division between two scalar values.
 */
template<typename FromOp>
struct DivOpLikeScalarsLowering : public ModelicaOpConversion<FromOp>
{
	using ModelicaOpConversion<FromOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(FromOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		// Check if the operands are compatible
		if (!isNumeric(op.lhs()))
			return rewriter.notifyMatchFailure(op, "Scalar-scalar division: left-hand side value is not a scalar");

		if (!isNumeric(op.rhs()))
			return rewriter.notifyMatchFailure(op, "Scalar-scalar division: right-hand side value is not a scalar");

		// Cast the operands to the most generic type, in order to avoid
		// information loss.
		llvm::SmallVector<mlir::Value, 3> castedOperands;
		mlir::Type type = castToMostGenericType(rewriter, op->getOperands(), castedOperands);
		this->materializeTargetConversion(rewriter, castedOperands);
		typename ModelicaOpConversion<FromOp>::Adaptor transformed(castedOperands);

		// Compute the result
		if (type.isa<mlir::IndexType, BooleanType, IntegerType>())
		{
			mlir::Value result = rewriter.create<mlir::SignedDivIOp>(loc, transformed.lhs(), transformed.rhs());
			result = this->getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		if (type.isa<RealType>())
		{
			mlir::Value result = rewriter.create<mlir::DivFOp>(loc, transformed.lhs(), transformed.rhs());
			result = this->getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
			rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
			return mlir::success();
		}

		return rewriter.notifyMatchFailure(op, "Unknown type");
	}
};

struct DivOpScalarsLowering : public DivOpLikeScalarsLowering<DivOp>
{
	using DivOpLikeScalarsLowering<DivOp>::DivOpLikeScalarsLowering;
};

/**
 * Division between an array and a scalar value.
 */
struct DivOpArrayScalarLowering : public mlir::OpRewritePattern<DivOp>
{
	using mlir::OpRewritePattern<DivOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(DivOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();

		// Check if the operands are compatible
		if (!op.lhs().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Array-scalar division: left-hand size value is not an array");

		if (!isNumericType(op.lhs().getType().cast<ArrayType>().getElementType()))
			return rewriter.notifyMatchFailure(op, "Array-scalar division: right-hand side array has not numeric elements");

		if (!isNumeric(op.rhs()))
			return rewriter.notifyMatchFailure(op, "Array-scalar division: right-hand size value is not a scalar");

		// Allocate the result array
		auto resultType = op.resultType().cast<ArrayType>();
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

struct DivElementWiseOpScalarsLowering : public DivOpLikeScalarsLowering<DivElementWiseOp>
{
	using DivOpLikeScalarsLowering<DivElementWiseOp>::DivOpLikeScalarsLowering;
};

/**
 * Division between an array and a scalar, or the opposite.
 */
struct DivElementWiseOpMixedLowering : public mlir::OpRewritePattern<DivElementWiseOp>
{
	using mlir::OpRewritePattern<DivElementWiseOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(DivElementWiseOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		if (!op.lhs().getType().isa<ArrayType>() && !op.rhs().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "None of the operands is an array");

		if (op.lhs().getType().isa<ArrayType>() && isNumeric(op.rhs()))
			return rewriter.notifyMatchFailure(op, "Right-hand side operand is not a scalar");

		if (op.rhs().getType().isa<ArrayType>() && isNumeric(op.lhs()))
			return rewriter.notifyMatchFailure(op, "Left-hand side operand is not a scalar");

		mlir::Value array = op.lhs().getType().isa<ArrayType>() ? op.lhs() : op.rhs();

		// Allocate the result array
		auto resultType = op.resultType().cast<ArrayType>();
		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
		getArrayDynamicDimensions(rewriter, loc, array, dynamicDimensions);
		mlir::Value result = allocate(rewriter, loc, resultType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		// Apply the operation on each array position
		iterateArray(rewriter, loc, array,
								 [&](mlir::ValueRange position) {
									 mlir::Value lhs = op.lhs().getType().isa<ArrayType>() ?
																		 rewriter.create<LoadOp>(loc, op.lhs(), position) : op.lhs();

									 mlir::Value rhs = op.lhs().getType().isa<ArrayType>() ?
																		 op.rhs() : rewriter.create<LoadOp>(loc, op.rhs(), position);

									 mlir::Value value = rewriter.create<DivOp>(loc, resultType.getElementType(), lhs, rhs);
									 rewriter.create<StoreOp>(loc, value, result, position);
								 });

		return mlir::success();
	}
};

/**
 * Element-wise division of two numeric arrays.
 */
struct DivElementWiseOpArraysLowering : public mlir::OpRewritePattern<DivElementWiseOp>
{
	DivElementWiseOpArraysLowering(mlir::MLIRContext* ctx, ModelicaConversionOptions options)
			: mlir::OpRewritePattern<DivElementWiseOp>(ctx),
				options(std::move(options))
	{
	}

	mlir::LogicalResult matchAndRewrite(DivElementWiseOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		// Check if the operands are compatible
		if (!op.lhs().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Left-hand side value is not an array");

		if (!op.rhs().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Right-hand side value is not an array");

		auto lhsArrayType = op.lhs().getType().cast<ArrayType>();
		auto rhsArrayType = op.rhs().getType().cast<ArrayType>();

		for (auto pair : llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape()))
		{
			auto lhsDimension = std::get<0>(pair);
			auto rhsDimension = std::get<1>(pair);

			if (lhsDimension != -1 && rhsDimension != -1 && lhsDimension != rhsDimension)
				return rewriter.notifyMatchFailure(op, "Incompatible array dimensions");
		}

		if (!isNumericType(lhsArrayType.getElementType()))
			return rewriter.notifyMatchFailure(op, "Left-hand side array has not numeric elements");

		if (!isNumericType(rhsArrayType.getElementType()))
			return rewriter.notifyMatchFailure(op, "Right-hand side array has not numeric elements");

		if (options.assertions)
		{
			// Check if the dimensions are compatible
			auto lhsShape = lhsArrayType.getShape();
			auto rhsShape = rhsArrayType.getShape();

			assert(lhsArrayType.getRank() == rhsArrayType.getRank());

			for (size_t i = 0; i < lhsArrayType.getRank(); ++i)
			{
				if (lhsShape[i] == -1 || rhsShape[i] == -1)
				{
					mlir::Value dimensionIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i));
					mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.lhs(), dimensionIndex);
					mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.rhs(), dimensionIndex);
					mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
					rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
				}
			}
		}

		// Allocate the result array
		auto resultType = op.resultType().cast<ArrayType>();
		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
		getArrayDynamicDimensions(rewriter, loc, op.lhs(), dynamicDimensions);
		mlir::Value result = allocate(rewriter, loc, resultType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		// Apply the operation on each array position
		iterateArray(rewriter, loc, op.lhs(),
								 [&](mlir::ValueRange position) {
									 mlir::Value lhs = rewriter.create<LoadOp>(loc, op.lhs(), position);
									 mlir::Value rhs = rewriter.create<LoadOp>(loc, op.rhs(), position);
									 mlir::Value value = rewriter.create<DivOp>(loc, resultType.getElementType(), lhs, rhs);
									 rewriter.create<StoreOp>(loc, value, result, position);
								 });

		return mlir::success();
	}

	private:
	ModelicaConversionOptions options;
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
		mlir::Value base = rewriter.create<CastOp>(loc, op.base(), RealType::get(op.getContext()));
		base = materializeTargetConversion(rewriter, base);

		mlir::Value exponent = rewriter.create<CastOp>(loc, op.exponent(), RealType::get(op.getContext()));
		exponent = materializeTargetConversion(rewriter, exponent);

		mlir::Value result = rewriter.create<mlir::math::PowFOp>(loc, base, exponent);
		result = getTypeConverter()->materializeSourceConversion(rewriter, loc, RealType::get(op.getContext()), result);
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
		if (!op.base().getType().isa<ArrayType>())
			return rewriter.notifyMatchFailure(op, "Base is not an array");

		auto baseArrayType = op.base().getType().cast<ArrayType>();

		if (baseArrayType.getRank() != 2)
			return rewriter.notifyMatchFailure(op, "Base array is not 2-D");

		if (baseArrayType.getShape()[0] != -1 && baseArrayType.getShape()[1] != -1)
			if (baseArrayType.getShape()[0] != baseArrayType.getShape()[1])
				return rewriter.notifyMatchFailure(op, "Base is not a square matrix");

		if (!op.exponent().getType().isa<IntegerType>())
			return rewriter.notifyMatchFailure(op, "Exponent is not an integer");

		assert(baseArrayType.getRank() == 2);

		if (options.assertions)
		{
			// Check if the matrix is a square one
			auto shape = baseArrayType.getShape();

			if (shape[0] == -1 || shape[1] == -1)
			{
				mlir::Value zero = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
				mlir::Value one = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));
				mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.base(), one);
				mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.base(), zero);
				mlir::Value condition = rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
				rewriter.create<mlir::AssertOp>(loc, condition, rewriter.getStringAttr("Base matrix is not squared"));
			}
		}

		// Allocate the result array
		auto resultArrayType = op.resultType().cast<ArrayType>();
		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
		getArrayDynamicDimensions(rewriter, loc, op.base(), dynamicDimensions);
		mlir::Value result = allocate(rewriter, loc, resultArrayType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		// Compute the result
		mlir::Value exponent = rewriter.create<CastOp>(loc, op.exponent(), mlir::IndexType::get(op->getContext()));
		mlir::Value lowerBound = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));
		mlir::Value step = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));

		// The intermediate results must be allocated on the heap, in order
		// to avoid a potentially big allocation on the stack (due to the
		// iteration).
		auto intermediateResultType = op.base().getType().cast<ArrayType>().toAllocationScope(BufferAllocationScope::heap);
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

struct AbsOpLowering : public ModelicaOpConversion<AbsOp>
{
	using ModelicaOpConversion<AbsOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(AbsOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto realType = RealType::get(op.getContext());

		mlir::Value arg = rewriter.create<CastOp>(loc, op.operand(), realType);

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("abs", realType, arg),
				realType,
				arg.getType());

		mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
		return mlir::success();
	}
};

struct AcosOpLowering : public ModelicaOpConversion<AcosOp>
{
	using ModelicaOpConversion<AcosOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(AcosOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto realType = RealType::get(op.getContext());

		mlir::Value arg = rewriter.create<CastOp>(loc, op.operand(), realType);

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("acos", realType, arg),
				realType,
				arg.getType());

		mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
		return mlir::success();
	}
};

struct AsinOpLowering : public ModelicaOpConversion<AsinOp>
{
	using ModelicaOpConversion<AsinOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(AsinOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto realType = RealType::get(op.getContext());

		mlir::Value arg = rewriter.create<CastOp>(loc, op.operand(), realType);

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("asin", realType, arg),
				realType,
				arg.getType());

		mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
		return mlir::success();
	}
};

struct AtanOpLowering : public ModelicaOpConversion<AtanOp>
{
	using ModelicaOpConversion<AtanOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(AtanOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto realType = RealType::get(op.getContext());

		mlir::Value arg = rewriter.create<CastOp>(loc, op.operand(), realType);

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("atan", realType, arg),
				realType,
				arg.getType());

		mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
		return mlir::success();
	}
};

struct Atan2OpLowering : public ModelicaOpConversion<Atan2Op>
{
	using ModelicaOpConversion<Atan2Op>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(Atan2Op op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto realType = RealType::get(op.getContext());

		llvm::SmallVector<mlir::Value, 3> args;
		args.push_back(rewriter.create<CastOp>(loc, op.y(), realType));
		args.push_back(rewriter.create<CastOp>(loc, op.x(), realType));

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("atan2", realType, args),
				realType,
				args);

		mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, args).getResult(0);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
		return mlir::success();
	}
};

struct CosOpLowering : public ModelicaOpConversion<CosOp>
{
	using ModelicaOpConversion<CosOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(CosOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto realType = RealType::get(op.getContext());

		mlir::Value arg = rewriter.create<CastOp>(loc, op.operand(), realType);

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("cos", realType, arg),
				realType,
				arg.getType());

		mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
		return mlir::success();
	}
};

struct CoshhOpLowering : public ModelicaOpConversion<CoshOp>
{
	using ModelicaOpConversion<CoshOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(CoshOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto realType = RealType::get(op.getContext());

		mlir::Value arg = rewriter.create<CastOp>(loc, op.operand(), realType);

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("cosh", realType, arg),
				realType,
				arg.getType());

		mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
		return mlir::success();
	}
};

struct DiagonalOpLowering : public ModelicaOpConversion<DiagonalOp>
{
	using ModelicaOpConversion<DiagonalOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(DiagonalOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto arrayType = op.resultType().cast<ArrayType>();

		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
		mlir::Value castedSize = nullptr;

		for (const auto& size : arrayType.getShape())
		{
			if (size == -1)
			{
				if (castedSize == nullptr)
				{
					assert(op.values().getType().cast<ArrayType>().getRank() == 1);
					mlir::Value zeroValue = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
					castedSize = rewriter.create<DimOp>(loc, op.values(), zeroValue);
				}

				dynamicDimensions.push_back(castedSize);
			}
		}

		mlir::Value result = allocate(rewriter, loc, arrayType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		llvm::SmallVector<mlir::Value, 3> args;

		args.push_back(rewriter.create<ArrayCastOp>(
				loc, result,
				result.getType().cast<ArrayType>().toUnsized()));

		args.push_back(rewriter.create<ArrayCastOp>(
				loc, op.values(),
				op.values().getType().cast<ArrayType>().toUnsized()));

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("diagonal", llvm::None, args),
				llvm::None,
				args);

		rewriter.create<mlir::CallOp>(loc, callee.getName(), llvm::None, args);
		return mlir::success();
	}
};

struct ExpOpLowering : public ModelicaOpConversion<ExpOp>
{
	using ModelicaOpConversion<ExpOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(ExpOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto realType = RealType::get(op.getContext());

		mlir::Value arg = rewriter.create<CastOp>(loc, op.exponent(), realType);

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("exp", realType, arg),
				realType,
				arg.getType());

		mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
		return mlir::success();
	}
};

struct FillOpLowering : public ModelicaOpConversion<FillOp>
{
	using ModelicaOpConversion<FillOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(FillOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto arrayType = op.memory().getType().cast<ArrayType>();

		if (options.useRuntimeLibrary)
		{
			llvm::SmallVector<mlir::Value, 3> args;
			args.push_back(rewriter.create<ArrayCastOp>(loc, op.memory(), arrayType.toUnsized()));
			args.push_back(rewriter.create<CastOp>(loc, op.value(), arrayType.getElementType()));

			auto callee = getOrDeclareFunction(
					rewriter,
					op->getParentOfType<mlir::ModuleOp>(),
					getMangledFunctionName("fill", llvm::None, args),
					llvm::None,
					mlir::ValueRange(args).getTypes());

			rewriter.create<mlir::CallOp>(loc, callee.getName(), llvm::None, args);
		}
		else
		{
			mlir::Value value = rewriter.create<CastOp>(loc, op.value(), arrayType.getElementType());

			iterateArray(rewriter, loc, op.memory(),
									 [&](mlir::ValueRange position) {
										 rewriter.create<StoreOp>(loc, value, op.memory(), position);
									 });
		}

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct IdentityOpLowering : public ModelicaOpConversion<IdentityOp>
{
	using ModelicaOpConversion<IdentityOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(IdentityOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto arrayType = op.resultType().cast<ArrayType>();

		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
		mlir::Value castedSize = nullptr;

		for (const auto& size : arrayType.getShape())
		{
			if (size == -1)
			{
				if (castedSize == nullptr)
					castedSize = rewriter.create<CastOp>(loc, op.size(), rewriter.getIndexType());

				dynamicDimensions.push_back(castedSize);
			}
		}

		mlir::Value result = allocate(rewriter, loc, arrayType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		mlir::Value arg = rewriter.create<ArrayCastOp>(
				loc, result,
				result.getType().cast<ArrayType>().toUnsized());

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("identity", llvm::None, arg),
				llvm::None,
				arg.getType());

		rewriter.create<mlir::CallOp>(loc, callee.getName(), llvm::None, arg);
		return mlir::success();
	}
};

struct LinspaceOpLowering : public ModelicaOpConversion<LinspaceOp>
{
	using ModelicaOpConversion<LinspaceOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(LinspaceOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto arrayType = op.resultType().cast<ArrayType>();

		assert(arrayType.getRank() == 1);
		mlir::Value size = rewriter.create<CastOp>(loc, op.steps(), rewriter.getIndexType());

		mlir::Value result = allocate(rewriter, loc, arrayType, size);
		rewriter.replaceOp(op, result);

		llvm::SmallVector<mlir::Value, 3> args;

		args.push_back(rewriter.create<ArrayCastOp>(
				loc, result,
				result.getType().cast<ArrayType>().toUnsized()));

		args.push_back(rewriter.create<CastOp>(
				loc, op.start(),
				RealType::get(op->getContext())));

		args.push_back(rewriter.create<CastOp>(
				loc, op.end(),
				RealType::get(op->getContext())));

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("linspace", llvm::None, args),
				llvm::None,
				args);

		rewriter.create<mlir::CallOp>(loc, callee.getName(), llvm::None, args);
		return mlir::success();
	}
};

struct LogOpLowering : public ModelicaOpConversion<LogOp>
{
	using ModelicaOpConversion<LogOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(LogOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto realType = RealType::get(op.getContext());

		mlir::Value arg = rewriter.create<CastOp>(loc, op.operand(), realType);

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("log", realType, arg),
				realType,
				arg.getType());

		mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
		return mlir::success();
	}
};

struct Log10OpLowering : public ModelicaOpConversion<Log10Op>
{
	using ModelicaOpConversion<Log10Op>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(Log10Op op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto realType = RealType::get(op.getContext());

		mlir::Value arg = rewriter.create<CastOp>(loc, op.operand(), realType);

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("log10", realType, arg),
				realType,
				arg.getType());

		mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
		return mlir::success();
	}
};

struct NDimsOpLowering : public mlir::OpRewritePattern<NDimsOp>
{
	using mlir::OpRewritePattern<NDimsOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(NDimsOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();
		auto arrayType = op.memory().getType().cast<ArrayType>();
		mlir::Value result = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(arrayType.getRank()));
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
		return mlir::success();
	}
};

struct OnesOpLowering : public ModelicaOpConversion<OnesOp>
{
	using ModelicaOpConversion<OnesOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(OnesOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto arrayType = op.resultType().cast<ArrayType>();

		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

		for (auto size : llvm::enumerate(arrayType.getShape()))
			if (size.value() == -1)
				dynamicDimensions.push_back(rewriter.create<CastOp>(loc, op.sizes()[size.index()], rewriter.getIndexType()));

		mlir::Value result = allocate(rewriter, loc, arrayType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		llvm::SmallVector<mlir::Value, 3> args;

		args.push_back(rewriter.create<ArrayCastOp>(
				loc, result,
				result.getType().cast<ArrayType>().toUnsized()));

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("ones", llvm::None, args),
				llvm::None,
				args);

		rewriter.create<mlir::CallOp>(loc, callee.getName(), llvm::None, args);
		return mlir::success();
	}
};

struct MaxOpArrayLowering : public ModelicaOpConversion<MaxOp>
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

		assert(operand.getType().isa<ArrayType>() &&
					 isNumericType(operand.getType().cast<ArrayType>().getElementType()));

		auto arrayType = operand.getType().cast<ArrayType>();
		operand = rewriter.create<ArrayCastOp>(loc, operand, arrayType.toUnsized());

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("max", arrayType.getElementType(), operand),
				arrayType.getElementType(),
				operand);

		auto call = rewriter.create<mlir::CallOp>(loc, callee.getName(), arrayType.getElementType(), operand);
		assert(call.getNumResults() == 1);
		rewriter.replaceOpWithNewOp<CastOp>(op, call->getResult(0), op.resultType());

		return mlir::success();
	}
};

struct MaxOpScalarsLowering : public ModelicaOpConversion<MaxOp>
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
				getMangledFunctionName("max", type, transformed.values()),
				type,
				transformed.values());

		auto call = rewriter.create<mlir::CallOp>(loc, callee.getName(), type, transformed.values());
		assert(call.getNumResults() == 1);
		rewriter.replaceOpWithNewOp<CastOp>(op, call->getResult(0), op.resultType());

		return mlir::success();
	}
};

struct MinOpArrayLowering : public ModelicaOpConversion<MinOp>
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

		assert(operand.getType().isa<ArrayType>() &&
					 isNumericType(operand.getType().cast<ArrayType>().getElementType()));

		auto arrayType = operand.getType().cast<ArrayType>();
		operand = rewriter.create<ArrayCastOp>(loc, operand, arrayType.toUnsized());

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("min", arrayType.getElementType(), operand),
				arrayType.getElementType(),
				operand);

		auto call = rewriter.create<mlir::CallOp>(loc, callee.getName(), arrayType.getElementType(), operand);
		assert(call.getNumResults() == 1);
		rewriter.replaceOpWithNewOp<CastOp>(op, call->getResult(0), op.resultType());

		return mlir::success();
	}
};

struct MinOpScalarsLowering : public ModelicaOpConversion<MinOp>
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
				getMangledFunctionName("min", type, transformed.values()),
				type,
				transformed.values());

		auto call = rewriter.create<mlir::CallOp>(loc, callee.getName(), type, transformed.values());
		assert(call.getNumResults() == 1);
		rewriter.replaceOpWithNewOp<CastOp>(op, call->getResult(0), op.resultType());

		return mlir::success();
	}
};

struct ProductOpLowering : public ModelicaOpConversion<ProductOp>
{
	using ModelicaOpConversion<ProductOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(ProductOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto arrayType = op.array().getType().cast<ArrayType>();

		mlir::Value arg = rewriter.create<ArrayCastOp>(
				loc, op.array(),
				op.array().getType().cast<ArrayType>().toUnsized());

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("product", arrayType.getElementType(), arg),
				arrayType.getElementType(),
				arg);

		auto call = rewriter.create<mlir::CallOp>(loc, callee.getName(), arrayType.getElementType(), arg);
		assert(call.getNumResults() == 1);
		rewriter.replaceOpWithNewOp<CastOp>(op, call->getResult(0), op.resultType());

		return mlir::success();
	}
};

struct SignOpLowering : public ModelicaOpConversion<SignOp>
{
	using ModelicaOpConversion<SignOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(SignOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto integerType = IntegerType::get(op.getContext());

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("sign", integerType, op.operand()),
				integerType,
				op.operand().getType());

		mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), integerType, op.operand()).getResult(0);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
		return mlir::success();
	}
};

struct SinOpLowering : public ModelicaOpConversion<SinOp>
{
	using ModelicaOpConversion<SinOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(SinOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto realType = RealType::get(op.getContext());

		mlir::Value arg = rewriter.create<CastOp>(loc, op.operand(), realType);

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("sin", realType, arg),
				realType,
				arg.getType());

		mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
		return mlir::success();
	}
};

struct SinhOpLowering : public ModelicaOpConversion<SinhOp>
{
	using ModelicaOpConversion<SinhOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(SinhOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto realType = RealType::get(op.getContext());

		mlir::Value arg = rewriter.create<CastOp>(loc, op.operand(), realType);

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("sinh", realType, arg),
				realType,
				arg.getType());

		mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
		return mlir::success();
	}
};

/**
 * Get the size of a specific array dimension.
 */
struct SizeOpDimensionLowering : public mlir::OpRewritePattern<SizeOp>
{
	using mlir::OpRewritePattern<SizeOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(SizeOp op, mlir::PatternRewriter& rewriter) const override
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
 * Get the size of all the array dimensions.
 */
struct SizeOpArrayLowering : public mlir::OpRewritePattern<SizeOp>
{
	using mlir::OpRewritePattern<SizeOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(SizeOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		if (op.hasIndex())
			return rewriter.notifyMatchFailure(op, "Index specified");

		assert(op.resultType().isa<ArrayType>());
		auto resultType = op.resultType().cast<ArrayType>();
		mlir::Value result = allocate(rewriter, loc, resultType, llvm::None);

		// Iterate on each dimension
		mlir::Value zeroValue = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(0));

		auto arrayType = op.memory().getType().cast<ArrayType>();
		mlir::Value rank = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(arrayType.getRank()));
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

struct SqrtOpLowering : public ModelicaOpConversion<SqrtOp>
{
	using ModelicaOpConversion<SqrtOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(SqrtOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto realType = RealType::get(op.getContext());

		mlir::Value arg = rewriter.create<CastOp>(loc, op.operand(), realType);

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("sqrt", realType, arg),
				realType,
				arg.getType());

		mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
		return mlir::success();
	}
};

struct SumOpLowering : public ModelicaOpConversion<SumOp>
{
	using ModelicaOpConversion<SumOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(SumOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto arrayType = op.array().getType().cast<ArrayType>();

		mlir::Value arg = rewriter.create<ArrayCastOp>(
				loc, op.array(),
				op.array().getType().cast<ArrayType>().toUnsized());

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("sum", arrayType.getElementType(), arg),
				arrayType.getElementType(),
				arg);

		auto call = rewriter.create<mlir::CallOp>(loc, callee.getName(), arrayType.getElementType(), arg);
		assert(call.getNumResults() == 1);
		rewriter.replaceOpWithNewOp<CastOp>(op, call->getResult(0), op.resultType());

		return mlir::success();
	}
};

struct SymmetricOpLowering : public ModelicaOpConversion<SymmetricOp>
{
	using ModelicaOpConversion<SymmetricOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(SymmetricOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto arrayType = op.matrix().getType().cast<ArrayType>();
		auto shape = arrayType.getShape();

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

		mlir::Value result = allocate(rewriter, loc, arrayType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		llvm::SmallVector<mlir::Value, 3> args;

		args.push_back(rewriter.create<ArrayCastOp>(
				loc, result,
				result.getType().cast<ArrayType>().toUnsized()));

		args.push_back(rewriter.create<ArrayCastOp>(
				loc, op.matrix(),
				op.matrix().getType().cast<ArrayType>().toUnsized()));

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("symmetric", llvm::None, args),
				llvm::None,
				args);

		rewriter.create<mlir::CallOp>(loc, callee.getName(), llvm::None, args);
		return mlir::success();
	}
};

struct TanOpLowering : public ModelicaOpConversion<TanOp>
{
	using ModelicaOpConversion<TanOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(TanOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto realType = RealType::get(op.getContext());

		mlir::Value arg = rewriter.create<CastOp>(loc, op.operand(), realType);

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("tan", realType, arg),
				realType,
				arg.getType());

		mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
		return mlir::success();
	}
};

struct TanhOpLowering : public ModelicaOpConversion<TanhOp>
{
	using ModelicaOpConversion<TanhOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(TanhOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto realType = RealType::get(op.getContext());

		mlir::Value arg = rewriter.create<CastOp>(loc, op.operand(), realType);

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("tanh", realType, arg),
				realType,
				arg.getType());

		mlir::Value result = rewriter.create<mlir::CallOp>(loc, callee.getName(), realType, arg).getResult(0);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op.resultType());
		return mlir::success();
	}
};

struct TransposeOpLowering : public ModelicaOpConversion<TransposeOp>
{
	using ModelicaOpConversion<TransposeOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(TransposeOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto arrayType = op.resultType().cast<ArrayType>();
		auto shape = arrayType.getShape();

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

		mlir::Value result = allocate(rewriter, loc, arrayType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		llvm::SmallVector<mlir::Value, 3> args;

		args.push_back(rewriter.create<ArrayCastOp>(
				loc, result,
				result.getType().cast<ArrayType>().toUnsized()));

		args.push_back(rewriter.create<ArrayCastOp>(
				loc, op.matrix(),
				op.matrix().getType().cast<ArrayType>().toUnsized()));

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("transpose", llvm::None, args),
				llvm::None,
				args);

		rewriter.create<mlir::CallOp>(loc, callee.getName(), llvm::None, args);
		return mlir::success();
	}
};

struct ZerosOpLowering : public ModelicaOpConversion<ZerosOp>
{
	using ModelicaOpConversion<ZerosOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(ZerosOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto arrayType = op.resultType().cast<ArrayType>();

		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

		for (auto size : llvm::enumerate(arrayType.getShape()))
			if (size.value() == -1)
				dynamicDimensions.push_back(rewriter.create<CastOp>(loc, op.sizes()[size.index()], rewriter.getIndexType()));

		mlir::Value result = allocate(rewriter, loc, arrayType, dynamicDimensions);
		rewriter.replaceOp(op, result);

		llvm::SmallVector<mlir::Value, 3> args;

		args.push_back(rewriter.create<ArrayCastOp>(
				loc, result,
				result.getType().cast<ArrayType>().toUnsized()));

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("zeros", llvm::None, args),
				llvm::None,
				args);

		rewriter.create<mlir::CallOp>(loc, callee.getName(), llvm::None, args);
		return mlir::success();
	}
};

struct PrintOpLowering : public ModelicaOpConversion<PrintOp>
{
	using ModelicaOpConversion<PrintOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(PrintOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		mlir::Value arg = op.value();

		if (auto arrayType = arg.getType().dyn_cast<ArrayType>())
			arg = rewriter.create<ArrayCastOp>(loc, arg, arrayType.toUnsized());

		auto callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("print", llvm::None, arg),
				llvm::None,
				arg.getType());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), llvm::None, arg);
		return mlir::success();
	}
};

class FunctionConversionPass : public mlir::PassWrapper<FunctionConversionPass, mlir::OperationPass<mlir::ModuleOp>>
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
		target.addIllegalOp<FunctionOp, MemberCreateOp>();

		// Provide the set of patterns that will lower the Modelica operations
		mlir::OwningRewritePatternList patterns(&getContext());
		patterns.insert<FunctionOpLowering, MemberAllocOpLowering>(&getContext());

		// With the target and rewrite patterns defined, we can now attempt the
		// conversion. The conversion will signal failure if any of our "illegal"
		// operations were not converted successfully.

		if (failed(applyPartialConversion(module, target, std::move(patterns))))
		{
			mlir::emitError(module.getLoc(), "Error in converting the Modelica functions\n");
			return signalPassFailure();
		}
	}
};

std::unique_ptr<mlir::Pass> marco::codegen::createFunctionConversionPass()
{
	return std::make_unique<FunctionConversionPass>();
}

static void populateModelicaConversionPatterns(
		mlir::OwningRewritePatternList& patterns,
		mlir::MLIRContext* context,
		marco::codegen::TypeConverter& typeConverter,
		ModelicaConversionOptions options)
{
	patterns.insert<
	    AssignmentOpScalarLowering,
			AssignmentOpArrayLowering,
			CallOpLowering,
			NotOpArrayLowering,
			NegateOpArrayLowering,
			AddElementWiseOpMixedLowering,
			SubElementWiseOpMixedLowering,
			MulOpScalarProductLowering,
			MulElementWiseOpScalarProductLowering,
			DivOpArrayScalarLowering,
			DivElementWiseOpMixedLowering,
			NDimsOpLowering,
			SizeOpDimensionLowering,
			SizeOpArrayLowering>(context);

	patterns.insert<
			AndOpArrayLowering,
			OrOpArrayLowering,
			AddOpArraysLowering,
			AddElementWiseOpArraysLowering,
			SubOpArraysLowering,
			SubElementWiseOpArraysLowering,
			MulElementWiseOpArraysLowering,
			DivElementWiseOpArraysLowering>(context, options);

	patterns.insert<
			ConstantOpLowering,
			NotOpScalarLowering,
			AndOpScalarLowering,
			OrOpScalarLowering,
			EqOpLowering,
			NotEqOpLowering,
			GtOpLowering,
			GteOpLowering,
			LtOpLowering,
			LteOpLowering,
			NegateOpScalarLowering,
			AddOpScalarsLowering,
			AddElementWiseOpScalarsLowering,
			SubOpScalarsLowering,
			SubElementWiseOpScalarsLowering,
			MulOpScalarsLowering,
			MulElementWiseOpScalarsLowering,
			MulOpCrossProductLowering,
			MulOpVectorMatrixLowering,
			MulOpMatrixVectorLowering,
			MulOpMatrixLowering,
			DivOpScalarsLowering,
			DivElementWiseOpScalarsLowering,
			PowOpLowering,
			PowOpMatrixLowering>(context, typeConverter, options);

	patterns.insert<
	    AbsOpLowering,
			AcosOpLowering,
			AsinOpLowering,
			AtanOpLowering,
			Atan2OpLowering,
			CosOpLowering,
			CoshhOpLowering,
			DiagonalOpLowering,
			ExpOpLowering,
			IdentityOpLowering,
			FillOpLowering,
			LinspaceOpLowering,
			LogOpLowering,
			Log10OpLowering,
			MaxOpArrayLowering,
			MaxOpScalarsLowering,
			MinOpArrayLowering,
			MinOpScalarsLowering,
			OnesOpLowering,
			ProductOpLowering,
			SignOpLowering,
			SinOpLowering,
			SinhOpLowering,
			SqrtOpLowering,
			SumOpLowering,
			SymmetricOpLowering,
			TanOpLowering,
			TanhOpLowering,
			TransposeOpLowering,
			ZerosOpLowering>(context, typeConverter, options);

	patterns.insert<PrintOpLowering>(context, typeConverter, options);
}

class ModelicaConversionPass : public mlir::PassWrapper<ModelicaConversionPass, mlir::OperationPass<mlir::ModuleOp>>
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
		if (mlir::failed(convertOperations()))
		{
			mlir::emitError(getOperation().getLoc(), "Error in converting the Modelica operations\n");
			return signalPassFailure();
		}

		if (mlir::failed(bufferForwarding()))
		{
			mlir::emitError(getOperation().getLoc(), "Error in forwarding the buffers\n");
			return signalPassFailure();
		}
	}

	private:
	mlir::LogicalResult convertOperations()
	{
		auto module = getOperation();
		mlir::ConversionTarget target(getContext());

		target.addIllegalOp<
				ConstantOp,
				AssignmentOp,
				CallOp,
				FillOp,
				NotOp, AndOp, OrOp,
				EqOp, NotEqOp, GtOp, GteOp, LtOp, LteOp,
				NegateOp,
				AddOp, AddElementWiseOp,
				SubOp, SubElementWiseOp,
				MulOp, MulElementWiseOp,
				DivOp, DivElementWiseOp,
				PowOp, PowElementWiseOp>();

		target.addIllegalOp<
		    AbsOp, AcosOp, AsinOp, AtanOp, Atan2Op, CosOp, CoshOp, DiagonalOp,
				ExpOp, IdentityOp, FillOp, LinspaceOp, LogOp, Log10Op, MaxOp, MinOp,
				NDimsOp, OnesOp, ProductOp, SignOp, SinOp, SinhOp, SizeOp, SqrtOp,
				SumOp, SymmetricOp, TanOp, TanhOp, TransposeOp, ZerosOp>();

		target.addIllegalOp<PrintOp>();

		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });

		mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
		TypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);

		mlir::OwningRewritePatternList patterns(&getContext());
		populateModelicaConversionPatterns(patterns, &getContext(), typeConverter, options);

		if (auto status = applyPartialConversion(module, target, std::move(patterns)); failed(status))
			return status;

		return mlir::success();
	}

	mlir::LogicalResult bufferForwarding()
	{
		auto module = getOperation();

		// Erase the clone operations for which a forward of the original
		// allocation is enough. The buffer forwarding is possible only when
		// the clone has the same type of the source, including the allocation
		// scope.

		module.walk([](ArrayCloneOp op) {
			if (!op.canSourceBeForwarded())
				return;

			if (auto arrayType = op.source().getType().dyn_cast<ArrayType>())
			{
				if (arrayType != op.resultType())
					return;

				mlir::Operation* sourceOp = op.source().getDefiningOp();

				if (auto sourceAllocator = mlir::dyn_cast<HeapAllocator>(sourceOp))
				{
					bool shouldBeFreed = sourceAllocator.shouldBeFreed();

					for (const auto& user : op.source().getUsers())
						if (auto userAllocator = mlir::dyn_cast<HeapAllocator>(user))
							shouldBeFreed &= userAllocator.shouldBeFreed();

					if (shouldBeFreed)
						sourceAllocator.setAsAutomaticallyFreed();
					else
						sourceAllocator.setAsManuallyFreed();
				}

				op.replaceAllUsesWith(op.source());
				op.erase();
			}
		});

		// The remaining clone operations can't be optimized more, so just
		// convert them into naive copies.

		mlir::ConversionTarget target(getContext());

		target.addIllegalOp<ArrayCloneOp>();
		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });

		mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
		TypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);

		mlir::OwningRewritePatternList patterns(&getContext());
		patterns.insert<ArrayCloneOpLowering>(&getContext(), typeConverter, options);

		if (auto status = applyPartialConversion(module, target, std::move(patterns)); failed(status))
			return status;

		return mlir::success();
	}

	ModelicaConversionOptions options;
	unsigned int bitWidth;
};

std::unique_ptr<mlir::Pass> marco::codegen::createModelicaConversionPass(ModelicaConversionOptions options, unsigned int bitWidth)
{
	return std::make_unique<ModelicaConversionPass>(options, bitWidth);
}
