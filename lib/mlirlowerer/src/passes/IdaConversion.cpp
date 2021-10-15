#include <mlir/Transforms/DialectConversion.h>
#include <marco/mlirlowerer/dialects/ida/IdaDialect.h>
#include <marco/mlirlowerer/dialects/modelica/ModelicaDialect.h>
#include <marco/mlirlowerer/passes/IdaConversion.h>
#include <marco/mlirlowerer/passes/TypeConverter.h>
#include <numeric>

using namespace marco::codegen;
using namespace ida;

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

template<typename FromOp>
class IdaOpConversion : public mlir::OpConversionPattern<FromOp>
{
	public:
	IdaOpConversion(TypeConverter& typeConverter, mlir::MLIRContext* context)
			: mlir::OpConversionPattern<FromOp>(typeConverter, context)
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
		if (type.isa<BooleanType>() || type.isa<modelica::BooleanType>())
			return "i1";

		if (auto integerType = type.dyn_cast<IntegerType>())
			return "i" + std::to_string(convertType(integerType).getIntOrFloatBitWidth());

		if (auto integerType = type.dyn_cast<modelica::IntegerType>())
			return "i" + std::to_string(convertType(integerType).getIntOrFloatBitWidth());

		if (auto realType = type.dyn_cast<RealType>())
			return "f" + std::to_string(convertType(realType).getIntOrFloatBitWidth());

		if (auto realType = type.dyn_cast<modelica::RealType>())
			return "f" + std::to_string(convertType(realType).getIntOrFloatBitWidth());

		if (type.isa<OpaquePointerType>())
			return "pvoid";

		if (auto arrayType = type.dyn_cast<modelica::UnsizedArrayType>())
			return "a" + getMangledType(arrayType.getElementType());

		if (type.isa<mlir::IndexType>())
			return getMangledType(convertType(type));

		if (auto integerType = type.dyn_cast<mlir::IntegerType>())
			return "i" + std::to_string(integerType.getWidth());

		if (auto floatType = type.dyn_cast<mlir::FloatType>())
			return "f" + std::to_string(floatType.getWidth());

		if (auto ptrType = type.dyn_cast<mlir::LLVM::LLVMPointerType>())
		{
			auto funcType = ptrType.getElementType().cast<mlir::LLVM::LLVMFunctionType>();
			return getMangledType(funcType.getReturnType()) + "ptr";
		}

		assert(false && "Unreachable");
	}
};

struct ConstantValueOpLowering : public mlir::OpConversionPattern<ConstantValueOp>
{
	using mlir::OpConversionPattern<ConstantValueOp>::OpConversionPattern;

	mlir::LogicalResult matchAndRewrite(ConstantValueOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		llvm::Optional<mlir::Attribute> attribute = convertAttribute(rewriter, op.resultType(), op.value());

		rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, *attribute);
		return mlir::success();
	}

	private:
	llvm::Optional<mlir::Attribute> convertAttribute(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Attribute attribute) const
	{
		if (attribute.getType().isa<mlir::IndexType>())
			return attribute;

		resultType = getTypeConverter()->convertType(resultType);

		if (auto booleanAttribute = attribute.dyn_cast<BooleanAttribute>())
			return builder.getBoolAttr(booleanAttribute.getValue());

		if (auto integerAttribute = attribute.dyn_cast<IntegerAttribute>())
			return builder.getIntegerAttr(resultType, integerAttribute.getValue());

		if (auto realAttribute = attribute.dyn_cast<RealAttribute>())
			return builder.getFloatAttr(resultType, realAttribute.getValue());

		assert(false && "Unreachable");
	}
};

struct AllocUserDataOpLowering : public IdaOpConversion<AllocUserDataOp>
{
	using IdaOpConversion<AllocUserDataOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(AllocUserDataOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		OpaquePointerType result = OpaquePointerType::get(op.getContext());

		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("allocIdaUserData", result, op.args()),
				result,
				op.args());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), result, op.args());
		return mlir::success();
	}
};

struct InitOpLowering : public IdaOpConversion<InitOp>
{
	using IdaOpConversion<InitOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(InitOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		BooleanType result = BooleanType::get(op.getContext());

		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("idaInit", result, op.args()),
				result,
				op.args());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), result, op.args());
		return mlir::success();
	}
};

struct StepOpLowering : public IdaOpConversion<StepOp>
{
	using IdaOpConversion<StepOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(StepOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		BooleanType result = BooleanType::get(op.getContext());

		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("idaStep", result, op.args()),
				result,
				op.args());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), result, op.args());
		return mlir::success();
	}
};

struct FreeUserDataOpLowering : public IdaOpConversion<FreeUserDataOp>
{
	using IdaOpConversion<FreeUserDataOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(FreeUserDataOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		BooleanType result = BooleanType::get(op.getContext());

		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("freeIdaUserData", result, op.args()),
				result,
				op.args());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), result, op.args());
		return mlir::success();
	}
};

struct AddTimeOpLowering : public IdaOpConversion<AddTimeOp>
{
	using IdaOpConversion<AddTimeOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(AddTimeOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("addTime", llvm::None, op.args()),
				llvm::None,
				op.args());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), llvm::None, op.args());
		return mlir::success();
	}
};

struct AddToleranceOpLowering : public IdaOpConversion<AddToleranceOp>
{
	using IdaOpConversion<AddToleranceOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(AddToleranceOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("addTolerance", llvm::None, op.args()),
				llvm::None,
				op.args());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), llvm::None, op.args());
		return mlir::success();
	}
};

struct AddRowLengthOpLowering : public IdaOpConversion<AddRowLengthOp>
{
	using IdaOpConversion<AddRowLengthOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(AddRowLengthOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		IntegerType result = IntegerType::get(op.getContext());

		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("addRowLength", result, op.args()),
				result,
				op.args());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), result, op.args());
		return mlir::success();
	}
};

struct AddColumnIndexOpLowering : public IdaOpConversion<AddColumnIndexOp>
{
	using IdaOpConversion<AddColumnIndexOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(AddColumnIndexOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("addColumnIndex", llvm::None, op.args()),
				llvm::None,
				op.args());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), llvm::None, op.args());
		return mlir::success();
	}
};

struct AddEqDimensionOpLowering : public IdaOpConversion<AddEqDimensionOp>
{
	using IdaOpConversion<AddEqDimensionOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(AddEqDimensionOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		// Cast sized array into unsized array.
		auto arrayType = op.start().getType().cast<modelica::ArrayType>();
		llvm::SmallVector<mlir::Value, 3> args = op.args();
		args[1] = rewriter.create<modelica::ArrayCastOp>(op.getLoc(), op.start(), arrayType.toUnsized());
		args[2] = rewriter.create<modelica::ArrayCastOp>(op.getLoc(), op.end(), arrayType.toUnsized());

		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("addEqDimension", llvm::None, args),
				llvm::None,
				mlir::ValueRange(args));

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), llvm::None, args);
		return mlir::success();
	}
};

struct AddResidualOpLowering : public IdaOpConversion<AddResidualOp>
{
	using IdaOpConversion<AddResidualOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(AddResidualOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("addResidual", llvm::None, op.args()),
				llvm::None,
				op.args());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), llvm::None, op.args());
		return mlir::success();
	}
};

struct AddJacobianOpLowering : public IdaOpConversion<AddJacobianOp>
{
	using IdaOpConversion<AddJacobianOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(AddJacobianOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("addJacobian", llvm::None, op.args()),
				llvm::None,
				op.args());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), llvm::None, op.args());
		return mlir::success();
	}
};

struct GetTimeOpLowering : public IdaOpConversion<GetTimeOp>
{
	using IdaOpConversion<GetTimeOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(GetTimeOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		modelica::RealType result = modelica::RealType::get(op.getContext());

		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("getIdaTime", result, op.args()),
				result,
				op.args());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), result, op.args());
		return mlir::success();
	}
};

struct GetVariableOpLowering : public IdaOpConversion<GetVariableOp>
{
	using IdaOpConversion<GetVariableOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(GetVariableOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		modelica::RealType result = modelica::RealType::get(op.getContext());

		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("getIdaVariable", result, op.args()),
				result,
				op.args());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), result, op.args());
		return mlir::success();
	}
};

struct GetDerivativeOpLowering : public IdaOpConversion<GetDerivativeOp>
{
	using IdaOpConversion<GetDerivativeOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(GetDerivativeOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		modelica::RealType result = modelica::RealType::get(op.getContext());

		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("getIdaDerivative", result, op.args()),
				result,
				op.args());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), result, op.args());
		return mlir::success();
	}
};

struct AddVarOffsetOpLowering : public IdaOpConversion<AddVarOffsetOp>
{
	using IdaOpConversion<AddVarOffsetOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(AddVarOffsetOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		IntegerType result = IntegerType::get(op.getContext());

		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("addVarOffset", result, op.args()),
				result,
				op.args());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), result, op.args());
		return mlir::success();
	}
};

struct AddVarDimensionOpLowering : public IdaOpConversion<AddVarDimensionOp>
{
	using IdaOpConversion<AddVarDimensionOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(AddVarDimensionOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		// Cast sized array into unsized array.
		auto arrayType = op.dimensions().getType().cast<modelica::ArrayType>();
		llvm::SmallVector<mlir::Value, 2> args = op.args();
		args[1] = rewriter.create<modelica::ArrayCastOp>(op.getLoc(), op.dimensions(), arrayType.toUnsized());

		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("addVarDimension", llvm::None, args),
				llvm::None,
				mlir::ValueRange(args));

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), llvm::None, args);
		return mlir::success();
	}
};

struct AddVarAccessOpLowering : public IdaOpConversion<AddVarAccessOp>
{
	using IdaOpConversion<AddVarAccessOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(AddVarAccessOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		// Cast sized array into unsized array.
		auto arrayType = op.offsets().getType().cast<modelica::ArrayType>();
		llvm::SmallVector<mlir::Value, 4> args = op.args();
		args[2] = rewriter.create<modelica::ArrayCastOp>(op.getLoc(), op.offsets(), arrayType.toUnsized());
		args[3] = rewriter.create<modelica::ArrayCastOp>(op.getLoc(), op.inductions(), arrayType.toUnsized());

		IntegerType result = IntegerType::get(op.getContext());

		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("addVarAccess", result, args),
				result,
				mlir::ValueRange(args));

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), result, args);
		return mlir::success();
	}
};

struct SetInitialValueOpLowering : public IdaOpConversion<SetInitialValueOp>
{
	using IdaOpConversion<SetInitialValueOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(SetInitialValueOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("setInitialValue", llvm::None, op.args()),
				llvm::None,
				op.args());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), llvm::None, op.args());
		return mlir::success();
	}
};

struct SetInitialArrayOpLowering : public IdaOpConversion<SetInitialArrayOp>
{
	using IdaOpConversion<SetInitialArrayOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(SetInitialArrayOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		// Cast sized array into unsized array.
		auto arrayType = op.array().getType().cast<modelica::ArrayType>();
		llvm::SmallVector<mlir::Value, 5> args = op.args();
		args[3] = rewriter.create<modelica::ArrayCastOp>(op.getLoc(), op.array(), arrayType.toUnsized());

		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("setInitialArray", llvm::None, args),
				llvm::None,
				mlir::ValueRange(args));

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), llvm::None, args);
		return mlir::success();
	}
};

template<typename FromOp, typename FromOpLowering>
struct LambdaLikeLowering : public IdaOpConversion<FromOp>
{
	using IdaOpConversion<FromOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(FromOp fromOp, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Operation* op = static_cast<mlir::Operation*>(fromOp);
		IntegerType result = IntegerType::get(fromOp.getContext());

		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				this->getMangledFunctionName(FromOpLowering::operationName, result, fromOp.args()),
				result,
				fromOp.args());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(fromOp, callee.getName(), result, fromOp.args());
		return mlir::success();
	}
};

struct LambdaConstantOpLowering : public LambdaLikeLowering<LambdaConstantOp, LambdaConstantOpLowering>
{
	using LambdaLikeLowering<LambdaConstantOp, LambdaConstantOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaConstant";
};

struct LambdaTimeOpLowering : public LambdaLikeLowering<LambdaTimeOp, LambdaTimeOpLowering>
{
	using LambdaLikeLowering<LambdaTimeOp, LambdaTimeOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaTime";
};

struct LambdaInductionOpLowering : public LambdaLikeLowering<LambdaInductionOp, LambdaInductionOpLowering>
{
	using LambdaLikeLowering<LambdaInductionOp, LambdaInductionOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaInduction";
};

struct LambdaVariableOpLowering : public LambdaLikeLowering<LambdaVariableOp, LambdaVariableOpLowering>
{
	using LambdaLikeLowering<LambdaVariableOp, LambdaVariableOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaVariable";
};

struct LambdaDerivativeOpLowering : public LambdaLikeLowering<LambdaDerivativeOp, LambdaDerivativeOpLowering>
{
	using LambdaLikeLowering<LambdaDerivativeOp, LambdaDerivativeOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaDerivative";
};

struct LambdaAddOpLowering : public LambdaLikeLowering<LambdaAddOp, LambdaAddOpLowering>
{
	using LambdaLikeLowering<LambdaAddOp, LambdaAddOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaAdd";
};

struct LambdaSubOpLowering : public LambdaLikeLowering<LambdaSubOp, LambdaSubOpLowering>
{
	using LambdaLikeLowering<LambdaSubOp, LambdaSubOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaSub";
};

struct LambdaMulOpLowering : public LambdaLikeLowering<LambdaMulOp, LambdaMulOpLowering>
{
	using LambdaLikeLowering<LambdaMulOp, LambdaMulOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaMul";
};

struct LambdaDivOpLowering : public LambdaLikeLowering<LambdaDivOp, LambdaDivOpLowering>
{
	using LambdaLikeLowering<LambdaDivOp, LambdaDivOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaDiv";
};

struct LambdaPowOpLowering : public LambdaLikeLowering<LambdaPowOp, LambdaPowOpLowering>
{
	using LambdaLikeLowering<LambdaPowOp, LambdaPowOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaPow";
};

struct LambdaNegateOpLowering : public LambdaLikeLowering<LambdaNegateOp, LambdaNegateOpLowering>
{
	using LambdaLikeLowering<LambdaNegateOp, LambdaNegateOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaNegate";
};

struct LambdaAbsOpLowering : public LambdaLikeLowering<LambdaAbsOp, LambdaAbsOpLowering>
{
	using LambdaLikeLowering<LambdaAbsOp, LambdaAbsOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaAbs";
};

struct LambdaSignOpLowering : public LambdaLikeLowering<LambdaSignOp, LambdaSignOpLowering>
{
	using LambdaLikeLowering<LambdaSignOp, LambdaSignOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaSign";
};

struct LambdaSqrtOpLowering : public LambdaLikeLowering<LambdaSqrtOp, LambdaSqrtOpLowering>
{
	using LambdaLikeLowering<LambdaSqrtOp, LambdaSqrtOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaSqrt";
};

struct LambdaExpOpLowering : public LambdaLikeLowering<LambdaExpOp, LambdaExpOpLowering>
{
	using LambdaLikeLowering<LambdaExpOp, LambdaExpOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaExp";
};

struct LambdaLogOpLowering : public LambdaLikeLowering<LambdaLogOp, LambdaLogOpLowering>
{
	using LambdaLikeLowering<LambdaLogOp, LambdaLogOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaLog";
};

struct LambdaLog10OpLowering : public LambdaLikeLowering<LambdaLog10Op, LambdaLog10OpLowering>
{
	using LambdaLikeLowering<LambdaLog10Op, LambdaLog10OpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaLog10";
};

struct LambdaSinOpLowering : public LambdaLikeLowering<LambdaSinOp, LambdaSinOpLowering>
{
	using LambdaLikeLowering<LambdaSinOp, LambdaSinOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaSin";
};

struct LambdaCosOpLowering : public LambdaLikeLowering<LambdaCosOp, LambdaCosOpLowering>
{
	using LambdaLikeLowering<LambdaCosOp, LambdaCosOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaCos";
};

struct LambdaTanOpLowering : public LambdaLikeLowering<LambdaTanOp, LambdaTanOpLowering>
{
	using LambdaLikeLowering<LambdaTanOp, LambdaTanOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaTan";
};

struct LambdaAsinOpLowering : public LambdaLikeLowering<LambdaAsinOp, LambdaAsinOpLowering>
{
	using LambdaLikeLowering<LambdaAsinOp, LambdaAsinOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaAsin";
};

struct LambdaAcosOpLowering : public LambdaLikeLowering<LambdaAcosOp, LambdaAcosOpLowering>
{
	using LambdaLikeLowering<LambdaAcosOp, LambdaAcosOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaAcos";
};

struct LambdaAtanOpLowering : public LambdaLikeLowering<LambdaAtanOp, LambdaAtanOpLowering>
{
	using LambdaLikeLowering<LambdaAtanOp, LambdaAtanOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaAtan";
};

struct LambdaAtan2OpLowering : public LambdaLikeLowering<LambdaAtan2Op, LambdaAtan2OpLowering>
{
	using LambdaLikeLowering<LambdaAtan2Op, LambdaAtan2OpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaAtan2";
};

struct LambdaSinhOpLowering : public LambdaLikeLowering<LambdaSinhOp, LambdaSinhOpLowering>
{
	using LambdaLikeLowering<LambdaSinhOp, LambdaSinhOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaSinh";
};

struct LambdaCoshOpLowering : public LambdaLikeLowering<LambdaCoshOp, LambdaCoshOpLowering>
{
	using LambdaLikeLowering<LambdaCoshOp, LambdaCoshOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaCosh";
};

struct LambdaTanhOpLowering : public LambdaLikeLowering<LambdaTanhOp, LambdaTanhOpLowering>
{
	using LambdaLikeLowering<LambdaTanhOp, LambdaTanhOpLowering>::LambdaLikeLowering;
	static constexpr llvm::StringRef operationName = "lambdaTanh";
};

struct LambdaCallOpLowering : public IdaOpConversion<LambdaCallOp>
{
	using IdaOpConversion<LambdaCallOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(LambdaCallOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		IntegerType result = IntegerType::get(op.getContext());

		// Update the address-of operation with the lowered type of the LLVM function.
		mlir::Type realType = getTypeConverter()->convertType(RealType::get(rewriter.getContext()));

		LambdaAddressOfOp oldAddressOfOp = mlir::cast<LambdaAddressOfOp>(op.functionAddress().getDefiningOp());
		LambdaAddressOfOp newAddressOfOp = rewriter.create<LambdaAddressOfOp>(op.getLoc(), oldAddressOfOp.callee(), realType);
		oldAddressOfOp->replaceAllUsesWith(newAddressOfOp);
		oldAddressOfOp->erase();

		oldAddressOfOp = mlir::cast<LambdaAddressOfOp>(op.pderAddress().getDefiningOp());
		newAddressOfOp = rewriter.create<LambdaAddressOfOp>(op.getLoc(), oldAddressOfOp.callee(), realType);
		oldAddressOfOp->replaceAllUsesWith(newAddressOfOp);
		oldAddressOfOp->erase();

		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("lambdaCall", result, op.args()),
				result,
				op.args());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), result, op.args());
		return mlir::success();
	}
};

static void populateIdaConversionPatterns(
		mlir::OwningRewritePatternList& patterns,
		mlir::MLIRContext* context,
		marco::codegen::TypeConverter& typeConverter)
{
	// Allocation, initialization, usage and deletion.
	patterns.insert<
			ConstantValueOpLowering,
			AllocUserDataOpLowering,
			InitOpLowering,
			StepOpLowering,
			FreeUserDataOpLowering,
			AddTimeOpLowering,
			AddToleranceOpLowering>(typeConverter, context);

	// Equation setters.
	patterns.insert<
			AddRowLengthOpLowering,
			AddColumnIndexOpLowering,
			AddEqDimensionOpLowering,
			AddResidualOpLowering,
			AddJacobianOpLowering>(typeConverter, context);

	// Variable setters.
	patterns.insert<
			AddVarOffsetOpLowering,
			AddVarDimensionOpLowering,
			AddVarAccessOpLowering,
			SetInitialValueOpLowering,
			SetInitialArrayOpLowering>(typeConverter, context);

	// Getters.
	patterns.insert<
			GetTimeOpLowering,
			GetVariableOpLowering,
			GetDerivativeOpLowering>(typeConverter, context);

	// Lambda constructions.
	patterns.insert<
			LambdaConstantOpLowering,
			LambdaTimeOpLowering,
			LambdaInductionOpLowering,
			LambdaVariableOpLowering,
			LambdaDerivativeOpLowering>(typeConverter, context);

	patterns.insert<
			LambdaAddOpLowering,
			LambdaSubOpLowering,
			LambdaMulOpLowering,
			LambdaDivOpLowering,
			LambdaPowOpLowering,
			LambdaNegateOpLowering,
			LambdaAbsOpLowering,
			LambdaSignOpLowering,
			LambdaSqrtOpLowering,
			LambdaExpOpLowering,
			LambdaLogOpLowering,
			LambdaLog10OpLowering>(typeConverter, context);

	patterns.insert<
			LambdaSinOpLowering,
			LambdaCosOpLowering,
			LambdaTanOpLowering,
			LambdaAsinOpLowering,
			LambdaAcosOpLowering,
			LambdaAtanOpLowering,
			LambdaAtan2OpLowering,
			LambdaSinhOpLowering,
			LambdaCoshOpLowering,
			LambdaTanhOpLowering>(typeConverter, context);
	
	patterns.insert<LambdaCallOpLowering>(typeConverter, context);
}

class IdaConversionPass : public mlir::PassWrapper<IdaConversionPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	explicit IdaConversionPass(unsigned int bitWidth)
			: bitWidth(bitWidth)
	{
	}

	void getDependentDialects(mlir::DialectRegistry &registry) const override
	{
		registry.insert<IdaDialect>();
	}

	void runOnOperation() override
	{
		if (mlir::failed(convertOperations()))
		{
			mlir::emitError(getOperation().getLoc(), "Error in converting the Ida operations\n");
			return signalPassFailure();
		}
	}

	private:
	mlir::LogicalResult convertOperations()
	{
		auto module = getOperation();
		mlir::ConversionTarget target(getContext());

		// Allocation, initialization, usage and deletion.
		target.addIllegalOp<
				ConstantValueOp,
				AllocUserDataOp,
				InitOp,
				StepOp,
				FreeUserDataOp,
				AddTimeOp,
				AddToleranceOp>();

		// Equation setters.
		target.addIllegalOp<
				AddRowLengthOp,
				AddColumnIndexOp,
				AddEqDimensionOp,
				AddResidualOp,
				AddJacobianOp>();

		// Variable setters.
		target.addIllegalOp<
				AddVarOffsetOp,
				AddVarDimensionOp,
				AddVarAccessOp,
				SetInitialValueOp,
				SetInitialArrayOp>();

		// Getters.
		target.addIllegalOp<GetTimeOp, GetVariableOp, GetDerivativeOp>();

		// Lambda constructions.
		target.addIllegalOp<
				LambdaConstantOp,
				LambdaTimeOp,
				LambdaInductionOp,
				LambdaVariableOp,
				LambdaDerivativeOp>();

		target.addIllegalOp<
				LambdaAddOp,
				LambdaSubOp,
				LambdaMulOp,
				LambdaDivOp,
				LambdaPowOp,
				LambdaNegateOp,
				LambdaAbsOp,
				LambdaSignOp,
				LambdaSqrtOp,
				LambdaExpOp,
				LambdaLogOp,
				LambdaLog10Op>();

		target.addIllegalOp<
				LambdaSinOp,
				LambdaCosOp,
				LambdaTanOp,
				LambdaAsinOp,
				LambdaAcosOp,
				LambdaAtanOp,
				LambdaSinhOp,
				LambdaCoshOp,
				LambdaTanhOp>();

		target.addIllegalOp<LambdaCallOp>();

		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });

		mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
		TypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);

		mlir::OwningRewritePatternList patterns(&getContext());
		populateIdaConversionPatterns(patterns, &getContext(), typeConverter);

		if (auto status = applyPartialConversion(module, target, std::move(patterns)); failed(status))
			return status;

		return mlir::success();
	}

	unsigned int bitWidth;
};

std::unique_ptr<mlir::Pass> marco::codegen::createIdaConversionPass(unsigned int bitWidth)
{
	return std::make_unique<IdaConversionPass>(bitWidth);
}
