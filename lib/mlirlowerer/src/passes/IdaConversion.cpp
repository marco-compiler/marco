#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Transforms/DialectConversion.h>
#include <marco/mlirlowerer/dialects/ida/IdaDialect.h>
// #include <marco/mlirlowerer/dialects/modelica/ModelicaDialect.h>
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

/**
 * Generic conversion pattern that provides some utility functions.
 *
 * @tparam FromOp type of the operation to be converted
 */
template<typename FromOp>
class IdaOpConversion : public mlir::OpConversionPattern<FromOp>
{
	public:
	IdaOpConversion(mlir::MLIRContext* ctx, TypeConverter& typeConverter)
			: mlir::OpConversionPattern<FromOp>(typeConverter, ctx, 1)
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
};

struct ConstantValueOpLowering : public IdaOpConversion<ConstantValueOp>
{
	using IdaOpConversion<ConstantValueOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(ConstantValueOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		llvm::Optional<mlir::Attribute> attribute = convertAttribute(rewriter, op.resultType(), op.value());

		if (!attribute)
			return rewriter.notifyMatchFailure(op, "Unknown attribute type");

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

		return llvm::None;
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
				"allocIdaUserData",
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
				"freeIdaUserData",
				result,
				op.args());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), result, op.args());
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
				"setInitialValue",
				llvm::None,
				op.args());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), llvm::None, op.args());
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
				"idaInit",
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
				"idaStep",
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
				"addTime",
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
				"addTolerance",
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
		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				"addRowLength",
				llvm::None,
				op.args());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), llvm::None, op.args());
		return mlir::success();
	}
};

struct AddDimensionOpLowering : public IdaOpConversion<AddDimensionOp>
{
	using IdaOpConversion<AddDimensionOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(AddDimensionOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				"addDimension",
				llvm::None,
				op.args());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), llvm::None, op.args());
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
				"addResidual",
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
				"addJacobian",
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
		RealType result = RealType::get(op.getContext());

		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				"getIdaTime",
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
		RealType result = RealType::get(op.getContext());

		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				"getIdaVariable",
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
			FreeUserDataOpLowering,
			SetInitialValueOpLowering,
			InitOpLowering,
			StepOpLowering>(context, typeConverter);

	// Setters.
	patterns.insert<
			AddTimeOpLowering,
			AddToleranceOpLowering,
			AddRowLengthOpLowering,
			AddDimensionOpLowering,
			AddResidualOpLowering,
			AddJacobianOpLowering>(context, typeConverter);
	
	// Getters.
	patterns.insert<
			GetTimeOpLowering,
			GetVariableOpLowering>(context, typeConverter);
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
		// registry.insert<modelica::ModelicaDialect>();
		registry.insert<mlir::StandardOpsDialect>();
		registry.insert<mlir::scf::SCFDialect>();
		registry.insert<mlir::LLVM::LLVMDialect>();
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
				FreeUserDataOp,
				SetInitialValueOp,
				InitOp,
				StepOp>();

		// Setters.
		target.addIllegalOp<
				AddTimeOp,
				AddToleranceOp,
				AddRowLengthOp,
				AddDimensionOp,
				AddResidualOp,
				AddJacobianOp>();

		// Getters.
		target.addIllegalOp<GetTimeOp, GetVariableOp>();

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
