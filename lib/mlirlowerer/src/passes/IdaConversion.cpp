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

	[[nodiscard]] mlir::Type convertFunctionType(mlir::Type type) const
	{
		mlir::LLVM::LLVMFunctionType functionType =
				type.cast<mlir::LLVM::LLVMPointerType>().getElementType().cast<mlir::LLVM::LLVMFunctionType>();

		mlir::Type returnType = convertType(functionType.getReturnType());
		llvm::SmallVector<mlir::Type, 4> argTypes;
		for (unsigned int i = 0; i < functionType.getNumParams(); i++)
			argTypes.push_back(convertType(functionType.getParamType(i)));

		return mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMFunctionType::get(returnType, argTypes));
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

struct ConstantValueOpLowering : public IdaOpConversion<ConstantValueOp>
{
	using IdaOpConversion<ConstantValueOp>::IdaOpConversion;

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

		if (auto booleanAttribute = attribute.dyn_cast<BooleanAttribute>())
			return builder.getBoolAttr(booleanAttribute.getValue());

		if (auto integerAttribute = attribute.dyn_cast<IntegerAttribute>())
			return builder.getIntegerAttr(convertType(resultType), integerAttribute.getValue());

		if (auto realAttribute = attribute.dyn_cast<RealAttribute>())
			return builder.getFloatAttr(convertType(resultType), realAttribute.getValue());

		assert(false && "Unreachable");
	}
};

struct AllocDataOpLowering : public IdaOpConversion<AllocDataOp>
{
	using IdaOpConversion<AllocDataOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(AllocDataOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		OpaquePointerType result = OpaquePointerType::get(op.getContext());

		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("idaAllocData", result, op.args()),
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

struct FreeDataOpLowering : public IdaOpConversion<FreeDataOp>
{
	using IdaOpConversion<FreeDataOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(FreeDataOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		BooleanType result = BooleanType::get(op.getContext());

		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("idaFreeData", result, op.args()),
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
		// Update the address-of operation with the lowered type of the LLVM function.
		FuncAddressOfOp addressOfOp = mlir::cast<FuncAddressOfOp>(op.residualAddress().getDefiningOp());
		mlir::Type residualType = convertFunctionType(addressOfOp.resultType());

		FuncAddressOfOp newAddressOfOp = rewriter.create<FuncAddressOfOp>(addressOfOp.getLoc(), addressOfOp.callee(), residualType);
		addressOfOp->replaceAllUsesWith(newAddressOfOp);
		addressOfOp->erase();

		// Replace the operation with a MLIR function operation.
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
		// Update the address-of operation with the lowered type of the LLVM function.
		FuncAddressOfOp addressOfOp = mlir::cast<FuncAddressOfOp>(op.jacobianAddress().getDefiningOp());
		mlir::Type jacobianType = convertFunctionType(addressOfOp.resultType());
		
		FuncAddressOfOp newAddressOfOp = rewriter.create<FuncAddressOfOp>(addressOfOp.getLoc(), addressOfOp.callee(), jacobianType);
		addressOfOp->replaceAllUsesWith(newAddressOfOp);
		addressOfOp->erase();

		// Replace the operation with a MLIR function operation.
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

struct AddVariableOpLowering : public IdaOpConversion<AddVariableOp>
{
	using IdaOpConversion<AddVariableOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(AddVariableOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		// Cast sized array into unsized array.
		auto arrayType = op.array().getType().cast<modelica::ArrayType>();
		llvm::SmallVector<mlir::Value, 4> args = op.args();
		args[2] = rewriter.create<modelica::ArrayCastOp>(op.getLoc(), op.array(), arrayType.toUnsized());

		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("addVariable", llvm::None, args),
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

struct ResidualFunctionOpLowering : public IdaOpConversion<ResidualFunctionOp>
{
	using IdaOpConversion<ResidualFunctionOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(ResidualFunctionOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::FuncOp function = rewriter.replaceOpWithNewOp<mlir::FuncOp>(op, op.name(), op.getType());
		rewriter.inlineRegionBefore(op.getBody(), function.getBody(), function.getBody().begin());
		return mlir::success();
	}
};

struct JacobianFunctionOpLowering : public IdaOpConversion<JacobianFunctionOp>
{
	using IdaOpConversion<JacobianFunctionOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(JacobianFunctionOp op, llvm::ArrayRef<mlir::Value> operands,  mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::FuncOp function = rewriter.replaceOpWithNewOp<mlir::FuncOp>(op, op.name(), op.getType());
		rewriter.inlineRegionBefore(op.getBody(), function.getBody(), function.getBody().begin());
		return mlir::success();
	}
};

struct FunctionTerminatorOpLowering : public IdaOpConversion<FunctionTerminatorOp>
{
	using IdaOpConversion<FunctionTerminatorOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(FunctionTerminatorOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op, op.returnValue());
		return mlir::success();
	}
};

struct PrintStatisticsOpLowering : public IdaOpConversion<PrintStatisticsOp>
{
	using IdaOpConversion<PrintStatisticsOp>::IdaOpConversion;

	mlir::LogicalResult matchAndRewrite(PrintStatisticsOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::FuncOp callee = getOrDeclareFunction(
				rewriter,
				op->getParentOfType<mlir::ModuleOp>(),
				getMangledFunctionName("printStatistics", llvm::None, op.args()),
				llvm::None,
				op.args());

		rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), llvm::None, op.args());
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
			AllocDataOpLowering,
			InitOpLowering,
			StepOpLowering,
			FreeDataOpLowering,
			AddTimeOpLowering,
			AddToleranceOpLowering>(typeConverter, context);

	// Equation setters.
	patterns.insert<
			AddColumnIndexOpLowering,
			AddEqDimensionOpLowering,
			AddResidualOpLowering,
			AddJacobianOpLowering>(typeConverter, context);

	// Variable setters.
	patterns.insert<
			AddVariableOpLowering,
			AddVarAccessOpLowering>(typeConverter, context);

	// Getters.
	patterns.insert<GetTimeOpLowering>(typeConverter, context);

	// Residual and Jacobian construction helpers.
	patterns.insert<
			ResidualFunctionOpLowering,
			JacobianFunctionOpLowering,
			FunctionTerminatorOpLowering>(typeConverter, context);

	// Statistics.
	patterns.insert<PrintStatisticsOpLowering>(typeConverter, context);
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
				AllocDataOp,
				InitOp,
				StepOp,
				FreeDataOp,
				AddTimeOp,
				AddToleranceOp>();

		// Equation setters.
		target.addIllegalOp<
				AddColumnIndexOp,
				AddEqDimensionOp,
				AddResidualOp,
				AddJacobianOp>();

		// Variable setters.
		target.addIllegalOp<AddVariableOp, AddVarAccessOp>();

		// Getters.
		target.addIllegalOp<GetTimeOp>();

		// Residual and Jacobian construction helpers.
		target.addIllegalOp<ResidualFunctionOp, JacobianFunctionOp, FunctionTerminatorOp>();

		// Statistics.
		target.addIllegalOp<PrintStatisticsOp>();

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
