#include "marco/codegen/dialects/modelica/ModelicaDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace marco::codegen::modelica;

class ModelicaInlinerInterface : public mlir::DialectInlinerInterface
{
	public:
	using mlir::DialectInlinerInterface::DialectInlinerInterface;

	bool isLegalToInline(mlir::Operation* call, mlir::Operation* callable, bool wouldBeCloned) const final
	{
		if (!mlir::isa<FunctionOp>(callable))
			return false;

		auto function = mlir::cast<FunctionOp>(callable);

		if (!function->hasAttr("inline"))
			return false;

		auto inlineAttribute = function->getAttrOfType<mlir::BoolAttr>("inline");
		return inlineAttribute.getValue();
	}

	bool isLegalToInline(mlir::Operation* op, mlir::Region* dest, bool wouldBeCloned, mlir::BlockAndValueMapping& valueMapping) const final
	{
		return false;
	}

	void handleTerminator(mlir::Operation* op, llvm::ArrayRef<mlir::Value> valuesToReplace) const final {
		auto functionOp = op->getParentOfType<FunctionOp>();

		// Map the members for a faster access
		llvm::StringMap<mlir::Value> members;

		functionOp->walk([&members](MemberCreateOp member) {
			members[member.name()] = member;
		});

		mlir::OpBuilder builder(op);

		// Replace the values directly with the return operands
		assert(functionOp.resultsNames().size() == valuesToReplace.size());

		auto newValueFn = [](mlir::OpBuilder& builder, mlir::Value member) -> mlir::Value {
			auto memberType = member.getType().cast<MemberType>();

			if (memberType.getShape().empty())
				return builder.create<MemberLoadOp>(member.getLoc(), memberType.getElementType(), member);

			return builder.create<MemberLoadOp>(member.getLoc(), memberType.toArrayType(), member);
		};

		for (const auto& [name, toBeReplaced] : llvm::zip(functionOp.resultsNames(), valuesToReplace))
		{
			mlir::Value member = members[name.cast<mlir::StringAttr>().getValue()];
			toBeReplaced.replaceAllUsesWith(newValueFn(builder, member));
		}
	}
};

ModelicaDialect::ModelicaDialect(mlir::MLIRContext* context)
		: Dialect("modelica", context, mlir::TypeID::get<ModelicaDialect>())
{
	addTypes<BooleanType, IntegerType, RealType, MemberType, ArrayType, UnsizedArrayType, StructType>();
	addAttributes<BooleanAttribute, IntegerAttribute, RealAttribute, DerivativeAttribute, InverseFunctionsAttribute>();
	addInterfaces<ModelicaInlinerInterface>();

	addOperations<FunctionOp, FunctionTerminatorOp, DerFunctionOp>();

	// Basic operations
	addOperations<ConstantOp, PackOp, ExtractOp, CastOp, AssignmentOp, CallOp>();

	// Memory operations
	addOperations<MemberCreateOp, MemberLoadOp, MemberStoreOp>();
	addOperations<AllocaOp, AllocOp, FreeOp>();
	addOperations<ArrayCastOp, DimOp, SubscriptionOp>();
	addOperations<LoadOp, StoreOp>();
	addOperations<ArrayCloneOp>();

	// Math operations
	addOperations<
	    NegateOp,
			AddOp, AddElementWiseOp,
			SubOp, SubElementWiseOp,
			MulOp, MulElementWiseOp,
			DivOp, DivElementWiseOp,
			PowOp, PowElementWiseOp>();

	// Logic operations
	addOperations<NotOp, AndOp, OrOp>();
	addOperations<EqOp>();
	addOperations<NotEqOp>();
	addOperations<GtOp>();
	addOperations<GteOp>();
	addOperations<LtOp>();
	addOperations<LteOp>();

	// Control flow operations
	addOperations<ForOp, IfOp, WhileOp>();
	addOperations<ConditionOp, YieldOp, BreakOp, ReturnOp>();

	// Built-in operations
	addOperations<
	    AbsOp,
			SignOp,
			SqrtOp,
			SinOp, CosOp, TanOp,
			AsinOp, AcosOp, AtanOp, Atan2Op,
			SinhOp, CoshOp, TanhOp,
			ExpOp, LogOp, Log10Op,
			NDimsOp, SizeOp,
			IdentityOp, DiagonalOp,
			ZerosOp, OnesOp,
			LinspaceOp,
			FillOp,
			MinOp, MaxOp,
			SumOp, ProductOp,
			TransposeOp,
			SymmetricOp>();

	addOperations<ModelOp, EquationOp, ForEquationOp, EquationSidesOp, DerOp, DerSeedOp>();
	addOperations<PrintOp>();
}

mlir::StringRef ModelicaDialect::getDialectNamespace()
{
	return "modelica";
}

mlir::Type ModelicaDialect::parseType(mlir::DialectAsmParser& parser) const
{
	return parseModelicaType(parser);
}

void ModelicaDialect::printType(mlir::Type type, mlir::DialectAsmPrinter& printer) const
{
	return printModelicaType(type, printer);
}

mlir::Attribute ModelicaDialect::parseAttribute(mlir::DialectAsmParser& parser, mlir::Type type) const
{
	return parseModelicaAttribute(parser, type);
}

void ModelicaDialect::printAttribute(mlir::Attribute attribute, mlir::DialectAsmPrinter& printer) const
{
	return printModelicaAttribute(attribute, printer);
}

mlir::Operation* ModelicaDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value, mlir::Type type, mlir::Location loc)
{
	return builder.create<ConstantOp>(loc, value);
}
