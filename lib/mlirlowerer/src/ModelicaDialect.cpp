#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Transforms/InliningUtils.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>

using namespace modelica::codegen;

class ModelicaInlinerInterface : public mlir::DialectInlinerInterface
{
	public:
	using mlir::DialectInlinerInterface::DialectInlinerInterface;

	bool isLegalToInline(mlir::Operation* call, mlir::Operation* callable, bool wouldBeCloned) const final
	{
		auto function = mlir::cast<mlir::FuncOp>(callable);

		if (!function->hasAttr("inline"))
			return false;

		auto inlineAttribute = function->getAttrOfType<BooleanAttribute>("inline");
		return inlineAttribute.getValue();
	}

	bool isLegalToInline(mlir::Operation* op, mlir::Region* dest, bool wouldBeCloned, mlir::BlockAndValueMapping &valueMapping) const final
	{
		return true;
	}

	void handleTerminator(mlir::Operation* op, llvm::ArrayRef<mlir::Value> valuesToReplace) const final {
		auto returnOp = mlir::cast<mlir::ReturnOp>(op);

		// Replace the values directly with the return operands
		assert(returnOp.getNumOperands() == valuesToReplace.size());

		for (const auto& it : llvm::enumerate(returnOp.getOperands()))
			valuesToReplace[it.index()].replaceAllUsesWith(it.value());
	}
};

ModelicaDialect::ModelicaDialect(mlir::MLIRContext* context)
		: Dialect("modelica", context, mlir::TypeID::get<ModelicaDialect>())
{
	addTypes<BooleanType, IntegerType, RealType, PointerType, UnsizedPointerType, OpaquePointerType, StructType>();
	addAttributes<BooleanAttribute, IntegerAttribute, RealAttribute, InverseFunctionsAttribute>();
	addInterfaces<ModelicaInlinerInterface>();

	// Basic operations
	addOperations<ConstantOp, PackOp, ExtractOp, CastOp, CastCommonOp, AssignmentOp, CallOp, PrintOp>();

	// Memory operations
	addOperations<AllocaOp, AllocOp, FreeOp>();
	addOperations<PtrCastOp, DimOp, SubscriptionOp>();
	addOperations<LoadOp, StoreOp>();
	addOperations<ArrayCloneOp>();

	// Math operations
	addOperations<NegateOp, AddOp, SubOp, MulOp, DivOp, PowOp>();

	// Logic operations
	addOperations<NotOp, AndOp, OrOp>();
	addOperations<EqOp>();
	addOperations<NotEqOp>();
	addOperations<GtOp>();
	addOperations<GteOp>();
	addOperations<LtOp>();
	addOperations<LteOp>();

	// Control flow operations
	addOperations<IfOp>();
	addOperations<ForOp>();
	addOperations<BreakableForOp>();
	addOperations<BreakableWhileOp>();
	addOperations<ConditionOp>();
	addOperations<YieldOp>();

	// Built-in operations
	addOperations<NDimsOp, SizeOp, IdentityOp, FillOp>();

	addOperations<SimulationOp, EquationOp, InductionOp, ForEquationOp, EquationSidesOp, DerOp>();
}

mlir::StringRef ModelicaDialect::getDialectNamespace()
{
	return "modelica";
}

void ModelicaDialect::printType(mlir::Type type, mlir::DialectAsmPrinter& printer) const {
	return printModelicaType(type, printer);
}

void ModelicaDialect::printAttribute(mlir::Attribute attribute, mlir::DialectAsmPrinter& printer) const {
	return printModelicaAttribute(attribute, printer);
}

mlir::Operation* ModelicaDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value, mlir::Type type, mlir::Location loc)
{
	return builder.create<ConstantOp>(loc, value);
}
