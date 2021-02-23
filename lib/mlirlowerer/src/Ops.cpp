#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <modelica/mlirlowerer/Ops.h>

using namespace modelica;

//===----------------------------------------------------------------------===//
// Modelica::AllocaOp
//===----------------------------------------------------------------------===//

llvm::StringRef AllocaOp::getOperationName()
{
	return "modelica.alloca";
}

void AllocaOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type elementType, PointerType::Shape shape, mlir::ValueRange dimensions)
{
	size_t dynamicDimensions = 0;

	for (long dimension : shape)
		if (dimension == -1)
			dynamicDimensions++;

	assert(dynamicDimensions == dimensions.size() && "Dynamic dimensions amount doesn't match with the number of provided values");

	if (shape.empty())
		state.addTypes(PointerType::get(state.getContext(), elementType));
	else
		state.addTypes(PointerType::get(state.getContext(), elementType, shape));

	state.addOperands(dimensions);
}

void AllocaOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.alloca ";
	printer.printOperands(getOperands());
	printer << ": ";
	printer.printType(getOperation()->getResultTypes()[0]);
}

PointerType AllocaOp::getPointerType()
{
	return getOperation()->getResultTypes()[0].cast<PointerType>();
}

//===----------------------------------------------------------------------===//
// Modelica::AllocOp
//===----------------------------------------------------------------------===//

llvm::StringRef AllocOp::getOperationName()
{
	return "modelica.alloc";
}

void AllocOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type elementType, PointerType::Shape shape, mlir::ValueRange dimensions)
{
	size_t dynamicDimensions = 0;

	for (long dimension : shape)
		if (dimension == -1)
			dynamicDimensions++;

	assert(dynamicDimensions == dimensions.size() && "Dynamic dimensions amount doesn't match with the number of provided values");

	if (shape.empty())
		state.addTypes(PointerType::get(state.getContext(), elementType));
	else
		state.addTypes(PointerType::get(state.getContext(), elementType, shape));

	state.addOperands(dimensions);
}

void AllocOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.alloca ";
	printer.printOperands(getOperands());
	printer << ": ";
	printer.printType(getOperation()->getResultTypes()[0]);
}

PointerType AllocOp::getPointerType()
{
	return getOperation()->getResultTypes()[0].cast<PointerType>();
}

//===----------------------------------------------------------------------===//
// Modelica::FreeOp
//===----------------------------------------------------------------------===//

FreeOpAdaptor::FreeOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs)
		: values(values), attrs(attrs)
{
}

FreeOpAdaptor::FreeOpAdaptor(FreeOp& op)
		: values(op->getOperands()), attrs(op->getAttrDictionary())
{
}

mlir::Value FreeOpAdaptor::memory()
{
	return values[0];
}

llvm::StringRef FreeOp::getOperationName()
{
	return "modelica.free";
}

void FreeOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value memory)
{
	state.addOperands(memory);
}

void FreeOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.free " << memory();
}

mlir::Value FreeOp::memory()
{
	return Adaptor(*this).memory();
}

//===----------------------------------------------------------------------===//
// Modelica::DimOp
//===----------------------------------------------------------------------===//

DimOpAdaptor::DimOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs)
		: values(values), attrs(attrs)
{
}

DimOpAdaptor::DimOpAdaptor(DimOp& op)
		: values(op->getOperands()), attrs(op->getAttrDictionary())
{
}

mlir::Value DimOpAdaptor::memory()
{
	return values[0];
}

mlir::Value DimOpAdaptor::dimension()
{
	return values[1];
}

llvm::StringRef DimOp::getOperationName()
{
	return "modelica.dim";
}

void DimOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value memory, mlir::Value dimension)
{
	state.addOperands({ memory, dimension });
	state.addTypes(builder.getIndexType());
}

void DimOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.dim " << getOperands() << " : " << memory().getType();
}

PointerType DimOp::getPointerType()
{
	return memory().getType().cast<PointerType>();
}

mlir::Value DimOp::memory()
{
	return Adaptor(*this).memory();
}

mlir::Value DimOp::dimension()
{
	return Adaptor(*this).dimension();
}

//===----------------------------------------------------------------------===//
// Modelica::LoadOp
//===----------------------------------------------------------------------===//

LoadOpAdaptor::LoadOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs)
		: values(values), attrs(attrs)
{
}

LoadOpAdaptor::LoadOpAdaptor(LoadOp& op)
		: values(op->getOperands()), attrs(op->getAttrDictionary())
{
}

mlir::Value LoadOpAdaptor::memory()
{
	return values[0];
}

mlir::ValueRange LoadOpAdaptor::indexes()
{
	return mlir::ValueRange(std::next(values.begin(), 1), values.end());
}

llvm::StringRef LoadOp::getOperationName()
{
	return "modelica.load";
}

void LoadOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value memory, mlir::ValueRange indexes)
{
	state.addOperands(memory);
	state.addOperands(indexes);
	state.addTypes(memory.getType().cast<PointerType>().getElementType());
}

void LoadOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.load " << memory() << "[";
	printer.printOperands(indexes());
	printer << "] : ";
	printer.printType(memory().getType());
}

PointerType LoadOp::getPointerType()
{
	return memory().getType().cast<PointerType>();
}

mlir::Value LoadOp::memory()
{
	return Adaptor(*this).memory();
}

mlir::ValueRange LoadOp::indexes()
{
	return Adaptor(*this).indexes();
}

//===----------------------------------------------------------------------===//
// Modelica::StoreOp
//===----------------------------------------------------------------------===//

StoreOpAdaptor::StoreOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs)
		: values(values), attrs(attrs)
{
}

StoreOpAdaptor::StoreOpAdaptor(StoreOp& op)
		: values(op->getOperands()), attrs(op->getAttrDictionary())
{
}

mlir::Value StoreOpAdaptor::value()
{
	return values[0];
}

mlir::Value StoreOpAdaptor::memory()
{
	return values[1];
}

mlir::ValueRange StoreOpAdaptor::indexes()
{
	return mlir::ValueRange(std::next(values.begin(), 2), values.end());
}

llvm::StringRef StoreOp::getOperationName()
{
	return "modelica.store";
}

void StoreOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value value, mlir::Value memory, mlir::ValueRange indexes)
{
	state.addOperands(value);
	state.addOperands(memory);
	state.addOperands(indexes);
}

void StoreOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.store " << value() << ", " << memory() << "[";
	printer.printOperands(indexes());
	printer << "] : ";
	printer.printType(memory().getType());
}

PointerType StoreOp::getPointerType()
{
	return memory().getType().cast<PointerType>();
}

mlir::Value StoreOp::value()
{
	return Adaptor(*this).value();
}

mlir::Value StoreOp::memory()
{
	return Adaptor(*this).memory();
}

mlir::ValueRange StoreOp::indexes()
{
	return Adaptor(*this).indexes();
}

//===----------------------------------------------------------------------===//
// Modelica::IfOp
//===----------------------------------------------------------------------===//

IfOpAdaptor::IfOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs)
		: values(values), attrs(attrs)
{
}

IfOpAdaptor::IfOpAdaptor(IfOp& op)
		: values(op->getOperands()), attrs(op->getAttrDictionary())
{
}

mlir::Value IfOpAdaptor::condition()
{
	return values[0];
}

llvm::StringRef IfOp::getOperationName()
{
	return "modelica.if";
}

void IfOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value condition, bool withElseRegion)
{
	state.addOperands(condition);
	auto insertionPoint = builder.saveInsertionPoint();

	// "Then" region
	auto* thenRegion = state.addRegion();
	builder.createBlock(thenRegion);

	// "Else" region
	auto* elseRegion = state.addRegion();

	if (withElseRegion)
		builder.createBlock(elseRegion);

	builder.restoreInsertionPoint(insertionPoint);
}

void IfOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.if " << condition();
	printer.printRegion(thenRegion());

	if (!elseRegion().empty())
	{
		printer << " else";
		printer.printRegion(elseRegion());
	}
}

mlir::Value IfOp::condition()
{
	return Adaptor(*this).condition();
}

mlir::Region& IfOp::thenRegion()
{
	return getRegion(0);
}

mlir::Region& IfOp::elseRegion()
{
	return getRegion(1);
}

//===----------------------------------------------------------------------===//
// Modelica::ForOp
//===----------------------------------------------------------------------===//

ForOpAdaptor::ForOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs)
		: values(values), attrs(attrs)
{
}

ForOpAdaptor::ForOpAdaptor(ForOp& op)
		: values(op->getOperands()), attrs(op->getAttrDictionary())
{
}

mlir::Value ForOpAdaptor::breakCondition()
{
	return values[0];
}

mlir::Value ForOpAdaptor::returnCondition()
{
	return values[1];
}

mlir::ValueRange ForOpAdaptor::args()
{
	return mlir::ValueRange(std::next(values.begin(), 2), values.end());
}

llvm::StringRef ForOp::getOperationName()
{
	return "modelica.for";
}

void ForOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value breakCondition, mlir::Value returnCondition)
{
	build(builder, state, breakCondition, returnCondition, {});
}

void ForOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value breakCondition, mlir::Value returnCondition, mlir::ValueRange args)
{
	state.addOperands(breakCondition);
	state.addOperands(returnCondition);
	state.addOperands(args);

	auto insertionPoint = builder.saveInsertionPoint();

	// Condition block
	builder.createBlock(state.addRegion(), {}, args.getTypes());

	// Step block
	builder.createBlock(state.addRegion(), {}, args.getTypes());

	// Body block
	builder.createBlock(state.addRegion(), {}, args.getTypes());

	builder.restoreInsertionPoint(insertionPoint);
}

void ForOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.for (break on " << breakCondition() << ", return on " << returnCondition() << ") " << args();
	printer << " condition";
	printer.printRegion(condition(), true);
	printer << " body";
	printer.printRegion(body(), true);
	printer << " step";
	printer.printRegion(step(), true);
}

mlir::Region& ForOp::condition()
{
	return getOperation()->getRegion(0);
}

mlir::Region& ForOp::step()
{
	return getOperation()->getRegion(1);
}

mlir::Region& ForOp::body()
{
	return getOperation()->getRegion(2);
}

mlir::Value ForOp::breakCondition()
{
	return Adaptor(*this).breakCondition();
}

mlir::Value ForOp::returnCondition()
{
	return Adaptor(*this).returnCondition();
}

mlir::ValueRange ForOp::args()
{
	return Adaptor(*this).args();
}

//===----------------------------------------------------------------------===//
// Modelica::WhileOp
//===----------------------------------------------------------------------===//

WhileOpAdaptor::WhileOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs)
		: values(values), attrs(attrs)
{
}

WhileOpAdaptor::WhileOpAdaptor(WhileOp& op)
		: values(op->getOperands()), attrs(op->getAttrDictionary())
{
}

mlir::Value WhileOpAdaptor::breakCondition()
{
	return values[0];
}

mlir::Value WhileOpAdaptor::returnCondition()
{
	return values[1];
}

llvm::StringRef WhileOp::getOperationName()
{
	return "modelica.while";
}

void WhileOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value breakCondition, mlir::Value returnCondition)
{
	state.addOperands(breakCondition);
	state.addOperands(returnCondition);

	auto insertionPoint = builder.saveInsertionPoint();

	// Condition block
	builder.createBlock(state.addRegion());

	// Body block
	builder.createBlock(state.addRegion());

	builder.restoreInsertionPoint(insertionPoint);
}

void WhileOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.while (break on " << breakCondition() << ", return on " << returnCondition() << ")";
	printer.printRegion(condition(), false);
	printer << " do";
	printer.printRegion(body(), false);
}

mlir::Region& WhileOp::condition()
{
	return getOperation()->getRegion(0);
}

mlir::Region& WhileOp::body()
{
	return getOperation()->getRegion(1);
}

mlir::Value WhileOp::breakCondition()
{
	return Adaptor(*this).breakCondition();
}

mlir::Value WhileOp::returnCondition()
{
	return Adaptor(*this).returnCondition();
}

//===----------------------------------------------------------------------===//
// Modelica::ConditionOp
//===----------------------------------------------------------------------===//

ConditionOpAdaptor::ConditionOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs)
		: values(values), attrs(attrs)
{
}

ConditionOpAdaptor::ConditionOpAdaptor(ConditionOp& op)
		: values(op->getOperands()), attrs(op->getAttrDictionary())
{
}

mlir::Value ConditionOpAdaptor::condition()
{
	return values[0];
}

mlir::ValueRange ConditionOpAdaptor::args()
{
	return mlir::ValueRange(std::next(values.begin()), values.end());
}

llvm::StringRef ConditionOp::getOperationName()
{
	return "modelica.condition";
}

void ConditionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value condition, mlir::ValueRange args)
{
	state.addOperands(condition);
	state.addOperands(args);
}

void ConditionOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.condition (" << condition() << ") " << args();
}

mlir::Value ConditionOp::condition()
{
	return Adaptor(*this).condition();
}

mlir::ValueRange ConditionOp::args()
{
	return Adaptor(*this).args();
}

//===----------------------------------------------------------------------===//
// Modelica::YieldOp
//===----------------------------------------------------------------------===//

YieldOpAdaptor::YieldOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs)
		: values(values), attrs(attrs)
{
}

YieldOpAdaptor::YieldOpAdaptor(YieldOp& op)
		: values(op->getOperands()), attrs(op->getAttrDictionary())
{
}

mlir::ValueRange YieldOpAdaptor::args()
{
	return values;
}

llvm::StringRef YieldOp::getOperationName()
{
	return "modelica.yield";
}

void YieldOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange operands)
{
	state.addOperands(operands);
}

void YieldOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.yield " << getOperands();
}

mlir::ValueRange YieldOp::args()
{
	return Adaptor(*this).args();
}

//===----------------------------------------------------------------------===//
// Modelica::CastOp
//===----------------------------------------------------------------------===//

CastOpAdaptor::CastOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs)
		: values(values), attrs(attrs)
{
}

CastOpAdaptor::CastOpAdaptor(CastOp& op)
		: values(op->getOperands()), attrs(op->getAttrDictionary())
{
}

mlir::Value CastOpAdaptor::value()
{
	return values[0];
}

llvm::StringRef CastOp::getOperationName()
{
	return "modelica.cast";
}

void CastOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value value, mlir::Type resultType)
{
	state.addOperands(value);
	state.addTypes(resultType);
}

void CastOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.cast " << value() << " : " << resultType();
}

mlir::Value CastOp::value()
{
	return Adaptor(*this).value();
}

mlir::Type CastOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

//===----------------------------------------------------------------------===//
// Modelica::CastCommonOp
//===----------------------------------------------------------------------===//

CastCommonOpAdaptor::CastCommonOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs)
		: values(values), attrs(attrs)
{
}

CastCommonOpAdaptor::CastCommonOpAdaptor(CastCommonOp& op)
		: values(op->getOperands()), attrs(op->getAttrDictionary())
{
}

mlir::ValueRange CastCommonOpAdaptor::operands()
{
	return values;
}

static mlir::Type getMoreGenericType(mlir::Type x, mlir::Type y)
{
	mlir::Type xBase = x;
	mlir::Type yBase = y;

	while (xBase.isa<PointerType>())
		xBase = xBase.cast<PointerType>().getElementType();

	while (yBase.isa<PointerType>())
		yBase = yBase.cast<PointerType>().getElementType();

	if (xBase.isa<RealType>())
		return x;

	if (yBase.isa<RealType>())
		return y;

	if (xBase.isa<IntegerType>())
		return x;

	if (yBase.isa<IntegerType>())
		return y;

	return x;
}

llvm::StringRef CastCommonOp::getOperationName()
{
	return "modelica.cast_common";
}

void CastCommonOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange values)
{
	state.addOperands(values);

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

		if (resultBaseType.isa<IntegerType>())
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
			types.emplace_back(PointerType::get(pointerType.getContext(), resultBaseType, shape));
		}
		else
			types.emplace_back(resultBaseType);
	}

	state.addTypes(types);
}

void CastCommonOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.cast_common " << operands() << " : " << resultType();
}

mlir::Type CastCommonOp::resultType()
{
	return getResultTypes()[0];
}

mlir::ValueRange CastCommonOp::operands()
{
	return Adaptor(*this).operands();
}

//===----------------------------------------------------------------------===//
// Modelica::NegateOp
//===----------------------------------------------------------------------===//

NegateOpAdaptor::NegateOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs)
		: values(values), attrs(attrs)
{
}

NegateOpAdaptor::NegateOpAdaptor(NegateOp& op)
		: values(op->getOperands()), attrs(op->getAttrDictionary())
{
}

mlir::Value NegateOpAdaptor::operand()
{
	return values[0];
}

llvm::StringRef NegateOp::getOperationName()
{
	return "modelica.negate";
}

void NegateOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value operand)
{
	state.addOperands(operand);
	state.addTypes(operand.getType());
}

void NegateOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.neg " << getOperand() << " : " << getOperation()->getResultTypes();
}

mlir::Value NegateOp::operand()
{
	return Adaptor(*this).operand();
}

//===----------------------------------------------------------------------===//
// Modelica::EqOp
//===----------------------------------------------------------------------===//

EqOpAdaptor::EqOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs)
		: values(values), attrs(attrs)
{
}

EqOpAdaptor::EqOpAdaptor(EqOp& op)
		: values(op->getOperands()), attrs(op->getAttrDictionary())
{
}

mlir::Value EqOpAdaptor::lhs()
{
	return values[0];
}

mlir::Value EqOpAdaptor::rhs()
{
	return values[1];
}

llvm::StringRef EqOp::getOperationName()
{
	return "modelica.eq";
}

void EqOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(builder.getI1Type());
	state.addOperands({ lhs, rhs });
}

void EqOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::LogicalResult EqOp::verify()
{
	for (auto operand : getOperands())
		if (operand.getType().isa<mlir::ShapedType>())
			return emitOpError("Comparison operation are only defined for scalar operands of simple types");

	return mlir::success();
}

void EqOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.eq " << getOperands() << " : " << getOperation()->getResultTypes()[0];
}

mlir::Value EqOp::lhs()
{
	return Adaptor(*this).lhs();
}

mlir::Value EqOp::rhs()
{
	return Adaptor(*this).rhs();
}

//===----------------------------------------------------------------------===//
// Modelica::NotEqOp
//===----------------------------------------------------------------------===//

NotEqOpAdaptor::NotEqOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs)
		: values(values), attrs(attrs)
{
}

NotEqOpAdaptor::NotEqOpAdaptor(NotEqOp& op)
		: values(op->getOperands()), attrs(op->getAttrDictionary())
{
}

mlir::Value NotEqOpAdaptor::lhs()
{
	return values[0];
}

mlir::Value NotEqOpAdaptor::rhs()
{
	return values[1];
}

llvm::StringRef NotEqOp::getOperationName()
{
	return "modelica.neq";
}

void NotEqOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(builder.getI1Type());
	state.addOperands({ lhs, rhs });
}

void NotEqOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::LogicalResult NotEqOp::verify()
{
	for (auto operand : getOperands())
		if (operand.getType().isa<mlir::ShapedType>())
			return emitOpError("Comparison operation are only defined for scalar operands of simple types");

	return mlir::success();
}

void NotEqOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.neq " << getOperands() << " : " << getOperation()->getResultTypes()[0];
}

mlir::Value NotEqOp::lhs()
{
	return Adaptor(*this).lhs();
}

mlir::Value NotEqOp::rhs()
{
	return Adaptor(*this).rhs();
}

//===----------------------------------------------------------------------===//
// Modelica::GtOp
//===----------------------------------------------------------------------===//

GtOpAdaptor::GtOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs)
		: values(values), attrs(attrs)
{
}

GtOpAdaptor::GtOpAdaptor(GtOp& op)
		: values(op->getOperands()), attrs(op->getAttrDictionary())
{
}

mlir::Value GtOpAdaptor::lhs()
{
	return values[0];
}

mlir::Value GtOpAdaptor::rhs()
{
	return values[1];
}

llvm::StringRef GtOp::getOperationName()
{
	return "modelica.gt";
}

void GtOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(builder.getI1Type());
	state.addOperands({ lhs, rhs });
}

void GtOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::LogicalResult GtOp::verify()
{
	for (auto operand : getOperands())
		if (operand.getType().isa<mlir::ShapedType>())
			return emitOpError("Comparison operation are only defined for scalar operands of simple types");

	return mlir::success();
}

void GtOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.gt " << getOperands() << " : " << getOperation()->getResultTypes()[0];
}

mlir::Value GtOp::lhs()
{
	return Adaptor(*this).lhs();
}

mlir::Value GtOp::rhs()
{
	return Adaptor(*this).rhs();
}

//===----------------------------------------------------------------------===//
// Modelica::GteOp
//===----------------------------------------------------------------------===//

GteOpAdaptor::GteOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs)
		: values(values), attrs(attrs)
{
}

GteOpAdaptor::GteOpAdaptor(GteOp& op)
		: values(op->getOperands()), attrs(op->getAttrDictionary())
{
}

mlir::Value GteOpAdaptor::lhs()
{
	return values[0];
}

mlir::Value GteOpAdaptor::rhs()
{
	return values[1];
}

llvm::StringRef GteOp::getOperationName()
{
	return "modelica.gte";
}

void GteOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(builder.getI1Type());
	state.addOperands({ lhs, rhs });
}

void GteOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::LogicalResult GteOp::verify()
{
	for (auto operand : getOperands())
		if (operand.getType().isa<mlir::ShapedType>())
			return emitOpError("Comparison operation are only defined for scalar operands of simple types");

	return mlir::success();
}

void GteOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.gte " << getOperands() << " : " << getOperation()->getResultTypes()[0];
}

mlir::Value GteOp::lhs()
{
	return Adaptor(*this).lhs();
}

mlir::Value GteOp::rhs()
{
	return Adaptor(*this).rhs();
}

//===----------------------------------------------------------------------===//
// Modelica::LtOp
//===----------------------------------------------------------------------===//

LtOpAdaptor::LtOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs)
		: values(values), attrs(attrs)
{
}

LtOpAdaptor::LtOpAdaptor(LtOp& op)
		: values(op->getOperands()), attrs(op->getAttrDictionary())
{
}

mlir::Value LtOpAdaptor::lhs()
{
	return values[0];
}

mlir::Value LtOpAdaptor::rhs()
{
	return values[1];
}

llvm::StringRef LtOp::getOperationName()
{
	return "modelica.lt";
}

void LtOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(builder.getI1Type());
	state.addOperands({ lhs, rhs });
}

void LtOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::LogicalResult LtOp::verify()
{
	for (auto operand : getOperands())
		if (operand.getType().isa<mlir::ShapedType>())
			return emitOpError("Comparison operation are only defined for scalar operands of simple types");

	return mlir::success();
}

void LtOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.lt " << getOperands() << " : " << getOperation()->getResultTypes()[0];
}

mlir::Value LtOp::lhs()
{
	return Adaptor(*this).lhs();
}

mlir::Value LtOp::rhs()
{
	return Adaptor(*this).rhs();
}

//===----------------------------------------------------------------------===//
// Modelica::LteOp
//===----------------------------------------------------------------------===//

LteOpAdaptor::LteOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs)
		: values(values), attrs(attrs)
{
}

LteOpAdaptor::LteOpAdaptor(LteOp& op)
		: values(op->getOperands()), attrs(op->getAttrDictionary())
{
}

mlir::Value LteOpAdaptor::lhs()
{
	return values[0];
}

mlir::Value LteOpAdaptor::rhs()
{
	return values[1];
}

llvm::StringRef LteOp::getOperationName()
{
	return "modelica.lte";
}

void LteOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(builder.getI1Type());
	state.addOperands({ lhs, rhs });
}

void LteOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::LogicalResult LteOp::verify()
{
	for (auto operand : getOperands())
		if (operand.getType().isa<mlir::ShapedType>())
			return emitOpError("Comparison operation are only defined for scalar operands of simple types");

	return mlir::success();
}

void LteOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.lte " << getOperands() << " : " << getOperation()->getResultTypes()[0];
}

mlir::Value LteOp::lhs()
{
	return Adaptor(*this).lhs();
}

mlir::Value LteOp::rhs()
{
	return Adaptor(*this).rhs();
}

//===----------------------------------------------------------------------===//
// Modelica::AddOp
//===----------------------------------------------------------------------===//

AddOpAdaptor::AddOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs)
		: values(values), attrs(attrs)
{
}

AddOpAdaptor::AddOpAdaptor(AddOp& op)
		: values(op->getOperands()), attrs(op->getAttrDictionary())
{
}

mlir::Value AddOpAdaptor::lhs()
{
	return values[0];
}

mlir::Value AddOpAdaptor::rhs()
{
	return values[1];
}

llvm::StringRef AddOp::getOperationName()
{
	return "modelica.add";
}

void AddOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

void AddOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.add " << lhs() << ", " << rhs() << " : " << resultType();
}

mlir::Type AddOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value AddOp::lhs()
{
	return Adaptor(*this).lhs();
}

mlir::Value AddOp::rhs()
{
	return Adaptor(*this).rhs();
}
