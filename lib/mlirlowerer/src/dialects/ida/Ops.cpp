#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <marco/mlirlowerer/dialects/ida/Attribute.h>
#include <marco/mlirlowerer/dialects/ida/Ops.h>
#include <marco/mlirlowerer/dialects/modelica/Type.h>

using namespace marco::codegen::ida;

namespace marco::codegen::ida
{
	static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result, int operandsNum)
	{
		llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> operands;
		llvm::SmallVector<mlir::Type, 3> operandTypes;
		llvm::SmallVector<mlir::Type, 1> resultTypes;

		llvm::SMLoc operandsLoc = parser.getCurrentLocation();

		if (parser.parseOperandList(operands, operandsNum) ||
				parser.parseColon() ||
				parser.parseLParen() ||
				parser.parseTypeList(operandTypes) ||
				parser.parseRParen() ||
				parser.resolveOperands(operands, operandTypes, operandsLoc, result.operands) ||
				parser.parseOptionalArrowTypeList(resultTypes))
			return mlir::failure();

		result.addTypes(resultTypes);

		return mlir::success();
	}

	static void print(mlir::OpAsmPrinter& printer, llvm::StringLiteral opName, mlir::ValueRange values)
	{
		printer << opName << " " << values[0];

		for (size_t i = 1; i < values.size(); i++)
			printer << ", " << values[i];

		printer << " : (" << values[0].getType();

		for (size_t i = 1; i < values.size(); i++)
			printer << ", " << values[i].getType();
		
		printer << ")";
	}

	static void print(mlir::OpAsmPrinter& printer, llvm::StringLiteral opName, mlir::ValueRange values, mlir::Type resultType)
	{
		print(printer, opName, values);
		printer << " -> " << resultType;
	}
}

//===----------------------------------------------------------------------===//
// Ida::ConstantValueOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> ConstantValueOp::getAttributeNames()
{
	static llvm::StringRef attrNames[] = {llvm::StringRef("value")};
	return llvm::makeArrayRef(attrNames);
}

void ConstantValueOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Attribute attribute)
{
	state.addAttribute("value", attribute);
	state.addTypes(attribute.getType());
}

mlir::ParseResult ConstantValueOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::Attribute value;

	if (parser.parseAttribute(value))
		return mlir::failure();

	result.attributes.append("value", value);
	result.addTypes(value.getType());
	return mlir::success();
}

void ConstantValueOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << value();
}

mlir::OpFoldResult ConstantValueOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
{
	assert(operands.empty() && "constant has no operands");
	return value();
}

mlir::Attribute ConstantValueOp::value()
{
	return getOperation()->getAttr("value");
}

mlir::Type ConstantValueOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

//===----------------------------------------------------------------------===//
// Ida::AllocUserDataOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> AllocUserDataOp::getAttributeNames()
{
	return {};
}

void AllocUserDataOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value equationsNumber)
{
	state.addTypes(OpaquePointerType::get(builder.getContext()));
	state.addOperands(equationsNumber);
}

mlir::ParseResult AllocUserDataOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 1);
}

void AllocUserDataOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult AllocUserDataOp::verify()
{
	if (!equationsNumber().getType().isa<IntegerType>())
		return emitOpError("Requires number of equations to be an integer");

	return mlir::success();
}

void AllocUserDataOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());
}

OpaquePointerType AllocUserDataOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<OpaquePointerType>();
}

mlir::ValueRange AllocUserDataOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value AllocUserDataOp::equationsNumber()
{
	return getOperation()->getOperand(0);
}

//===----------------------------------------------------------------------===//
// Ida::FreeUserDataOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> FreeUserDataOp::getAttributeNames()
{
	return {};
}

void FreeUserDataOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData)
{
	state.addTypes(BooleanType::get(builder.getContext()));
	state.addOperands(userData);
}

mlir::ParseResult FreeUserDataOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 1);
}

void FreeUserDataOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult FreeUserDataOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	return mlir::success();
}

void FreeUserDataOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Free::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

BooleanType FreeUserDataOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<BooleanType>();
}

mlir::ValueRange FreeUserDataOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value FreeUserDataOp::userData()
{
	return getOperation()->getOperand(0);
}

//===----------------------------------------------------------------------===//
// Ida::SetInitialValueOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> SetInitialValueOp::getAttributeNames()
{
	return {};
}

void SetInitialValueOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value index, mlir::Value length, mlir::Value value, mlir::Value isState)
{
	state.addOperands(userData);
	state.addOperands(index);
	state.addOperands(length);
	state.addOperands(value);
	state.addOperands(isState);
}

mlir::ParseResult SetInitialValueOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 5);
}

void SetInitialValueOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args());
}

mlir::LogicalResult SetInitialValueOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!index().getType().isa<IntegerType>())
		return emitOpError("Requires variable index to be an integer");

	if (!length().getType().isa<IntegerType>())
		return emitOpError("Requires variable size to be an integer");

	if (!value().getType().isa<modelica::RealType>() && !value().getType().isa<modelica::IntegerType>())
		return emitOpError("Requires initialization value to be a number");

	if (!isState().getType().isa<BooleanType>())
		return emitOpError("Requires variable state to be a boolean");

	return mlir::success();
}

void SetInitialValueOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

mlir::ValueRange SetInitialValueOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value SetInitialValueOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value SetInitialValueOp::index()
{
	return getOperation()->getOperand(1);
}

mlir::Value SetInitialValueOp::length()
{
	return getOperation()->getOperand(2);
}

mlir::Value SetInitialValueOp::value()
{
	return getOperation()->getOperand(3);
}

mlir::Value SetInitialValueOp::isState()
{
	return getOperation()->getOperand(4);
}

//===----------------------------------------------------------------------===//
// Ida::SetInitialArrayOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> SetInitialArrayOp::getAttributeNames()
{
	return {};
}

void SetInitialArrayOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value index, mlir::Value length, mlir::Value array, mlir::Value isState)
{
	state.addOperands(userData);
	state.addOperands(index);
	state.addOperands(length);
	state.addOperands(array);
	state.addOperands(isState);
}

mlir::ParseResult SetInitialArrayOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 5);
}

void SetInitialArrayOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args());
}

mlir::LogicalResult SetInitialArrayOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!index().getType().isa<IntegerType>())
		return emitOpError("Requires variable index to be an integer");

	if (!length().getType().isa<IntegerType>())
		return emitOpError("Requires variable size to be an integer");

	if (!array().getType().isa<modelica::ArrayType>())
		return emitOpError("Requires initialization array to be an array");

	if (!isState().getType().isa<BooleanType>())
		return emitOpError("Requires variable state to be a boolean");

	return mlir::success();
}

void SetInitialArrayOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

mlir::ValueRange SetInitialArrayOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value SetInitialArrayOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value SetInitialArrayOp::index()
{
	return getOperation()->getOperand(1);
}

mlir::Value SetInitialArrayOp::length()
{
	return getOperation()->getOperand(2);
}

mlir::Value SetInitialArrayOp::array()
{
	return getOperation()->getOperand(3);
}

mlir::Value SetInitialArrayOp::isState()
{
	return getOperation()->getOperand(4);
}

//===----------------------------------------------------------------------===//
// Ida::InitOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> InitOp::getAttributeNames()
{
	return {};
}

void InitOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData)
{
	state.addTypes(BooleanType::get(builder.getContext()));
	state.addOperands(userData);
}

mlir::ParseResult InitOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 1);
}

void InitOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult InitOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	return mlir::success();
}

void InitOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Read::get(), userData(), mlir::SideEffects::DefaultResource::get());
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

BooleanType InitOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<BooleanType>();
}

mlir::ValueRange InitOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value InitOp::userData()
{
	return getOperation()->getOperand(0);
}

//===----------------------------------------------------------------------===//
// Ida::StepOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> StepOp::getAttributeNames()
{
	return {};
}

void StepOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData)
{
	state.addTypes(BooleanType::get(builder.getContext()));
	state.addOperands(userData);
}

mlir::ParseResult StepOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 1);
}

void StepOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult StepOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	return mlir::success();
}

void StepOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Read::get(), userData(), mlir::SideEffects::DefaultResource::get());
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

BooleanType StepOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<BooleanType>();
}

mlir::ValueRange StepOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value StepOp::userData()
{
	return getOperation()->getOperand(0);
}

//===----------------------------------------------------------------------===//
// Ida::AddTimeOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> AddTimeOp::getAttributeNames()
{
	return {};
}

void AddTimeOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value start, mlir::Value end, mlir::Value step)
{
	state.addOperands(userData);
	state.addOperands(start);
	state.addOperands(end);
	state.addOperands(step);
}

mlir::ParseResult AddTimeOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 3);
}

void AddTimeOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args());
}

mlir::LogicalResult AddTimeOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!start().getType().isa<RealType>() || !end().getType().isa<RealType>() || !step().getType().isa<RealType>())
		return emitOpError("Requires start, end and step time to be real numbers");

	return mlir::success();
}

void AddTimeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

mlir::ValueRange AddTimeOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value AddTimeOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value AddTimeOp::start()
{
	return getOperation()->getOperand(1);
}

mlir::Value AddTimeOp::end()
{
	return getOperation()->getOperand(2);
}

mlir::Value AddTimeOp::step()
{
	return getOperation()->getOperand(3);
}

//===----------------------------------------------------------------------===//
// Ida::AddToleranceOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> AddToleranceOp::getAttributeNames()
{
	return {};
}

void AddToleranceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value relTol, mlir::Value absTol)
{
	state.addOperands(userData);
	state.addOperands(relTol);
	state.addOperands(absTol);
}

mlir::ParseResult AddToleranceOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 3);
}

void AddToleranceOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args());
}

mlir::LogicalResult AddToleranceOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!relTol().getType().isa<RealType>() || !absTol().getType().isa<RealType>())
		return emitOpError("Requires start and stop time to be real numbers");

	return mlir::success();
}

void AddToleranceOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

mlir::ValueRange AddToleranceOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value AddToleranceOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value AddToleranceOp::relTol()
{
	return getOperation()->getOperand(1);
}

mlir::Value AddToleranceOp::absTol()
{
	return getOperation()->getOperand(2);
}

//===----------------------------------------------------------------------===//
// Ida::AddRowLengthOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> AddRowLengthOp::getAttributeNames()
{
	return {};
}

void AddRowLengthOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value rowLength)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(rowLength);
}

mlir::ParseResult AddRowLengthOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void AddRowLengthOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult AddRowLengthOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!rowLength().getType().isa<IntegerType>())
		return emitOpError("Requires BLT row pointer to be an integer");

	return mlir::success();
}

void AddRowLengthOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType AddRowLengthOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange AddRowLengthOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value AddRowLengthOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value AddRowLengthOp::rowLength()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::AddColumnIndexOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> AddColumnIndexOp::getAttributeNames()
{
	return {};
}

void AddColumnIndexOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value rowIndex, mlir::Value accessIndex)
{
	state.addOperands(userData);
	state.addOperands(rowIndex);
	state.addOperands(accessIndex);
}

mlir::ParseResult AddColumnIndexOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 3);
}

void AddColumnIndexOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args());
}

mlir::LogicalResult AddColumnIndexOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!rowIndex().getType().isa<IntegerType>())
		return emitOpError("Requires BLT row pointer to be an integer");

	if (!accessIndex().getType().isa<IntegerType>())
		return emitOpError("Requires BLT column index to be an integer");

	return mlir::success();
}

void AddColumnIndexOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

mlir::ValueRange AddColumnIndexOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value AddColumnIndexOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value AddColumnIndexOp::rowIndex()
{
	return getOperation()->getOperand(1);
}

mlir::Value AddColumnIndexOp::accessIndex()
{
	return getOperation()->getOperand(2);
}

//===----------------------------------------------------------------------===//
// Ida::AddEquationDimensionOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> AddEquationDimensionOp::getAttributeNames()
{
	return {};
}

void AddEquationDimensionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value index, mlir::Value min, mlir::Value max)
{
	state.addOperands(userData);
	state.addOperands(index);
	state.addOperands(min);
	state.addOperands(max);
}

mlir::ParseResult AddEquationDimensionOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 4);
}

void AddEquationDimensionOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args());
}

mlir::LogicalResult AddEquationDimensionOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!index().getType().isa<IntegerType>())
		return emitOpError("Requires equation index to be an integer");

	if (!min().getType().isa<IntegerType>() || !max().getType().isa<IntegerType>())
		return emitOpError("Requires min and max iteration index to be integers");

	return mlir::success();
}

void AddEquationDimensionOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

mlir::ValueRange AddEquationDimensionOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value AddEquationDimensionOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value AddEquationDimensionOp::index()
{
	return getOperation()->getOperand(1);
}

mlir::Value AddEquationDimensionOp::min()
{
	return getOperation()->getOperand(2);
}

mlir::Value AddEquationDimensionOp::max()
{
	return getOperation()->getOperand(3);
}

//===----------------------------------------------------------------------===//
// Ida::AddResidualOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> AddResidualOp::getAttributeNames()
{
	return {};
}

void AddResidualOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value leftIndex, mlir::Value rightIndex)
{
	state.addOperands(userData);
	state.addOperands(leftIndex);
	state.addOperands(rightIndex);
}

mlir::ParseResult AddResidualOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 3);
}

void AddResidualOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args());
}

mlir::LogicalResult AddResidualOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!leftIndex().getType().isa<IntegerType>() || !rightIndex().getType().isa<IntegerType>())
		return emitOpError("Requires left and right lambda indexes to be integers");

	return mlir::success();
}

void AddResidualOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

mlir::ValueRange AddResidualOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value AddResidualOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value AddResidualOp::leftIndex()
{
	return getOperation()->getOperand(1);
}

mlir::Value AddResidualOp::rightIndex()
{
	return getOperation()->getOperand(2);
}

//===----------------------------------------------------------------------===//
// Ida::AddJacobianOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> AddJacobianOp::getAttributeNames()
{
	return {};
}

void AddJacobianOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value leftIndex, mlir::Value rightIndex)
{
	state.addOperands(userData);
	state.addOperands(leftIndex);
	state.addOperands(rightIndex);
}

mlir::ParseResult AddJacobianOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 3);
}

void AddJacobianOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args());
}

mlir::LogicalResult AddJacobianOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!leftIndex().getType().isa<IntegerType>() || !rightIndex().getType().isa<IntegerType>())
		return emitOpError("Requires left and right lambda indexes to be integers");

	return mlir::success();
}

void AddJacobianOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

mlir::ValueRange AddJacobianOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value AddJacobianOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value AddJacobianOp::leftIndex()
{
	return getOperation()->getOperand(1);
}

mlir::Value AddJacobianOp::rightIndex()
{
	return getOperation()->getOperand(2);
}

//===----------------------------------------------------------------------===//
// Ida::AddVariableOffsetOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> AddVariableOffsetOp::getAttributeNames()
{
	return {};
}

void AddVariableOffsetOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value dimension)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(dimension);
}

mlir::ParseResult AddVariableOffsetOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void AddVariableOffsetOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult AddVariableOffsetOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!offset().getType().isa<IntegerType>())
		return emitOpError("Requires lambda dimension to be an integer");

	return mlir::success();
}

void AddVariableOffsetOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType AddVariableOffsetOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange AddVariableOffsetOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value AddVariableOffsetOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value AddVariableOffsetOp::offset()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::AddVariableDimensionOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> AddVariableDimensionOp::getAttributeNames()
{
	return {};
}

void AddVariableDimensionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value variable, mlir::Value dimension)
{
	state.addOperands(userData);
	state.addOperands(variable);
	state.addOperands(dimension);
}

mlir::ParseResult AddVariableDimensionOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 3);
}

void AddVariableDimensionOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args());
}

mlir::LogicalResult AddVariableDimensionOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!variable().getType().isa<IntegerType>())
		return emitOpError("Requires variable index to be an integer");

	if (!dimension().getType().isa<IntegerType>())
		return emitOpError("Requires variable dimension to be an integer");

	return mlir::success();
}

void AddVariableDimensionOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

mlir::ValueRange AddVariableDimensionOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value AddVariableDimensionOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value AddVariableDimensionOp::variable()
{
	return getOperation()->getOperand(1);
}

mlir::Value AddVariableDimensionOp::dimension()
{
	return getOperation()->getOperand(2);
}

//===----------------------------------------------------------------------===//
// Ida::AddNewVariableAccessOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> AddNewVariableAccessOp::getAttributeNames()
{
	return {};
}

void AddNewVariableAccessOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value variable, mlir::Value offset, mlir::Value induction)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(variable);
	state.addOperands(offset);
	state.addOperands(induction);
}

mlir::ParseResult AddNewVariableAccessOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 4);
}

void AddNewVariableAccessOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult AddNewVariableAccessOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!variable().getType().isa<IntegerType>())
		return emitOpError("Requires variable index to be an integer");

	if (!offset().getType().isa<IntegerType>() || !induction().getType().isa<IntegerType>())
		return emitOpError("Requires variable offset and induction to be integers");

	return mlir::success();
}

void AddNewVariableAccessOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType AddNewVariableAccessOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange AddNewVariableAccessOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value AddNewVariableAccessOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value AddNewVariableAccessOp::variable()
{
	return getOperation()->getOperand(1);
}

mlir::Value AddNewVariableAccessOp::offset()
{
	return getOperation()->getOperand(2);
}

mlir::Value AddNewVariableAccessOp::induction()
{
	return getOperation()->getOperand(3);
}

//===----------------------------------------------------------------------===//
// Ida::AddVariableAccessOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> AddVariableAccessOp::getAttributeNames()
{
	return {};
}

void AddVariableAccessOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value index, mlir::Value offset, mlir::Value induction)
{
	state.addOperands(userData);
	state.addOperands(index);
	state.addOperands(offset);
	state.addOperands(induction);
}

mlir::ParseResult AddVariableAccessOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 4);
}

void AddVariableAccessOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args());
}

mlir::LogicalResult AddVariableAccessOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!index().getType().isa<IntegerType>())
		return emitOpError("Requires variable access index to be an integer");

	if (!offset().getType().isa<IntegerType>() || !induction().getType().isa<IntegerType>())
		return emitOpError("Requires variable offset and induction to be integers");

	return mlir::success();
}

void AddVariableAccessOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

mlir::ValueRange AddVariableAccessOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value AddVariableAccessOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value AddVariableAccessOp::index()
{
	return getOperation()->getOperand(1);
}

mlir::Value AddVariableAccessOp::offset()
{
	return getOperation()->getOperand(2);
}

mlir::Value AddVariableAccessOp::induction()
{
	return getOperation()->getOperand(3);
}

//===----------------------------------------------------------------------===//
// Ida::GetTimeOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> GetTimeOp::getAttributeNames()
{
	return {};
}

void GetTimeOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData)
{
	state.addTypes(marco::codegen::modelica::RealType::get(builder.getContext()));
	state.addOperands(userData);
}

mlir::ParseResult GetTimeOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 1);
}

void GetTimeOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult GetTimeOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	return mlir::success();
}

void GetTimeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Read::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

marco::codegen::modelica::RealType GetTimeOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<marco::codegen::modelica::RealType>();
}

mlir::ValueRange GetTimeOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value GetTimeOp::userData()
{
	return getOperation()->getOperand(0);
}

//===----------------------------------------------------------------------===//
// Ida::GetVariableOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> GetVariableOp::getAttributeNames()
{
	return {};
}

void GetVariableOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value index)
{
	state.addTypes(marco::codegen::modelica::RealType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(index);
}

mlir::ParseResult GetVariableOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void GetVariableOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult GetVariableOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!index().getType().isa<IntegerType>() && !index().getType().isa<marco::codegen::modelica::IntegerType>())
		return emitOpError("Requires variable index to be an integer");

	return mlir::success();
}

void GetVariableOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Read::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

marco::codegen::modelica::RealType GetVariableOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<marco::codegen::modelica::RealType>();
}

mlir::ValueRange GetVariableOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value GetVariableOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value GetVariableOp::index()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::GetDerivativeOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> GetDerivativeOp::getAttributeNames()
{
	return {};
}

void GetDerivativeOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value index)
{
	state.addTypes(marco::codegen::modelica::RealType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(index);
}

mlir::ParseResult GetDerivativeOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void GetDerivativeOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult GetDerivativeOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!index().getType().isa<IntegerType>() && !index().getType().isa<marco::codegen::modelica::IntegerType>())
		return emitOpError("Requires variable index to be an integer");

	return mlir::success();
}

void GetDerivativeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Read::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

marco::codegen::modelica::RealType GetDerivativeOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<marco::codegen::modelica::RealType>();
}

mlir::ValueRange GetDerivativeOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value GetDerivativeOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value GetDerivativeOp::index()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaConstantOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaConstantOp::getAttributeNames()
{
	return {};
}

void LambdaConstantOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value constant)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(constant);
}

mlir::ParseResult LambdaConstantOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void LambdaConstantOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaConstantOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!constant().getType().isa<RealType>() && !constant().getType().isa<IntegerType>())
		return emitOpError("Requires lambda constant to be a number");

	return mlir::success();
}

void LambdaConstantOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaConstantOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaConstantOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaConstantOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaConstantOp::constant()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaTimeOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaTimeOp::getAttributeNames()
{
	return {};
}

void LambdaTimeOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
}

mlir::ParseResult LambdaTimeOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 1);
}

void LambdaTimeOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaTimeOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	return mlir::success();
}

void LambdaTimeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaTimeOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaTimeOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaTimeOp::userData()
{
	return getOperation()->getOperand(0);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaInductionOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaInductionOp::getAttributeNames()
{
	return {};
}

void LambdaInductionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value induction)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(induction);
}

mlir::ParseResult LambdaInductionOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void LambdaInductionOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaInductionOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!induction().getType().isa<IntegerType>())
		return emitOpError("Requires induction index to be an integer");

	return mlir::success();
}

void LambdaInductionOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaInductionOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaInductionOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaInductionOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaInductionOp::induction()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaVariableOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaVariableOp::getAttributeNames()
{
	return {};
}

void LambdaVariableOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value accessIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(accessIndex);
}

mlir::ParseResult LambdaVariableOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void LambdaVariableOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaVariableOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!accessIndex().getType().isa<IntegerType>())
		return emitOpError("Requires lambda access index to be an integer");

	return mlir::success();
}

void LambdaVariableOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaVariableOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaVariableOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaVariableOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaVariableOp::accessIndex()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaDerivativeOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaDerivativeOp::getAttributeNames()
{
	return {};
}

void LambdaDerivativeOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value accessIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(accessIndex);
}

mlir::ParseResult LambdaDerivativeOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void LambdaDerivativeOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaDerivativeOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!accessIndex().getType().isa<IntegerType>())
		return emitOpError("Requires lambda access index to be an integer");

	return mlir::success();
}

void LambdaDerivativeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaDerivativeOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaDerivativeOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaDerivativeOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaDerivativeOp::accessIndex()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaAddOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaAddOp::getAttributeNames()
{
	return {};
}

void LambdaAddOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value leftIndex, mlir::Value rightIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(leftIndex);
	state.addOperands(rightIndex);
}

mlir::ParseResult LambdaAddOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 3);
}

void LambdaAddOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaAddOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!leftIndex().getType().isa<IntegerType>() || !rightIndex().getType().isa<IntegerType>())
		return emitOpError("Requires left and right lambda indexes to be integers");

	return mlir::success();
}

void LambdaAddOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaAddOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaAddOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaAddOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaAddOp::leftIndex()
{
	return getOperation()->getOperand(1);
}

mlir::Value LambdaAddOp::rightIndex()
{
	return getOperation()->getOperand(2);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaSubOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaSubOp::getAttributeNames()
{
	return {};
}

void LambdaSubOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value leftIndex, mlir::Value rightIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(leftIndex);
	state.addOperands(rightIndex);
}

mlir::ParseResult LambdaSubOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 3);
}

void LambdaSubOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaSubOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!leftIndex().getType().isa<IntegerType>() || !rightIndex().getType().isa<IntegerType>())
		return emitOpError("Requires left and right lambda indexes to be integers");

	return mlir::success();
}

void LambdaSubOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaSubOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaSubOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaSubOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaSubOp::leftIndex()
{
	return getOperation()->getOperand(1);
}

mlir::Value LambdaSubOp::rightIndex()
{
	return getOperation()->getOperand(2);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaMulOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaMulOp::getAttributeNames()
{
	return {};
}

void LambdaMulOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value leftIndex, mlir::Value rightIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(leftIndex);
	state.addOperands(rightIndex);
}

mlir::ParseResult LambdaMulOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 3);
}

void LambdaMulOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaMulOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!leftIndex().getType().isa<IntegerType>() || !rightIndex().getType().isa<IntegerType>())
		return emitOpError("Requires left and right lambda indexes to be integers");

	return mlir::success();
}

void LambdaMulOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaMulOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaMulOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaMulOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaMulOp::leftIndex()
{
	return getOperation()->getOperand(1);
}

mlir::Value LambdaMulOp::rightIndex()
{
	return getOperation()->getOperand(2);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaDivOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaDivOp::getAttributeNames()
{
	return {};
}

void LambdaDivOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value leftIndex, mlir::Value rightIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(leftIndex);
	state.addOperands(rightIndex);
}

mlir::ParseResult LambdaDivOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 3);
}

void LambdaDivOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaDivOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!leftIndex().getType().isa<IntegerType>() || !rightIndex().getType().isa<IntegerType>())
		return emitOpError("Requires left and right lambda indexes to be integers");

	return mlir::success();
}

void LambdaDivOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaDivOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaDivOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaDivOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaDivOp::leftIndex()
{
	return getOperation()->getOperand(1);
}

mlir::Value LambdaDivOp::rightIndex()
{
	return getOperation()->getOperand(2);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaPowOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaPowOp::getAttributeNames()
{
	return {};
}

void LambdaPowOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value leftIndex, mlir::Value rightIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(leftIndex);
	state.addOperands(rightIndex);
}

mlir::ParseResult LambdaPowOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 3);
}

void LambdaPowOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaPowOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!leftIndex().getType().isa<IntegerType>() || !rightIndex().getType().isa<IntegerType>())
		return emitOpError("Requires left and right lambda indexes to be integers");

	return mlir::success();
}

void LambdaPowOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaPowOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaPowOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaPowOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaPowOp::leftIndex()
{
	return getOperation()->getOperand(1);
}

mlir::Value LambdaPowOp::rightIndex()
{
	return getOperation()->getOperand(2);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaAtan2Op
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaAtan2Op::getAttributeNames()
{
	return {};
}

void LambdaAtan2Op::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value leftIndex, mlir::Value rightIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(leftIndex);
	state.addOperands(rightIndex);
}

mlir::ParseResult LambdaAtan2Op::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 3);
}

void LambdaAtan2Op::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaAtan2Op::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!leftIndex().getType().isa<IntegerType>() || !rightIndex().getType().isa<IntegerType>())
		return emitOpError("Requires left and right lambda indexes to be integers");

	return mlir::success();
}

void LambdaAtan2Op::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaAtan2Op::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaAtan2Op::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaAtan2Op::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaAtan2Op::leftIndex()
{
	return getOperation()->getOperand(1);
}

mlir::Value LambdaAtan2Op::rightIndex()
{
	return getOperation()->getOperand(2);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaNegateOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaNegateOp::getAttributeNames()
{
	return {};
}

void LambdaNegateOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(operandIndex);
}

mlir::ParseResult LambdaNegateOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void LambdaNegateOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaNegateOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!operandIndex().getType().isa<IntegerType>())
		return emitOpError("Requires operand lambda index to be an integer");

	return mlir::success();
}

void LambdaNegateOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaNegateOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaNegateOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaNegateOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaNegateOp::operandIndex()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaAbsOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaAbsOp::getAttributeNames()
{
	return {};
}

void LambdaAbsOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(operandIndex);
}

mlir::ParseResult LambdaAbsOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void LambdaAbsOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaAbsOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!operandIndex().getType().isa<IntegerType>())
		return emitOpError("Requires operand lambda index to be an integer");

	return mlir::success();
}

void LambdaAbsOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaAbsOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaAbsOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaAbsOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaAbsOp::operandIndex()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaSignOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaSignOp::getAttributeNames()
{
	return {};
}

void LambdaSignOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(operandIndex);
}

mlir::ParseResult LambdaSignOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void LambdaSignOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaSignOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!operandIndex().getType().isa<IntegerType>())
		return emitOpError("Requires operand lambda index to be an integer");

	return mlir::success();
}

void LambdaSignOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaSignOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaSignOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaSignOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaSignOp::operandIndex()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaSqrtOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaSqrtOp::getAttributeNames()
{
	return {};
}

void LambdaSqrtOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(operandIndex);
}

mlir::ParseResult LambdaSqrtOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void LambdaSqrtOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaSqrtOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!operandIndex().getType().isa<IntegerType>())
		return emitOpError("Requires operand lambda index to be an integer");

	return mlir::success();
}

void LambdaSqrtOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaSqrtOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaSqrtOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaSqrtOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaSqrtOp::operandIndex()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaExpOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaExpOp::getAttributeNames()
{
	return {};
}

void LambdaExpOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(operandIndex);
}

mlir::ParseResult LambdaExpOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void LambdaExpOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaExpOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!operandIndex().getType().isa<IntegerType>())
		return emitOpError("Requires operand lambda index to be an integer");

	return mlir::success();
}

void LambdaExpOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaExpOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaExpOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaExpOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaExpOp::operandIndex()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaLogOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaLogOp::getAttributeNames()
{
	return {};
}

void LambdaLogOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(operandIndex);
}

mlir::ParseResult LambdaLogOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void LambdaLogOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaLogOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!operandIndex().getType().isa<IntegerType>())
		return emitOpError("Requires operand lambda index to be an integer");

	return mlir::success();
}

void LambdaLogOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaLogOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaLogOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaLogOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaLogOp::operandIndex()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaLog10Op
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaLog10Op::getAttributeNames()
{
	return {};
}

void LambdaLog10Op::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(operandIndex);
}

mlir::ParseResult LambdaLog10Op::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void LambdaLog10Op::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaLog10Op::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!operandIndex().getType().isa<IntegerType>())
		return emitOpError("Requires operand lambda index to be an integer");

	return mlir::success();
}

void LambdaLog10Op::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaLog10Op::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaLog10Op::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaLog10Op::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaLog10Op::operandIndex()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaSinOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaSinOp::getAttributeNames()
{
	return {};
}

void LambdaSinOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(operandIndex);
}

mlir::ParseResult LambdaSinOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void LambdaSinOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaSinOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!operandIndex().getType().isa<IntegerType>())
		return emitOpError("Requires operand lambda index to be an integer");

	return mlir::success();
}

void LambdaSinOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaSinOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaSinOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaSinOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaSinOp::operandIndex()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaCosOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaCosOp::getAttributeNames()
{
	return {};
}

void LambdaCosOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(operandIndex);
}

mlir::ParseResult LambdaCosOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void LambdaCosOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaCosOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!operandIndex().getType().isa<IntegerType>())
		return emitOpError("Requires operand lambda index to be an integer");

	return mlir::success();
}

void LambdaCosOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaCosOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaCosOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaCosOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaCosOp::operandIndex()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaTanOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaTanOp::getAttributeNames()
{
	return {};
}

void LambdaTanOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(operandIndex);
}

mlir::ParseResult LambdaTanOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void LambdaTanOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaTanOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!operandIndex().getType().isa<IntegerType>())
		return emitOpError("Requires operand lambda index to be an integer");

	return mlir::success();
}

void LambdaTanOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaTanOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaTanOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaTanOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaTanOp::operandIndex()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaAsinOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaAsinOp::getAttributeNames()
{
	return {};
}

void LambdaAsinOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(operandIndex);
}

mlir::ParseResult LambdaAsinOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void LambdaAsinOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaAsinOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!operandIndex().getType().isa<IntegerType>())
		return emitOpError("Requires operand lambda index to be an integer");

	return mlir::success();
}

void LambdaAsinOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaAsinOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaAsinOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaAsinOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaAsinOp::operandIndex()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaAcosOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaAcosOp::getAttributeNames()
{
	return {};
}

void LambdaAcosOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(operandIndex);
}

mlir::ParseResult LambdaAcosOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void LambdaAcosOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaAcosOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!operandIndex().getType().isa<IntegerType>())
		return emitOpError("Requires operand lambda index to be an integer");

	return mlir::success();
}

void LambdaAcosOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaAcosOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaAcosOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaAcosOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaAcosOp::operandIndex()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaAtanOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaAtanOp::getAttributeNames()
{
	return {};
}

void LambdaAtanOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(operandIndex);
}

mlir::ParseResult LambdaAtanOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void LambdaAtanOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaAtanOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!operandIndex().getType().isa<IntegerType>())
		return emitOpError("Requires operand lambda index to be an integer");

	return mlir::success();
}

void LambdaAtanOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaAtanOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaAtanOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaAtanOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaAtanOp::operandIndex()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaSinhOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaSinhOp::getAttributeNames()
{
	return {};
}

void LambdaSinhOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(operandIndex);
}

mlir::ParseResult LambdaSinhOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void LambdaSinhOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaSinhOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!operandIndex().getType().isa<IntegerType>())
		return emitOpError("Requires operand lambda index to be an integer");

	return mlir::success();
}

void LambdaSinhOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaSinhOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaSinhOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaSinhOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaSinhOp::operandIndex()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaCoshOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaCoshOp::getAttributeNames()
{
	return {};
}

void LambdaCoshOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(operandIndex);
}

mlir::ParseResult LambdaCoshOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void LambdaCoshOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaCoshOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!operandIndex().getType().isa<IntegerType>())
		return emitOpError("Requires operand lambda index to be an integer");

	return mlir::success();
}

void LambdaCoshOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaCoshOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaCoshOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaCoshOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaCoshOp::operandIndex()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaTanhOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaTanhOp::getAttributeNames()
{
	return {};
}

void LambdaTanhOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(operandIndex);
}

mlir::ParseResult LambdaTanhOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void LambdaTanhOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaTanhOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!operandIndex().getType().isa<IntegerType>())
		return emitOpError("Requires operand lambda index to be an integer");

	return mlir::success();
}

void LambdaTanhOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaTanhOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaTanhOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaTanhOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaTanhOp::operandIndex()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaCallOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaCallOp::getAttributeNames()
{
	return {};
}

void LambdaCallOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex, mlir::Value functionAddress, mlir::Value pderAddress)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(operandIndex);
	state.addOperands(functionAddress);
	state.addOperands(pderAddress);
}

mlir::ParseResult LambdaCallOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 3);
}

void LambdaCallOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult LambdaCallOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!operandIndex().getType().isa<IntegerType>())
		return emitOpError("Requires operand lambda index to be an integer");

	if (!functionAddress().getType().isa<mlir::LLVM::LLVMPointerType>() || !pderAddress().getType().isa<mlir::LLVM::LLVMPointerType>())
		return emitOpError("Requires callee addresses to be a pointeres to functions");

	return mlir::success();
}

void LambdaCallOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaCallOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaCallOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaCallOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaCallOp::operandIndex()
{
	return getOperation()->getOperand(1);
}

mlir::Value LambdaCallOp::functionAddress()
{
	return getOperation()->getOperand(2);
}

mlir::Value LambdaCallOp::pderAddress()
{
	return getOperation()->getOperand(3);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaAddressOfOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LambdaAddressOfOp::getAttributeNames()
{
	static llvm::StringRef attrNames[] = {llvm::StringRef("callee")};
	return llvm::makeArrayRef(attrNames);
}

void LambdaAddressOfOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::StringRef callee, mlir::Type realType)
{
	state.addTypes(mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMFunctionType::get(realType, realType)));
	state.addAttribute("callee", builder.getSymbolRefAttr(callee));
}

mlir::ParseResult LambdaAddressOfOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::Builder& builder = parser.getBuilder();

	mlir::FlatSymbolRefAttr calleeAttr;
	mlir::Type resultType;

	if (parser.parseAttribute(calleeAttr, builder.getType<mlir::NoneType>(), "callee", result.attributes) ||
			parser.parseColon() ||
			parser.parseType(resultType))
		return mlir::failure();

	result.addTypes(resultType);

	return mlir::success();
}

void LambdaAddressOfOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " @" << callee() << " : " << resultType();
}

mlir::LogicalResult LambdaAddressOfOp::verify()
{
	return mlir::success();
}

mlir::Type LambdaAddressOfOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::StringRef LambdaAddressOfOp::callee()
{
	return getOperation()->getAttrOfType<mlir::FlatSymbolRefAttr>("callee").getValue();
}
