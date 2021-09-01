#include <mlir/Conversion/Passes.h>
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

		for (size_t i = 0; i < values.size(); i++)
			printer << ", " << values[i];

		printer << " : (" << values[0].getType();

		for (size_t i = 0; i < values.size(); i++)
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

void AllocUserDataOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value neq, mlir::Value nnz)
{
	state.addTypes(OpaquePointerType::get(builder.getContext()));
	state.addOperands(neq);
	state.addOperands(nnz);
}

mlir::ParseResult AllocUserDataOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void AllocUserDataOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
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

mlir::Value AllocUserDataOp::neq()
{
	return getOperation()->getOperand(0);
}

mlir::Value AllocUserDataOp::nnz()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::FreeUserDataOp
//===----------------------------------------------------------------------===//

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
// Ida::InitOp
//===----------------------------------------------------------------------===//

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

void AddTimeOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value start, mlir::Value stop)
{
	state.addOperands(userData);
	state.addOperands(start);
	state.addOperands(stop);
}

mlir::ParseResult AddTimeOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 3);
}

void AddTimeOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args());
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

mlir::Value AddTimeOp::stop()
{
	return getOperation()->getOperand(2);
}

//===----------------------------------------------------------------------===//
// Ida::AddToleranceOp
//===----------------------------------------------------------------------===//

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

void AddRowLengthOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value rowLength)
{
	state.addOperands(userData);
	state.addOperands(rowLength);
}

mlir::ParseResult AddRowLengthOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void AddRowLengthOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args());
}

void AddRowLengthOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
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
// Ida::AddDimensionOp
//===----------------------------------------------------------------------===//

void AddDimensionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value index, mlir::Value min, mlir::Value max)
{
	state.addOperands(userData);
	state.addOperands(index);
	state.addOperands(min);
	state.addOperands(max);
}

mlir::ParseResult AddDimensionOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 4);
}

void AddDimensionOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args());
}

void AddDimensionOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

mlir::ValueRange AddDimensionOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value AddDimensionOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value AddDimensionOp::index()
{
	return getOperation()->getOperand(1);
}

mlir::Value AddDimensionOp::min()
{
	return getOperation()->getOperand(2);
}

mlir::Value AddDimensionOp::max()
{
	return getOperation()->getOperand(3);
}

//===----------------------------------------------------------------------===//
// Ida::AddResidualOp
//===----------------------------------------------------------------------===//

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
// Ida::GetTimeOp
//===----------------------------------------------------------------------===//

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

void GetVariableOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value index)
{
	state.addTypes(RealType::get(builder.getContext()));
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
// Ida::AddNewLambdaAccessOp
//===----------------------------------------------------------------------===//

void AddNewLambdaAccessOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value offset, mlir::Value indices)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(offset);
	state.addOperands(indices);
}

mlir::ParseResult AddNewLambdaAccessOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 3);
}

void AddNewLambdaAccessOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

void AddNewLambdaAccessOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType AddNewLambdaAccessOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange AddNewLambdaAccessOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value AddNewLambdaAccessOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value AddNewLambdaAccessOp::offset()
{
	return getOperation()->getOperand(1);
}

mlir::Value AddNewLambdaAccessOp::indices()
{
	return getOperation()->getOperand(2);
}

//===----------------------------------------------------------------------===//
// Ida::AddLambdaAccessOp
//===----------------------------------------------------------------------===//

void AddLambdaAccessOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value index, mlir::Value offset, mlir::Value indices)
{
	state.addOperands(userData);
	state.addOperands(index);
	state.addOperands(offset);
	state.addOperands(indices);
}

mlir::ParseResult AddLambdaAccessOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 4);
}

void AddLambdaAccessOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args());
}

void AddLambdaAccessOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

mlir::ValueRange AddLambdaAccessOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value AddLambdaAccessOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value AddLambdaAccessOp::index()
{
	return getOperation()->getOperand(1);
}

mlir::Value AddLambdaAccessOp::offset()
{
	return getOperation()->getOperand(2);
}

mlir::Value AddLambdaAccessOp::indices()
{
	return getOperation()->getOperand(3);
}

//===----------------------------------------------------------------------===//
// Ida::AddLambdaDimensionOp
//===----------------------------------------------------------------------===//

void AddLambdaDimensionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value index, mlir::Value dimension)
{
	state.addOperands(userData);
	state.addOperands(index);
	state.addOperands(dimension);
}

mlir::ParseResult AddLambdaDimensionOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 3);
}

void AddLambdaDimensionOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args());
}

void AddLambdaDimensionOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

mlir::ValueRange AddLambdaDimensionOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value AddLambdaDimensionOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value AddLambdaDimensionOp::index()
{
	return getOperation()->getOperand(1);
}

mlir::Value AddLambdaDimensionOp::dimension()
{
	return getOperation()->getOperand(2);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaConstantOp
//===----------------------------------------------------------------------===//

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
// Ida::LambdaScalarVariableOp
//===----------------------------------------------------------------------===//

void LambdaScalarVariableOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value offset)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(offset);
}

mlir::ParseResult LambdaScalarVariableOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void LambdaScalarVariableOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

void LambdaScalarVariableOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaScalarVariableOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaScalarVariableOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaScalarVariableOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaScalarVariableOp::offset()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaScalarDerivativeOp
//===----------------------------------------------------------------------===//

void LambdaScalarDerivativeOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value offset)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(offset);
}

mlir::ParseResult LambdaScalarDerivativeOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 2);
}

void LambdaScalarDerivativeOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

void LambdaScalarDerivativeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaScalarDerivativeOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaScalarDerivativeOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaScalarDerivativeOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaScalarDerivativeOp::offset()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaVectorVariableOp
//===----------------------------------------------------------------------===//

void LambdaVectorVariableOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value offset, mlir::Value index)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(offset);
	state.addOperands(index);
}

mlir::ParseResult LambdaVectorVariableOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 3);
}

void LambdaVectorVariableOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

void LambdaVectorVariableOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaVectorVariableOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaVectorVariableOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaVectorVariableOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaVectorVariableOp::offset()
{
	return getOperation()->getOperand(1);
}

mlir::Value LambdaVectorVariableOp::index()
{
	return getOperation()->getOperand(2);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaVectorDerivativeOp
//===----------------------------------------------------------------------===//

void LambdaVectorDerivativeOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value offset, mlir::Value index)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(offset);
	state.addOperands(index);
}

mlir::ParseResult LambdaVectorDerivativeOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return marco::codegen::ida::parse(parser, result, 3);
}

void LambdaVectorDerivativeOp::print(mlir::OpAsmPrinter& printer)
{
	marco::codegen::ida::print(printer, getOperationName(), args(), resultType());
}

void LambdaVectorDerivativeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType LambdaVectorDerivativeOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange LambdaVectorDerivativeOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LambdaVectorDerivativeOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value LambdaVectorDerivativeOp::offset()
{
	return getOperation()->getOperand(1);
}

mlir::Value LambdaVectorDerivativeOp::index()
{
	return getOperation()->getOperand(2);
}

//===----------------------------------------------------------------------===//
// Ida::LambdaAddOp
//===----------------------------------------------------------------------===//

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
// Ida::LambdaNegateOp
//===----------------------------------------------------------------------===//

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
