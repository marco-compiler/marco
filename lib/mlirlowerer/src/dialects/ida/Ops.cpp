#include <mlir/Conversion/Passes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <marco/mlirlowerer/dialects/ida/Attribute.h>
#include <marco/mlirlowerer/dialects/ida/Ops.h>

using namespace marco::codegen::ida;

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
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 2> operands;
	llvm::SmallVector<mlir::Type, 2> operandsTypes;
	mlir::Type resultType;

	llvm::SMLoc operandsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(operands, 2) ||
			parser.parseColon() ||
			parser.parseLParen() ||
			parser.parseTypeList(operandsTypes) ||
			parser.parseRParen() ||
			parser.resolveOperands(operands, operandsTypes, operandsLoc, result.operands) ||
			parser.parseArrow() ||
			parser.parseType(resultType))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void AllocUserDataOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName()
					<< " " << neq() << ", " << nnz()
					<< " : (" << neq().getType() << ", " << nnz().getType()
					<< ") -> " << resultType();
}

void AllocUserDataOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::SideEffects::DefaultResource::get());
}

mlir::Type AllocUserDataOp::resultType()
{
	return getOperation()->getResultTypes()[0];
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
	mlir::OpAsmParser::OperandType operand;
	mlir::Type operandType;
	mlir::Type resultType;

	if (parser.parseOperand(operand) ||
			parser.parseColon() ||
			parser.parseType(operandType) ||
			parser.resolveOperand(operand, operandType, result.operands) ||
			parser.parseArrow() ||
			parser.parseType(resultType))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void FreeUserDataOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << userData() << " : " << userData().getType() << " -> " << resultType();
}

void FreeUserDataOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::SideEffects::DefaultResource::get());
}

mlir::Type FreeUserDataOp::resultType()
{
	return getOperation()->getResultTypes()[0];
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

void SetInitialValueOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value index, mlir::Value value, mlir::Value isState)
{
	state.addOperands(userData);
	state.addOperands(index);
	state.addOperands(value);
	state.addOperands(isState);
}

mlir::ParseResult SetInitialValueOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> operands;
	llvm::SmallVector<mlir::Type, 4> operandsTypes;

	llvm::SMLoc operandsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(operands, 4) ||
			parser.parseColon() ||
			parser.parseLParen() ||
			parser.parseTypeList(operandsTypes) ||
			parser.parseRParen() ||
			parser.resolveOperands(operands, operandsTypes, operandsLoc, result.operands))
		return mlir::failure();

	return mlir::success();
}

void SetInitialValueOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " "
					<< userData() << ", " << index() << ", "
					<< value() << ", " << isState() << " : ("
					<< userData().getType() << ", "
					<< index().getType() << ", "
					<< value().getType() << ", "
					<< isState().getType() << ")";
}

void SetInitialValueOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::SideEffects::DefaultResource::get());
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

mlir::Value SetInitialValueOp::value()
{
	return getOperation()->getOperand(2);
}

mlir::Value SetInitialValueOp::isState()
{
	return getOperation()->getOperand(3);
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
	mlir::OpAsmParser::OperandType operand;
	mlir::Type operandType;
	mlir::Type resultType;

	if (parser.parseOperand(operand) ||
			parser.parseColon() ||
			parser.parseType(operandType) ||
			parser.resolveOperand(operand, operandType, result.operands) ||
			parser.parseArrow() ||
			parser.parseType(resultType))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void InitOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << userData() << " : " << userData().getType() << " -> " << resultType();
}

void InitOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::SideEffects::DefaultResource::get());
}

mlir::Type InitOp::resultType()
{
	return getOperation()->getResultTypes()[0];
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
	mlir::OpAsmParser::OperandType operand;
	mlir::Type operandType;
	mlir::Type resultType;

	if (parser.parseOperand(operand) ||
			parser.parseColon() ||
			parser.parseType(operandType) ||
			parser.resolveOperand(operand, operandType, result.operands) ||
			parser.parseArrow() ||
			parser.parseType(resultType))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void StepOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << userData() << " : " << userData().getType() << " -> " << resultType();
}

void StepOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::SideEffects::DefaultResource::get());
}

mlir::Type StepOp::resultType()
{
	return getOperation()->getResultTypes()[0];
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
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> operands;
	llvm::SmallVector<mlir::Type, 3> operandsTypes;

	llvm::SMLoc operandsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(operands, 3) ||
			parser.parseColon() ||
			parser.parseLParen() ||
			parser.parseTypeList(operandsTypes) ||
			parser.parseRParen() ||
			parser.resolveOperands(operands, operandsTypes, operandsLoc, result.operands))
		return mlir::failure();

	return mlir::success();
}

void AddTimeOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " "
					<< userData() << ", " << start() << ", " << stop() << " : ("
					<< userData().getType() << ", "
					<< start().getType() << ", "
					<< stop().getType() << ")";
}

void AddTimeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::SideEffects::DefaultResource::get());
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
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> operands;
	llvm::SmallVector<mlir::Type, 3> operandsTypes;

	llvm::SMLoc operandsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(operands, 3) ||
			parser.parseColon() ||
			parser.parseLParen() ||
			parser.parseTypeList(operandsTypes) ||
			parser.parseRParen() ||
			parser.resolveOperands(operands, operandsTypes, operandsLoc, result.operands))
		return mlir::failure();

	return mlir::success();
}

void AddToleranceOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " "
					<< userData() << ", " << relTol() << ", " << absTol() << " : ("
					<< userData().getType() << ", "
					<< relTol().getType() << ", "
					<< absTol().getType() << ")";
}

void AddToleranceOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::SideEffects::DefaultResource::get());
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
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 2> operands;
	llvm::SmallVector<mlir::Type, 2> operandsTypes;

	llvm::SMLoc operandsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(operands, 2) ||
			parser.parseColon() ||
			parser.parseLParen() ||
			parser.parseTypeList(operandsTypes) ||
			parser.parseRParen() ||
			parser.resolveOperands(operands, operandsTypes, operandsLoc, result.operands))
		return mlir::failure();

	return mlir::success();
}

void AddRowLengthOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " "
					<< userData() << ", " << rowLength() << " : ("
					<< userData().getType() << ", "
					<< rowLength().getType() << ")";
}

void AddRowLengthOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::SideEffects::DefaultResource::get());
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
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> operands;
	llvm::SmallVector<mlir::Type, 4> operandsTypes;

	llvm::SMLoc operandsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(operands, 4) ||
			parser.parseColon() ||
			parser.parseLParen() ||
			parser.parseTypeList(operandsTypes) ||
			parser.parseRParen() ||
			parser.resolveOperands(operands, operandsTypes, operandsLoc, result.operands))
		return mlir::failure();

	return mlir::success();
}

void AddDimensionOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " "
					<< userData() << ", " << index() << ", "
					<< min() << ", " << max() << " : ("
					<< userData().getType() << ", "
					<< index().getType() << ", "
					<< min().getType() << ", "
					<< max().getType() << ")";
}

void AddDimensionOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::SideEffects::DefaultResource::get());
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
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> operands;
	llvm::SmallVector<mlir::Type, 3> operandsTypes;

	llvm::SMLoc operandsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(operands, 3) ||
			parser.parseColon() ||
			parser.parseLParen() ||
			parser.parseTypeList(operandsTypes) ||
			parser.parseRParen() ||
			parser.resolveOperands(operands, operandsTypes, operandsLoc, result.operands))
		return mlir::failure();

	return mlir::success();
}

void AddResidualOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " "
					<< userData() << ", " << leftIndex() << ", " << rightIndex() << " : ("
					<< userData().getType() << ", "
					<< leftIndex().getType() << ", "
					<< rightIndex().getType() << ")";
}

void AddResidualOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::SideEffects::DefaultResource::get());
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
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> operands;
	llvm::SmallVector<mlir::Type, 3> operandsTypes;

	llvm::SMLoc operandsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(operands, 3) ||
			parser.parseColon() ||
			parser.parseLParen() ||
			parser.parseTypeList(operandsTypes) ||
			parser.parseRParen() ||
			parser.resolveOperands(operands, operandsTypes, operandsLoc, result.operands))
		return mlir::failure();

	return mlir::success();
}

void AddJacobianOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " "
					<< userData() << ", " << leftIndex() << ", " << rightIndex() << " : ("
					<< userData().getType() << ", "
					<< leftIndex().getType() << ", "
					<< rightIndex().getType() << ")";
}

void AddJacobianOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::SideEffects::DefaultResource::get());
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
	state.addTypes(RealType::get(builder.getContext()));
	state.addOperands(userData);
}

mlir::ParseResult GetTimeOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::OpAsmParser::OperandType operand;
	mlir::Type operandType;
	mlir::Type resultType;

	if (parser.parseOperand(operand) ||
			parser.parseColon() ||
			parser.parseType(operandType) ||
			parser.resolveOperand(operand, operandType, result.operands) ||
			parser.parseArrow() ||
			parser.parseType(resultType))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void GetTimeOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << userData() << " : " << userData().getType() << " -> " << resultType();
}

void GetTimeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Read::get(), mlir::SideEffects::DefaultResource::get());
}

mlir::Type GetTimeOp::resultType()
{
	return getOperation()->getResultTypes()[0];
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
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 2> operands;
	llvm::SmallVector<mlir::Type, 2> operandsTypes;
	mlir::Type resultType;

	llvm::SMLoc operandsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(operands, 2) ||
			parser.parseColon() ||
			parser.parseLParen() ||
			parser.parseTypeList(operandsTypes) ||
			parser.parseRParen() ||
			parser.resolveOperands(operands, operandsTypes, operandsLoc, result.operands) ||
			parser.parseArrow() ||
			parser.parseType(resultType))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void GetVariableOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " "
					<< userData() << ", " << index() << " : ("
					<< userData().getType() << ", "
					<< index().getType() << ")";
}

void GetVariableOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Read::get(), mlir::SideEffects::DefaultResource::get());
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
