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
// Ida::AllocIdaUserDataOp
//===----------------------------------------------------------------------===//

mlir::Value AllocIdaUserDataOpAdaptor::neq()
{
	return getValues()[0];
}

mlir::Value AllocIdaUserDataOpAdaptor::nnz()
{
	return getValues()[1];
}

void AllocIdaUserDataOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value neq, mlir::Value nnz)
{
	state.addTypes(OpaquePointerType::get(builder.getContext()));
	state.addOperands(neq);
	state.addOperands(nnz);
}

mlir::ParseResult AllocIdaUserDataOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void AllocIdaUserDataOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName()
					<< " " << neq() << ", " << nnz()
					<< " : (" << neq().getType() << ", " << nnz().getType()
					<< ") -> " << resultType();
}

void AllocIdaUserDataOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::SideEffects::DefaultResource::get());
}

mlir::Type AllocIdaUserDataOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::ValueRange AllocIdaUserDataOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value AllocIdaUserDataOp::neq()
{
	return Adaptor(*this).neq();
}

mlir::Value AllocIdaUserDataOp::nnz()
{
	return Adaptor(*this).nnz();
}
