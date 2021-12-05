#include <marco/mlirlowerer/dialects/ida/Attribute.h>
#include <marco/mlirlowerer/dialects/ida/IdaBuilder.h>
#include <marco/mlirlowerer/dialects/ida/Ops.h>
#include <marco/mlirlowerer/dialects/modelica/Attribute.h>
#include <marco/mlirlowerer/dialects/modelica/ModelicaBuilder.h>
#include <marco/mlirlowerer/dialects/modelica/Ops.h>
#include <marco/mlirlowerer/dialects/modelica/Traits.h>
#include <marco/mlirlowerer/dialects/modelica/Type.h>
#include <marco/mlirlowerer/passes/model/VectorAccess.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/FunctionImplementation.h>
#include <mlir/IR/OpImplementation.h>

using namespace marco::codegen;
using namespace ida;

namespace marco::codegen::ida
{
	static bool isNumeric(mlir::Type type)
	{
		return type.isa<IntegerType, RealType, modelica::IntegerType, modelica::RealType>();
	}

	static bool isNumeric(mlir::Value value)
	{
		return isNumeric(value.getType());
	}

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

	/**
	 * Writes inside a function how to compute the monodimensional offset of a
	 * variable needed by IDA, given the variable access and the indexes.
	 */
	static mlir::Value computeVariableOffset(
			modelica::ModelicaBuilder& builder,
			const model::Variable& variable,
			const model::Expression& expression,
			int64_t varOffset,
			mlir::BlockArgument indexes)
	{
		mlir::Location loc = expression.getOp()->getLoc();
		model::VectorAccess vectorAccess = model::AccessToVar::fromExp(expression).getAccess();
		marco::MultiDimInterval dimensions = variable.toMultiDimInterval();

		mlir::Value offset = builder.create<modelica::ConstantOp>(loc, builder.getIntegerAttribute(0));

		// For every dimension of the variable
		for (size_t i = 0; i < vectorAccess.size(); i++)
		{
			// Compute the offset of the current dimension.
			model::SingleDimensionAccess acc = vectorAccess[i];
			int64_t accOffset = acc.isDirectAccess() ? acc.getOffset() : (acc.getOffset() + 1);
			mlir::Value accessOffset = builder.create<modelica::ConstantOp>(loc, builder.getIntegerAttribute(accOffset));

			if (acc.isOffset())
			{
				// Add the offset that depends on the input indexes.
				mlir::Value indIndex = builder.create<modelica::ConstantOp>(loc, builder.getIntegerAttribute(acc.getInductionVar()));
				mlir::Value indValue = builder.create<LoadPointerOp>(loc, indexes, indIndex);
				accessOffset = builder.create<modelica::AddOp>(loc, accessOffset.getType(), accessOffset, indValue);
			}

			// Multiply the previous offset by the width of the current dimension.
			mlir::Value dimension = builder.create<modelica::ConstantOp>(loc, builder.getIntegerAttribute(dimensions[i].size()));
			offset = builder.create<modelica::MulOp>(loc, offset.getType(), offset, dimension);

			// Add the current dimension offset.
			offset = builder.create<modelica::AddOp>(loc, offset.getType(), offset, accessOffset);
		}

		// Add the offset from the start of the monodimensional variable array used by IDA.
		mlir::Value varOffsetValue = builder.create<modelica::ConstantOp>(loc, builder.getIntegerAttribute(varOffset));
		return builder.create<modelica::AddOp>(loc, offset.getType(), offset, varOffsetValue);
	}

	/**
	 * Writes inside a function how to compute the given expression starting from
	 * the given arguments (which are: time, vars, ders, indexes)
	 */
	static mlir::Value getFunction(
			modelica::ModelicaBuilder& builder,
			model::Model& model,
			const model::Expression& expression,
			OffsetMap offsetMap,
			llvm::ArrayRef<mlir::BlockArgument> args)
	{
		// Induction argument.
		if (expression.isInduction())
		{
			mlir::Location loc = expression.get<model::Induction>().getArgument().getLoc();
			unsigned int argNumber = expression.get<model::Induction>().getArgument().getArgNumber();
			mlir::Value indIndex = builder.create<modelica::ConstantOp>(loc, builder.getIntegerAttribute(argNumber));
			mlir::Value indValue = builder.create<LoadPointerOp>(loc, args[3], indIndex);

			// Add one because Modelica is 1-indexed.
			mlir::Value one = builder.create<modelica::ConstantOp>(loc, builder.getIntegerAttribute(1));
			return builder.create<modelica::AddOp>(loc, builder.getIntegerType(), indValue, one);
		}

		mlir::Operation* definingOp = expression.getOp();
		mlir::Location loc = definingOp->getLoc();

		// Constant value.
		if (mlir::isa<modelica::ConstantOp>(definingOp))
			return builder.clone(*definingOp)->getResult(0);

		// Variable reference.
		if (expression.isReferenceAccess())
		{
			model::Variable var = model.getVariable(expression.getReferredVectorAccess());

			// Time variable.
			if (var.isTime())
				return args[0];

			// Compute the IDA variable offset, which depends on the variable, the dimension and the access.
			mlir::Value varOffset = computeVariableOffset(builder, var, expression, offsetMap[var], args[3]);

			// Access and return the correct variable value.
			mlir::BlockArgument argArray = var.isDerivative() ? args[2] : args[1];
			return builder.create<LoadPointerOp>(loc, argArray, varOffset);
		}

		// Operation.
		assert(expression.isOperation());

		// Recursively compute and map the value of all the children.
		mlir::BlockAndValueMapping mapping;
		for (size_t i : marco::irange(expression.childrenCount()))
			mapping.map(
				expression.getOp()->getOperand(i),
				getFunction(builder, model, expression.getChild(i), offsetMap, args));

		// Add to the residual function and return the correct mapped operation.
		return builder.clone(*definingOp, mapping)->getResult(0);
	}

	/**
	 * Writes inside a function how to compute the derivative of the given
	 * expression starting from the given arguments (which are: time, vars, ders,
	 * indexes, derVar, alpha)
	 */
	static mlir::Value getDerFunction(
			modelica::ModelicaBuilder& builder,
			model::Model& model,
			const model::Expression& expression,
			OffsetMap offsetMap,
			llvm::ArrayRef<mlir::BlockArgument> args)
	{
		// Induction argument.
		if (expression.isInduction())
		{
			mlir::Location loc = expression.get<model::Induction>().getArgument().getLoc();
			return builder.create<modelica::ConstantOp>(loc, builder.getRealAttribute(0.0));
		}

		mlir::Operation* definingOp = expression.getOp();
		mlir::Location loc = definingOp->getLoc();

		// Constant value.
		if (mlir::isa<modelica::ConstantOp>(definingOp))
			return builder.create<modelica::ConstantOp>(loc, builder.getRealAttribute(0.0));

		// Variable reference.
		if (expression.isReferenceAccess())
		{
			model::Variable var = model.getVariable(expression.getReferredVectorAccess());

			// Time variable.
			if (var.isTime())
				return builder.create<modelica::ConstantOp>(loc, builder.getRealAttribute(0.0));

			// Compute the IDA variable offset, which depends on the variable, the dimension and the access.
			mlir::Value varOffset = computeVariableOffset(builder, var, expression, offsetMap[var], args[3]);

			// Check if the variable with respect to which we are currently derivating
			// is also the variable we are derivating.
			mlir::Value condition = builder.create<modelica::EqOp>(
					loc, builder.getBooleanType(), varOffset, args[5]);
			condition = builder.create<mlir::UnrealizedConversionCastOp>(loc, builder.getI1Type(), condition).getResult(0);

			// If yes, return alpha (if it is a derivative) or one (if it is a simple variable).
			mlir::Value thenValue = args[4];
			if (!var.isDerivative())
				thenValue = builder.create<modelica::ConstantOp>(loc, builder.getRealAttribute(1.0));

			// If no, return zero.
			mlir::Value elseValue = builder.create<modelica::ConstantOp>(loc, builder.getRealAttribute(0.0));
			return builder.create<mlir::SelectOp>(loc, builder.getRealType(), condition, thenValue, elseValue);
		}

		// Operation.
		assert(expression.isOperation());
		assert(definingOp->hasTrait<modelica::DerivativeInterface::Trait>());

		// Recursively compute and map the value of all the children.
		mlir::BlockAndValueMapping mapping;
		for (size_t i : marco::irange(expression.childrenCount()))
			mapping.map(
				expression.getOp()->getOperand(i),
				getFunction(builder, model, expression.getChild(i), offsetMap, args));

		// Clone the operation with the new operands.
		mlir::Operation* clonedOp = builder.clone(*definingOp, mapping);
		builder.setInsertionPoint(clonedOp);

		// Recursively compute and map the derivatives of all the children.
		mlir::BlockAndValueMapping derMapping;
		for (size_t i : marco::irange(expression.childrenCount()))
			derMapping.map(
				clonedOp->getOperand(i),
				getDerFunction(builder, model, expression.getChild(i), offsetMap, args));

		// Compute and return the derived operation.
		mlir::Value derivedOp = mlir::cast<modelica::DerivativeInterface>(clonedOp).derive(builder, derMapping).front();
		builder.setInsertionPointAfterValue(derivedOp);
		clonedOp->erase();
		return derivedOp;
	}
}

static void foldConstants(mlir::OpBuilder& builder, mlir::Block& block)
{
	llvm::SmallVector<mlir::Operation*, 3> operations;

	for (mlir::Operation& operation : block.getOperations())
		operations.push_back(&operation);

	// If an operation has only constants as operands, we can substitute it with
	// the corresponding constant value and erase the old operation.
	for (mlir::Operation* operation : operations)
		if (operation->hasTrait<modelica::FoldableOpInterface::Trait>())
			mlir::cast<modelica::FoldableOpInterface>(operation).foldConstants(builder);
}

static void cleanOperation(mlir::Block& block)
{
	llvm::SmallVector<mlir::Operation*, 3> operations;

	for (mlir::Operation& operation : block.getOperations())
		if (!mlir::isa<ida::FunctionTerminatorOp>(operation))
			operations.push_back(&operation);

	// If an operation has no uses, erase it.
	for (mlir::Operation* operation : llvm::reverse(operations))
		if (operation->use_empty())
			operation->erase();
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
// Ida::AllocDataOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> AllocDataOp::getAttributeNames()
{
	return {};
}

void AllocDataOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value equationsNumber)
{
	state.addTypes(OpaquePointerType::get(builder.getContext()));
	state.addOperands(equationsNumber);
}

mlir::ParseResult AllocDataOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return ida::parse(parser, result, 1);
}

void AllocDataOp::print(mlir::OpAsmPrinter& printer)
{
	ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult AllocDataOp::verify()
{
	if (!equationsNumber().getType().isa<IntegerType>())
		return emitOpError("Requires number of equations to be an integer");

	return mlir::success();
}

void AllocDataOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());
}

OpaquePointerType AllocDataOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<OpaquePointerType>();
}

mlir::ValueRange AllocDataOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value AllocDataOp::equationsNumber()
{
	return getOperation()->getOperand(0);
}

//===----------------------------------------------------------------------===//
// Ida::InitOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> InitOp::getAttributeNames()
{
	return {};
}

void InitOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value threads)
{
	state.addTypes(BooleanType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(threads);
}

mlir::ParseResult InitOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return ida::parse(parser, result, 2);
}

void InitOp::print(mlir::OpAsmPrinter& printer)
{
	ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult InitOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!threads().getType().isa<IntegerType>())
		return emitOpError("Requires number of threads to be an integer");

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

mlir::Value InitOp::threads()
{
	return getOperation()->getOperand(1);
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
	return ida::parse(parser, result, 1);
}

void StepOp::print(mlir::OpAsmPrinter& printer)
{
	ida::print(printer, getOperationName(), args(), resultType());
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
// Ida::FreeDataOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> FreeDataOp::getAttributeNames()
{
	return {};
}

void FreeDataOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData)
{
	state.addTypes(BooleanType::get(builder.getContext()));
	state.addOperands(userData);
}

mlir::ParseResult FreeDataOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return ida::parse(parser, result, 1);
}

void FreeDataOp::print(mlir::OpAsmPrinter& printer)
{
	ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult FreeDataOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	return mlir::success();
}

void FreeDataOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Free::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

BooleanType FreeDataOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<BooleanType>();
}

mlir::ValueRange FreeDataOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value FreeDataOp::userData()
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
	return ida::parse(parser, result, 3);
}

void AddTimeOp::print(mlir::OpAsmPrinter& printer)
{
	ida::print(printer, getOperationName(), args());
}

mlir::LogicalResult AddTimeOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!isNumeric(start()) || !isNumeric(end()) || !isNumeric(step()))
		return emitOpError("Requires start, end and step time to be numbers");

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
	return ida::parse(parser, result, 3);
}

void AddToleranceOp::print(mlir::OpAsmPrinter& printer)
{
	ida::print(printer, getOperationName(), args());
}

mlir::LogicalResult AddToleranceOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!isNumeric(relTol()) || !isNumeric(absTol()))
		return emitOpError("Requires relative and absolute tolerances to be numbers");

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
	return ida::parse(parser, result, 3);
}

void AddColumnIndexOp::print(mlir::OpAsmPrinter& printer)
{
	ida::print(printer, getOperationName(), args());
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
// Ida::AddEqDimensionOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> AddEqDimensionOp::getAttributeNames()
{
	return {};
}

void AddEqDimensionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value start, mlir::Value end)
{
	state.addOperands(userData);
	state.addOperands(start);
	state.addOperands(end);
}

mlir::ParseResult AddEqDimensionOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return ida::parse(parser, result, 3);
}

void AddEqDimensionOp::print(mlir::OpAsmPrinter& printer)
{
	ida::print(printer, getOperationName(), args());
}

mlir::LogicalResult AddEqDimensionOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!start().getType().isa<modelica::ArrayType>() ||
			!start().getType().cast<modelica::ArrayType>().getElementType().isa<modelica::IntegerType>())
		return emitOpError("Requires start iteration indexes to be arrays of integers");

	if (!end().getType().isa<modelica::ArrayType>() ||
			!end().getType().cast<modelica::ArrayType>().getElementType().isa<modelica::IntegerType>())
		return emitOpError("Requires end iteration indexes to be arrays of integers");

	return mlir::success();
}

void AddEqDimensionOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

mlir::ValueRange AddEqDimensionOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value AddEqDimensionOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value AddEqDimensionOp::start()
{
	return getOperation()->getOperand(1);
}

mlir::Value AddEqDimensionOp::end()
{
	return getOperation()->getOperand(2);
}

//===----------------------------------------------------------------------===//
// Ida::AddResidualOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> AddResidualOp::getAttributeNames()
{
	return {};
}

void AddResidualOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value residualAddress)
{
	state.addOperands(userData);
	state.addOperands(residualAddress);
}

mlir::ParseResult AddResidualOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return ida::parse(parser, result, 2);
}

void AddResidualOp::print(mlir::OpAsmPrinter& printer)
{
	ida::print(printer, getOperationName(), args());
}

mlir::LogicalResult AddResidualOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!residualAddress().getType().isa<mlir::LLVM::LLVMPointerType>())
		return emitOpError("Requires residual function address to be a pointer");

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

mlir::Value AddResidualOp::residualAddress()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::AddJacobianOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> AddJacobianOp::getAttributeNames()
{
	return {};
}

void AddJacobianOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value jacobianAddress)
{
	state.addOperands(userData);
	state.addOperands(jacobianAddress);
}

mlir::ParseResult AddJacobianOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return ida::parse(parser, result, 2);
}

void AddJacobianOp::print(mlir::OpAsmPrinter& printer)
{
	ida::print(printer, getOperationName(), args());
}

mlir::LogicalResult AddJacobianOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!jacobianAddress().getType().isa<mlir::LLVM::LLVMPointerType>())
		return emitOpError("Requires residual function address to be a pointer");

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

mlir::Value AddJacobianOp::jacobianAddress()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::AddVariableOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> AddVariableOp::getAttributeNames()
{
	return {};
}

void AddVariableOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value index, mlir::Value array, mlir::Value isState)
{
	state.addOperands(userData);
	state.addOperands(index);
	state.addOperands(array);
	state.addOperands(isState);
}

mlir::ParseResult AddVariableOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return ida::parse(parser, result, 4);
}

void AddVariableOp::print(mlir::OpAsmPrinter& printer)
{
	ida::print(printer, getOperationName(), args());
}

mlir::LogicalResult AddVariableOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!index().getType().isa<IntegerType>())
		return emitOpError("Requires variable index to be an integer");

	if (!array().getType().isa<modelica::ArrayType>())
		return emitOpError("Requires initialization array to be an array");

	if (!isState().getType().isa<BooleanType>())
		return emitOpError("Requires variable state to be a boolean");

	return mlir::success();
}

void AddVariableOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Read::get(), userData(), mlir::SideEffects::DefaultResource::get());
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

mlir::ValueRange AddVariableOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value AddVariableOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value AddVariableOp::index()
{
	return getOperation()->getOperand(1);
}

mlir::Value AddVariableOp::array()
{
	return getOperation()->getOperand(2);
}

mlir::Value AddVariableOp::isState()
{
	return getOperation()->getOperand(3);
}

//===----------------------------------------------------------------------===//
// Ida::GetVariableAllocOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> GetVariableAllocOp::getAttributeNames()
{
	return {};
}

void GetVariableAllocOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value offset, mlir::Value isDer, mlir::Type returnType)
{
	state.addTypes(returnType);
	state.addOperands(userData);
	state.addOperands(offset);
	state.addOperands(isDer);
}

mlir::ParseResult GetVariableAllocOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return ida::parse(parser, result, 3);
}

void GetVariableAllocOp::print(mlir::OpAsmPrinter& printer)
{
	ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult GetVariableAllocOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!offset().getType().isa<IntegerType>())
		return emitOpError("Requires variable offset to be an integer");

	if (!isDer().getType().isa<BooleanType>())
		return emitOpError("Requires isDerivative to be a boolean");

	if (!resultType().getElementType().isa<modelica::RealType>())
		return emitOpError("Requires variable passed to IDA to be real numbers");

	return mlir::success();
}

void GetVariableAllocOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Read::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

modelica::ArrayType GetVariableAllocOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<modelica::ArrayType>();
}

mlir::ValueRange GetVariableAllocOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value GetVariableAllocOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value GetVariableAllocOp::offset()
{
	return getOperation()->getOperand(1);
}

mlir::Value GetVariableAllocOp::isDer()
{
	return getOperation()->getOperand(2);
}

//===----------------------------------------------------------------------===//
// Ida::AddVarAccessOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> AddVarAccessOp::getAttributeNames()
{
	return {};
}

void AddVarAccessOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value variable, mlir::Value offsets, mlir::Value inductions)
{
	state.addTypes(IntegerType::get(builder.getContext()));
	state.addOperands(userData);
	state.addOperands(variable);
	state.addOperands(offsets);
	state.addOperands(inductions);
}

mlir::ParseResult AddVarAccessOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return ida::parse(parser, result, 4);
}

void AddVarAccessOp::print(mlir::OpAsmPrinter& printer)
{
	ida::print(printer, getOperationName(), args(), resultType());
}

mlir::LogicalResult AddVarAccessOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	if (!variable().getType().isa<IntegerType>())
		return emitOpError("Requires variable index to be an integer");

	if (!offsets().getType().isa<modelica::ArrayType>() ||
			!offsets().getType().cast<modelica::ArrayType>().getElementType().isa<modelica::IntegerType>())
		return emitOpError("Requires variable offsets to be arrays of integers");

	if (!inductions().getType().isa<modelica::ArrayType>() ||
			!inductions().getType().cast<modelica::ArrayType>().getElementType().isa<modelica::IntegerType>())
		return emitOpError("Requires variable inductions to be arrays of integers");

	return mlir::success();
}

void AddVarAccessOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), userData(), mlir::SideEffects::DefaultResource::get());
}

IntegerType AddVarAccessOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<IntegerType>();
}

mlir::ValueRange AddVarAccessOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value AddVarAccessOp::userData()
{
	return getOperation()->getOperand(0);
}

mlir::Value AddVarAccessOp::variable()
{
	return getOperation()->getOperand(1);
}

mlir::Value AddVarAccessOp::offsets()
{
	return getOperation()->getOperand(2);
}

mlir::Value AddVarAccessOp::inductions()
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
	state.addTypes(modelica::RealType::get(builder.getContext()));
	state.addOperands(userData);
}

mlir::ParseResult GetTimeOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return ida::parse(parser, result, 1);
}

void GetTimeOp::print(mlir::OpAsmPrinter& printer)
{
	ida::print(printer, getOperationName(), args(), resultType());
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

modelica::RealType GetTimeOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<modelica::RealType>();
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
// Ida::ResidualFunctionOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> ResidualFunctionOp::getAttributeNames()
{
	return {};
}

void ResidualFunctionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, llvm::StringRef name, model::Model& model, model::Equation& equation, OffsetMap offsetMap)
{
	state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));

	// residualFunction(real time, real* variables, real* derivatives, int* indexes) -> real
	llvm::SmallVector<mlir::Type, 4> argTypes = {
		modelica::RealType::get(builder.getContext()),
		RealPointerType::get(builder.getContext()), 
		RealPointerType::get(builder.getContext()),
		IntegerPointerType::get(builder.getContext())
	};
	mlir::Type returnType = { modelica::RealType::get(builder.getContext()) };
	state.addAttribute(getTypeAttrName(), mlir::TypeAttr::get(builder.getFunctionType(argTypes, returnType)));

	mlir::Region* entryRegion = state.addRegion();
	mlir::Block& entryBlock = entryRegion->emplaceBlock();
	entryBlock.addArguments(argTypes);

	// Fill the only block of the function with how to compute the Residual of the given Equation.
	modelica::ModelicaBuilder modelicaBuilder(builder.getContext());
	modelicaBuilder.setInsertionPointToStart(&entryBlock);

	mlir::Value lhsResidual = getFunction(modelicaBuilder, model, equation.lhs(), offsetMap, entryBlock.getArguments());
	mlir::Value rhsResidual = getFunction(modelicaBuilder, model, equation.rhs(), offsetMap, entryBlock.getArguments());

	mlir::Value returnValue = modelicaBuilder.create<modelica::SubOp>(equation.getOp().getLoc(), modelicaBuilder.getRealType(), rhsResidual, lhsResidual);
	modelicaBuilder.create<ida::FunctionTerminatorOp>(equation.getOp().getLoc(), returnValue);

	// Fold the constants and clean the unused operations.
	foldConstants(builder, entryBlock);
	cleanOperation(entryBlock);
}

mlir::ParseResult ResidualFunctionOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::Builder& builder = parser.getBuilder();
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> args;
	llvm::SmallVector<mlir::Type, 3> argsTypes;
	mlir::Type resultType;

	mlir::StringAttr nameAttr;
	if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes))
		return mlir::failure();

	if (parser.parseArrow() || parser.parseType(resultType))
		return mlir::failure();

	mlir::FunctionType functionType = builder.getFunctionType(argsTypes, resultType);
	result.addAttribute(getTypeAttrName(), mlir::TypeAttr::get(functionType));

	mlir::Region* region = result.addRegion();
	if (parser.parseRegion(*region, args, argsTypes))
		return mlir::failure();

	return mlir::success();
}

void ResidualFunctionOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " @" << name() << "(";

	auto args = getArguments();

	for (const auto& arg : llvm::enumerate(args))
	{
		if (arg.index() > 0)
			printer << ", ";

		printer << arg.value() << " : " << arg.value().getType();
	}

	printer << ") -> " << getType().getResult(0);

	printer.printRegion(getBody(), false);
}

mlir::LogicalResult ResidualFunctionOp::verify()
{
	if (getNumArguments() != 4)
		return emitOpError("Requires to have exactly four arguments (tt, yy, yp, ind)");

	if (getNumResults() != 1 || !getType().getResult(0).isa<modelica::RealType>())
		return emitOpError("Requires to have exactly one result");

	return mlir::success();
}

unsigned int ResidualFunctionOp::getNumFuncArguments()
{
	return getType().getInputs().size();
}

unsigned int ResidualFunctionOp::getNumFuncResults()
{
	return getType().getResults().size();
}

mlir::Region* ResidualFunctionOp::getCallableRegion()
{
	return &getBody();
}

llvm::ArrayRef<mlir::Type> ResidualFunctionOp::getCallableResults()
{
	return getType().getResults();
}

llvm::StringRef ResidualFunctionOp::name()
{
	return getOperation()->getAttrOfType<mlir::StringAttr>(mlir::SymbolTable::getSymbolAttrName()).getValue();
}


//===----------------------------------------------------------------------===//
// Ida::JacobianFunctionOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> JacobianFunctionOp::getAttributeNames()
{
	return {};
}

void JacobianFunctionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, llvm::StringRef name, model::Model& model, model::Equation& equation, OffsetMap offsetMap)
{
	state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));

	// jacobianFunction(real time, real* variables, real* derivatives, int* indexes, real alpha, real der_var) -> real
	llvm::SmallVector<mlir::Type, 6> argTypes = {
		modelica::RealType::get(builder.getContext()),
		RealPointerType::get(builder.getContext()), 
		RealPointerType::get(builder.getContext()),
		IntegerPointerType::get(builder.getContext()),
		modelica::RealType::get(builder.getContext()),
		modelica::IntegerType::get(builder.getContext())
	};
	mlir::Type returnType = { modelica::RealType::get(builder.getContext()) };
	state.addAttribute(getTypeAttrName(), mlir::TypeAttr::get(builder.getFunctionType(argTypes, returnType)));

	mlir::Region* entryRegion = state.addRegion();
	mlir::Block& entryBlock = entryRegion->emplaceBlock();
	entryBlock.addArguments(argTypes);

	// Fill the only block of the function with how to compute the Residual of the given Equation.
	modelica::ModelicaBuilder modelicaBuilder(builder.getContext());
	modelicaBuilder.setInsertionPointToStart(&entryBlock);

	mlir::Value lhsJacobian = getDerFunction(modelicaBuilder, model, equation.lhs(), offsetMap, entryBlock.getArguments());
	mlir::Value rhsJacobian = getDerFunction(modelicaBuilder, model, equation.rhs(), offsetMap, entryBlock.getArguments());

	mlir::Value returnValue = modelicaBuilder.create<modelica::SubOp>(equation.getOp().getLoc(), modelicaBuilder.getRealType(), rhsJacobian, lhsJacobian);
	modelicaBuilder.create<ida::FunctionTerminatorOp>(equation.getOp().getLoc(), returnValue);

	// Fold the constants and clean the unused operations.
	foldConstants(builder, entryBlock);
	cleanOperation(entryBlock);
}

mlir::ParseResult JacobianFunctionOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::Builder& builder = parser.getBuilder();
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> args;
	llvm::SmallVector<mlir::Type, 3> argsTypes;
	mlir::Type resultType;

	mlir::StringAttr nameAttr;
	if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes))
		return mlir::failure();

	if (parser.parseArrow() || parser.parseType(resultType))
		return mlir::failure();

	mlir::FunctionType functionType = builder.getFunctionType(argsTypes, resultType);
	result.addAttribute(getTypeAttrName(), mlir::TypeAttr::get(functionType));

	mlir::Region* region = result.addRegion();
	if (parser.parseRegion(*region, args, argsTypes))
		return mlir::failure();

	return mlir::success();
}

void JacobianFunctionOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " @" << name() << "(";

	auto args = getArguments();

	for (const auto& arg : llvm::enumerate(args))
	{
		if (arg.index() > 0)
			printer << ", ";

		printer << arg.value() << " : " << arg.value().getType();
	}

	printer << ") -> " << getType().getResult(0);

	printer.printRegion(getBody(), false);
}

mlir::LogicalResult JacobianFunctionOp::verify()
{
	if (getNumArguments() != 6)
		return emitOpError("Requires to have exactly four arguments (tt, yy, yp, ind, cj, var)");

	if (getNumResults() != 1 || !getType().getResult(0).isa<modelica::RealType>())
		return emitOpError("Requires to have exactly one result");

	return mlir::success();
}

unsigned int JacobianFunctionOp::getNumFuncArguments()
{
	return getType().getInputs().size();
}

unsigned int JacobianFunctionOp::getNumFuncResults()
{
	return getType().getResults().size();
}

mlir::Region* JacobianFunctionOp::getCallableRegion()
{
	return &getBody();
}

llvm::ArrayRef<mlir::Type> JacobianFunctionOp::getCallableResults()
{
	return getType().getResults();
}

llvm::StringRef JacobianFunctionOp::name()
{
	return getOperation()->getAttrOfType<mlir::StringAttr>(mlir::SymbolTable::getSymbolAttrName()).getValue();
}

//===----------------------------------------------------------------------===//
// Ida::FunctionTerminatorOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> FunctionTerminatorOp::getAttributeNames()
{
	return {};
}

void FunctionTerminatorOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value returnValue)
{
	state.addOperands(returnValue);
}

mlir::ParseResult FunctionTerminatorOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::OpAsmParser::OperandType operand;
	mlir::Type operandType;

	llvm::SMLoc operandsLoc = parser.getCurrentLocation();

	if (parser.parseOperand(operand) ||
			parser.parseColon() ||
			parser.parseType(operandType) ||
			parser.resolveOperands({ operand }, operandType, operandsLoc, result.operands))
		return mlir::failure();

	return mlir::success();
}

void FunctionTerminatorOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName()
			<< " " << returnValue()
			<< " : " << returnValue().getType();
}

mlir::LogicalResult FunctionTerminatorOp::verify()
{
	if (!returnValue().getType().isa<RealType>() && !returnValue().getType().isa<modelica::RealType>())
		return emitOpError("Requires return value to be a real number");

	return mlir::success();
}

mlir::ValueRange FunctionTerminatorOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value FunctionTerminatorOp::returnValue()
{
	return getOperation()->getOperand(0);
}

//===----------------------------------------------------------------------===//
// Ida::FuncAddressOfOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> FuncAddressOfOp::getAttributeNames()
{
	static llvm::StringRef attrNames[] = {llvm::StringRef("callee")};
	return llvm::makeArrayRef(attrNames);
}

void FuncAddressOfOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::StringRef callee, mlir::Type type)
{
	state.addTypes(type);
	state.addAttribute("callee", builder.getSymbolRefAttr(callee));
}

mlir::ParseResult FuncAddressOfOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void FuncAddressOfOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " @" << callee() << " : " << resultType();
}

mlir::LogicalResult FuncAddressOfOp::verify()
{
	return mlir::success();
}

mlir::Type FuncAddressOfOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::StringRef FuncAddressOfOp::callee()
{
	return getOperation()->getAttrOfType<mlir::FlatSymbolRefAttr>("callee").getValue();
}

//===----------------------------------------------------------------------===//
// Ida::LoadPointerOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> LoadPointerOp::getAttributeNames()
{
	return {};
}

void LoadPointerOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value pointer, mlir::Value offset)
{
	if (pointer.getType().isa<IntegerPointerType>())
		state.addTypes(modelica::IntegerType::get(builder.getContext()));
	else if (pointer.getType().isa<RealPointerType>())
		state.addTypes(modelica::RealType::get(builder.getContext()));

	state.addOperands(pointer);
	state.addOperands(offset);
}

mlir::ParseResult LoadPointerOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::OpAsmParser::OperandType pointer;
	mlir::OpAsmParser::OperandType offset;
	llvm::SmallVector<mlir::Type, 3> operandTypes;
	llvm::SmallVector<mlir::Type, 1> resultTypes;

	llvm::SMLoc operandsLoc = parser.getCurrentLocation();

	if (parser.parseOperand(pointer) ||
			parser.parseLSquare() ||
			parser.parseOperand(offset) ||
			parser.parseRSquare() ||
			parser.parseColon() ||
			parser.parseLParen() ||
			parser.parseTypeList(operandTypes) ||
			parser.parseRParen() ||
			parser.resolveOperands({ pointer, offset }, operandTypes, operandsLoc, result.operands) ||
			parser.parseOptionalArrowTypeList(resultTypes))
		return mlir::failure();

	result.addTypes(resultTypes);

	return mlir::success();
}

void LoadPointerOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName()
			<< " " << pointer() << "[" << offset() << "]"
			<< " : (" << pointer().getType() << ", " << offset().getType()
			<< ") -> " << resultType();
}

mlir::LogicalResult LoadPointerOp::verify()
{
	if (!pointer().getType().isa<IntegerPointerType>() && !pointer().getType().isa<RealPointerType>())
		return emitOpError("Requires pointer to be a integer or real pointer");

	if (!offset().getType().isa<IntegerType>() && !offset().getType().isa<modelica::IntegerType>())
		return emitOpError("Requires offset to be an integer");

	return mlir::success();
}

void LoadPointerOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Read::get(), pointer(), mlir::SideEffects::DefaultResource::get());
}

mlir::Value LoadPointerOp::getViewSource()
{
	return pointer();
}

mlir::Type LoadPointerOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::ValueRange LoadPointerOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value LoadPointerOp::pointer()
{
	return getOperation()->getOperand(0);
}

mlir::Value LoadPointerOp::offset()
{
	return getOperation()->getOperand(1);
}

//===----------------------------------------------------------------------===//
// Ida::PrintStatisticsOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> PrintStatisticsOp::getAttributeNames()
{
	return {};
}

void PrintStatisticsOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData)
{
	state.addOperands(userData);
}

mlir::ParseResult PrintStatisticsOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return ida::parse(parser, result, 1);
}

void PrintStatisticsOp::print(mlir::OpAsmPrinter& printer)
{
	ida::print(printer, getOperationName(), args());
}

mlir::LogicalResult PrintStatisticsOp::verify()
{
	if (!userData().getType().isa<OpaquePointerType>())
		return emitOpError("Requires user data to be an opaque pointer");

	return mlir::success();
}

mlir::ValueRange PrintStatisticsOp::args()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

mlir::Value PrintStatisticsOp::userData()
{
	return getOperation()->getOperand(0);
}
