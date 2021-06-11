#include <mlir/Conversion/Passes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/FunctionImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <modelica/mlirlowerer/Attribute.h>
#include <modelica/mlirlowerer/Ops.h>

using namespace modelica::codegen;

static bool isNumeric(mlir::Type type)
{
	return type.isa<mlir::IndexType, BooleanType, IntegerType, RealType>();
}

static bool isNumeric(mlir::Value value)
{
	return isNumeric(value.getType());
}

static mlir::Value readValue(mlir::OpBuilder& builder, mlir::Value operand)
{
	if (auto pointerType = operand.getType().dyn_cast<PointerType>(); pointerType && pointerType.getRank() == 0)
		return builder.create<LoadOp>(operand.getLoc(), operand);

	return operand;
}

//===----------------------------------------------------------------------===//
// Modelica::PackOp
//===----------------------------------------------------------------------===//

mlir::ValueRange PackOpAdaptor::values()
{
	return getValues();
}

void PackOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange values)
{
	llvm::SmallVector<mlir::Type, 3> types;

	for (mlir::Value value : values)
		types.push_back(value.getType());

	state.addTypes(StructType::get(state.getContext(), types));
	state.addOperands(values);
}

mlir::ParseResult PackOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> values;
	llvm::SmallVector<mlir::Type, 3> types;
	mlir::Type resultType;

	llvm::SMLoc valuesLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(values) ||
			parser.parseColon())
		return mlir::failure();

	if (values.size() == 1)
	{
		if (parser.parseTypeList(types))
			return mlir::failure();
	}
	else if (values.size() > 1)
	{
		if (parser.parseLParen() ||
				parser.parseTypeList(types) ||
				parser.parseRParen())
			return mlir::failure();
	}

	if (!values.empty())
		if (parser.parseArrow())
			return mlir::failure();

	if (parser.parseType(resultType) ||
			parser.resolveOperands(values, types, valuesLoc, result.operands))
		return mlir::failure();

	return mlir::success();
}

void PackOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.pack " << values() << " : ";

	if (values().size() > 1)
		printer << "(";

	printer << values();

	if (values().size() > 1)
		printer << ")";

	if (!values().empty())
		printer << " -> ";

	printer << resultType();
}

StructType PackOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<StructType>();
}

mlir::ValueRange PackOp::values()
{
	return Adaptor(*this).values();
}

//===----------------------------------------------------------------------===//
// Modelica::ExtractOp
//===----------------------------------------------------------------------===//

mlir::Value ExtractOpAdaptor::packedValue()
{
	return getValues()[0];
}

unsigned int ExtractOpAdaptor::index()
{
	return getAttrs().getAs<mlir::IntegerAttr>("index").getInt();
}

void ExtractOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value packedValue, unsigned int index)
{
	state.addTypes(resultType);
	state.addOperands(packedValue);
	state.addAttribute("index", builder.getIndexAttr(index));
}

void ExtractOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.extract "
					<< packedValue() << getOperation()->getAttrDictionary()
					<< " : (" << packedValue().getType();
}

mlir::LogicalResult ExtractOp::verify()
{
	if (!packedValue().getType().isa<StructType>())
		return emitOpError("requires the operand to be a struct");

	if (auto structType = packedValue().getType().cast<StructType>(); index() >= structType.getElementTypes().size())
		return emitOpError("has an out of bounds index");

	return mlir::success();
}

mlir::Type ExtractOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value ExtractOp::packedValue()
{
	return Adaptor(*this).packedValue();
}

unsigned int ExtractOp::index()
{
	return Adaptor(*this).index();
}

//===----------------------------------------------------------------------===//
// Modelica::SimulationOp
//===----------------------------------------------------------------------===//

void SimulationOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, RealAttribute startTime, RealAttribute endTime, RealAttribute timeStep, mlir::TypeRange vars)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	state.addAttribute("startTime", startTime);
	state.addAttribute("endTime", endTime);
	state.addAttribute("timeStep", timeStep);

	// Init block
	builder.createBlock(state.addRegion());

	// Body block
	builder.createBlock(state.addRegion(), {}, vars);

	// Print block
	builder.createBlock(state.addRegion(), {}, vars);
}

void SimulationOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.simulation ("
					<< "start: " << startTime().getValue()
					<< ", end: " << endTime().getValue()
					<< ", step: " << timeStep().getValue() << ")";

	printer << " init";
	printer.printRegion(init(), false);

	printer << " step";
	printer.printRegion(body(), true);

	printer << " print";
	printer.printRegion(print(), true);
}

mlir::LogicalResult SimulationOp::verify()
{
	// TODO
	return mlir::success();
}

void SimulationOp::getSuccessorRegions(llvm::Optional<unsigned int> index, llvm::ArrayRef<mlir::Attribute> operands, llvm::SmallVectorImpl<mlir::RegionSuccessor>& regions)
{
	if (!index.hasValue())
	{
		regions.emplace_back(&body(), body().getArguments());
		return;
	}

	assert(*index < 1 && "There is only one region in a SimulationOp");

	if (*index == 0)
	{
		regions.emplace_back(&body(), body().getArguments());
		regions.emplace_back(getOperation()->getResults());
	}
}

RealAttribute SimulationOp::startTime()
{
	return getOperation()->getAttrOfType<RealAttribute>("startTime");
}

RealAttribute SimulationOp::endTime()
{
	return getOperation()->getAttrOfType<RealAttribute>("endTime");
}

RealAttribute SimulationOp::timeStep()
{
	return getOperation()->getAttrOfType<RealAttribute>("timeStep");
}

mlir::Region& SimulationOp::init()
{
	return getOperation()->getRegion(0);
}

mlir::Region& SimulationOp::body()
{
	return getOperation()->getRegion(1);
}

mlir::Region& SimulationOp::print()
{
	return getOperation()->getRegion(2);
}

mlir::Value SimulationOp::getVariableAllocation(mlir::Value var)
{
	unsigned int index = var.dyn_cast<mlir::BlockArgument>().getArgNumber();
	return mlir::cast<YieldOp>(init().back().getTerminator()).getOperand(index);
}

mlir::Value SimulationOp::time()
{
	return body().getArgument(0);
}

//===----------------------------------------------------------------------===//
// Modelica::EquationOp
//===----------------------------------------------------------------------===//

void EquationOp::build(mlir::OpBuilder& builder, mlir::OperationState& state)
{
	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.createBlock(state.addRegion());
}

mlir::ParseResult EquationOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::Region* body = result.addRegion();

	if (parser.parseRegion(*body))
		return mlir::failure();

	return mlir::success();
}

void EquationOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName();
	printer.printRegion(getRegion(), false);
}

mlir::Block* EquationOp::body()
{
	return &getRegion().front();
}

mlir::ValueRange EquationOp::inductions()
{
	return mlir::ValueRange();
}

mlir::Value EquationOp::induction(size_t index)
{
	assert(false && "EquationOp doesn't have induction variables");
	return mlir::Value();
}

long EquationOp::inductionIndex(mlir::Value induction)
{
	assert(false && "EquationOp doesn't have induction variables");
	return -1;
}

mlir::ValueRange EquationOp::lhs()
{
	auto terminator = mlir::cast<EquationSidesOp>(body()->getTerminator());
	return terminator.lhs();
}

mlir::ValueRange EquationOp::rhs()
{
	auto terminator = mlir::cast<EquationSidesOp>(body()->getTerminator());
	return terminator.rhs();
}

//===----------------------------------------------------------------------===//
// Modelica::ForEquationOp
//===----------------------------------------------------------------------===//

void ForEquationOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, size_t inductionsAmount)
{
	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.createBlock(state.addRegion());

	llvm::SmallVector<mlir::Type, 3> inductionsTypes(inductionsAmount, builder.getIndexType());
	builder.createBlock(state.addRegion(), {}, inductionsTypes);
}

void ForEquationOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.for_equation";
	printer.printRegion(getRegion(0), false);
	printer << " body";
	printer.printRegion(getRegion(1), true);
}

mlir::LogicalResult ForEquationOp::verify()
{
	for (auto value : inductionsDefinitions())
		if (!mlir::isa<InductionOp>(value.getDefiningOp()))
			return emitOpError("requires the inductions to be defined by InductionOp operations");

	return mlir::success();
}

mlir::Block* ForEquationOp::inductionsBlock()
{
	return &getRegion(0).front();
}

mlir::ValueRange ForEquationOp::inductionsDefinitions()
{
	return mlir::cast<YieldOp>(inductionsBlock()->getTerminator()).getOperands();
}

mlir::Block* ForEquationOp::body()
{
	return &getRegion(1).front();
}

mlir::ValueRange ForEquationOp::inductions()
{
	return body()->getArguments();
}

mlir::Value ForEquationOp::induction(size_t index)
{
	auto allInductions = inductions();
	assert(index < allInductions.size());
	return allInductions[index];
}

long ForEquationOp::inductionIndex(mlir::Value induction)
{
	assert(induction.isa<mlir::BlockArgument>());

	for (auto ind : llvm::enumerate(body()->getArguments()))
		if (ind.value() == induction)
			return ind.index();

	assert(false && "Induction variable not found");
	return -1;
}

mlir::ValueRange ForEquationOp::lhs()
{
	auto terminator = mlir::cast<EquationSidesOp>(body()->getTerminator());
	return terminator.lhs();
}

mlir::ValueRange ForEquationOp::rhs()
{
	auto terminator = mlir::cast<EquationSidesOp>(body()->getTerminator());
	return terminator.rhs();
}

//===----------------------------------------------------------------------===//
// Modelica::InductionOp
//===----------------------------------------------------------------------===//

long InductionOpAdaptor::start()
{
	return getAttrs().getAs<mlir::IntegerAttr>("start").getInt();
}

long InductionOpAdaptor::end()
{
	return getAttrs().getAs<mlir::IntegerAttr>("end").getInt();
}

void InductionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, long start, long end)
{
	state.addAttribute("start", builder.getI64IntegerAttr(start));
	state.addAttribute("end", builder.getI64IntegerAttr(end));

	state.addTypes(builder.getIndexType());
}

mlir::ParseResult InductionOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	llvm::SMLoc loc = parser.getCurrentLocation();
	mlir::NamedAttrList attributes;

	if (parser.parseOptionalAttrDict(attributes))
		return mlir::failure();

	result.attributes.append(attributes);

	if (!result.attributes.getNamed("start").hasValue())
		return parser.emitError(loc, "expected start value");

	if (!result.attributes.getNamed("end").hasValue())
		return parser.emitError(loc, "expected end value");

	return mlir::success();
}

void InductionOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << getOperation()->getAttrDictionary();
}

long InductionOp::start()
{
	return Adaptor(*this).start();
}

long InductionOp::end()
{
	return Adaptor(*this).end();
}

//===----------------------------------------------------------------------===//
// Modelica::EquationSidesOp
//===----------------------------------------------------------------------===//

mlir::ValueRange EquationSidesOpAdaptor::lhs()
{
	auto amount = getAttrs().get("lhs").cast<mlir::IntegerAttr>().getInt();
	return mlir::ValueRange(getValues().begin(), std::next(getValues().begin(), amount));
}

mlir::ValueRange EquationSidesOpAdaptor::rhs()
{
	auto amount = getAttrs().get("rhs").cast<mlir::IntegerAttr>().getInt();
	return mlir::ValueRange(std::next(getValues().begin(), getValues().size() - amount), getValues().end());
}

void EquationSidesOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange lhs, mlir::ValueRange rhs)
{
	state.addAttribute("lhs", builder.getI64IntegerAttr(lhs.size()));
	state.addAttribute("rhs", builder.getI64IntegerAttr(rhs.size()));

	state.addOperands(lhs);
	state.addOperands(rhs);
}

mlir::ParseResult EquationSidesOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	auto& builder = parser.getBuilder();

	llvm::SmallVector<mlir::OpAsmParser::OperandType, 1> lhs;
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 1> rhs;

	llvm::SmallVector<mlir::Type, 1> lhsTypes;
	llvm::SmallVector<mlir::Type, 1> rhsTypes;

	llvm::SMLoc lhsLocation = parser.getCurrentLocation();

	if (parser.parseOperandList(lhs, mlir::OpAsmParser::Delimiter::Paren))
		return mlir::failure();

	llvm::SMLoc rhsLocation = parser.getCurrentLocation();

	if (parser.parseOperandList(rhs, mlir::OpAsmParser::Delimiter::Paren))
		return mlir::failure();

	if (parser.parseColon() ||
			parser.parseLParen() ||
			parser.parseTypeList(lhsTypes) ||
			parser.parseRParen() ||
			parser.parseLParen() ||
			parser.parseTypeList(rhsTypes) ||
			parser.parseRParen())
		return mlir::failure();


	if (parser.resolveOperands(lhs, lhsTypes, lhsLocation, result.operands) ||
			parser.resolveOperands(rhs, rhsTypes, rhsLocation, result.operands))
		return mlir::failure();

	result.addAttribute("lhs", builder.getI64IntegerAttr(lhs.size()));
	result.addAttribute("rhs", builder.getI64IntegerAttr(rhs.size()));

	return mlir::success();
}

void EquationSidesOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName()
					<< " (" << lhs() << ") (" << rhs() << ")"
					<< " : (" << lhs().getTypes() << ") (" << rhs().getTypes() << ")";
}

mlir::ValueRange EquationSidesOp::lhs()
{
	return Adaptor(*this).lhs();
}

mlir::ValueRange EquationSidesOp::rhs()
{
	return Adaptor(*this).rhs();
}

//===----------------------------------------------------------------------===//
// Modelica::FunctionOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<mlir::Attribute> FunctionOpAdaptor::argsNames()
{
	return getAttrs().getAs<mlir::ArrayAttr>("args_names").getValue();
}

llvm::ArrayRef<mlir::Attribute> FunctionOpAdaptor::resultsNames()
{
	return getAttrs().getAs<mlir::ArrayAttr>("results_names").getValue();
}

void FunctionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, llvm::StringRef name, mlir::FunctionType type, llvm::ArrayRef<llvm::StringRef> argsNames, llvm::ArrayRef<llvm::StringRef> resultsNames)
{
	state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
	state.addAttribute(getTypeAttrName(), mlir::TypeAttr::get(type));

	state.addAttribute("args_names", builder.getStrArrayAttr(argsNames));
	state.addAttribute("results_names", builder.getStrArrayAttr(resultsNames));

	state.addRegion();
}

mlir::ParseResult FunctionOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	auto& builder = parser.getBuilder();
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> args;
	llvm::SmallVector<mlir::Type, 3> argsTypes;

	llvm::SmallVector<mlir::Type, 3> resultsTypes;

	mlir::StringAttr nameAttr;
	if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes))
		return mlir::failure();

	llvm::SmallVector<mlir::NamedAttrList> argsAttrs;
	bool isVariadic = false;

	// TODO: don't rely on FunctionLike parsing
	if (mlir::impl::parseFunctionArgumentList(parser, false, false, args, argsTypes, argsAttrs, isVariadic))
		return mlir::failure();

	if (parser.parseArrow() ||
			parser.parseLParen())
		return mlir::failure();

	if (mlir::failed(parser.parseOptionalRParen()))
	{
		if (parser.parseTypeList(resultsTypes) ||
				parser.parseRParen())
			return mlir::failure();
	}

	mlir::FunctionType functionType = builder.getFunctionType(argsTypes, resultsTypes);
	result.addAttribute(getTypeAttrName(), mlir::TypeAttr::get(functionType));

	mlir::NamedAttrList attributes;

	if (parser.parseOptionalAttrDictWithKeyword(attributes))
		return mlir::failure();

	result.attributes.append(attributes);

	mlir::Region* region = result.addRegion();

	if (parser.parseRegion(*region, args, argsTypes))
		return mlir::failure();

	return mlir::success();
}

void FunctionOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " @" << name() << "(";

	auto args = getArguments();

	for (const auto& arg : llvm::enumerate(args))
	{
		if (arg.index() > 0)
			printer << ", ";

		printer << arg.value() << " : " << arg.value().getType();
	}

	printer << ") -> (";

	auto results = getType().getResults();

	for (const auto& result : llvm::enumerate(results))
	{
		if (result.index() > 0)
			printer << ", ";

		printer << result.value();
	}

	printer << ")";

	size_t index = 0;

	auto attributes = getOperation()->getAttrs();

	if (attributes.size() > 2)
	{
		printer << " attributes {";

		for (auto attribute : attributes)
		{
			if (attribute.first == mlir::SymbolTable::getSymbolAttrName() ||
					attribute.first == getTypeAttrName())
				continue;

			if (index++ > 0)
				printer << ", ";

			printer << attribute.first;
			printer << " = ";
			printer << attribute.second;
		}

		printer << "}";
	}

	printer.printRegion(getBody(), false);
}

mlir::LogicalResult FunctionOp::verify()
{
	if (getNumArguments() != argsNames().size())
		return emitOpError("requires all the args to have their names defined");

	if (getNumResults() != resultsNames().size())
		return emitOpError("requires all the results to have their names defined");

	auto returnOp = mlir::cast<ReturnOp>(getBody().back().getTerminator());

	if (getNumResults() != returnOp.values().size())
		return emitOpError("requires the return operation to have the same number of values as the result types of the function");

	for (const auto& [returnType, functionType] : llvm::zip(returnOp.values().getTypes(), getType().getResults()))
		if (returnType != functionType)
			return emitOpError("requires the return values to match the function signature");

	return mlir::success();
}

unsigned int FunctionOp::getNumFuncArguments()
{
	return getType().getInputs().size();
}

unsigned int FunctionOp::getNumFuncResults()
{
	return getType().getResults().size();
}

mlir::Region* FunctionOp::getCallableRegion()
{
	return &getBody();
}

llvm::ArrayRef<mlir::Type> FunctionOp::getCallableResults()
{
	return getType().getResults();
}

llvm::StringRef FunctionOp::name()
{
	return getOperation()->getAttrOfType<mlir::StringAttr>(mlir::SymbolTable::getSymbolAttrName()).getValue();
}

llvm::ArrayRef<mlir::Attribute> FunctionOp::argsNames()
{
	return Adaptor(*this).argsNames();
}

llvm::ArrayRef<mlir::Attribute> FunctionOp::resultsNames()
{
	return Adaptor(*this).resultsNames();
}

bool FunctionOp::hasDerivative()
{
	return getOperation()->hasAttr("derivative");
}

//===----------------------------------------------------------------------===//
// Modelica::ReturnOp
//===----------------------------------------------------------------------===//

mlir::ValueRange ReturnOpAdaptor::values()
{
	return getValues();
}

void ReturnOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange values)
{
	state.addOperands(values);
}

mlir::ParseResult ReturnOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> values;
	llvm::SmallVector<mlir::Type, 3> types;

	llvm::SMLoc valuesLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(values) ||
			parser.parseOptionalColonTypeList(types) ||
			parser.resolveOperands(values, types, valuesLoc, result.operands))
		return mlir::failure();

	return mlir::success();
}

void ReturnOp::print(mlir::OpAsmPrinter& printer)
{
	auto operands = values();
	printer << getOperationName() << " " << operands;

	if (!operands.empty())
		printer << " : " << operands.getTypes();
}

void ReturnOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	for (auto value : values())
		if (value.getType().isa<PointerType>())
			effects.emplace_back(mlir::MemoryEffects::Read::get(), value, mlir::SideEffects::DefaultResource::get());
}

mlir::ValueRange ReturnOp::values()
{
	return Adaptor(*this).values();
}

//===----------------------------------------------------------------------===//
// Modelica::DerFunctionOp
//===----------------------------------------------------------------------===//

void DerFunctionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, llvm::StringRef name, llvm::StringRef derivedFunction, llvm::ArrayRef<llvm::StringRef> independentVariables)
{
	state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
	state.addAttribute("derived_function", builder.getStringAttr(derivedFunction));
	state.addAttribute("independent_variables", builder.getStrArrayAttr(independentVariables));
}

void DerFunctionOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.der_function @" << name();
	printer << " {";

	size_t index = 0;

	for (auto attribute : getOperation()->getAttrs())
	{
		if (attribute.first == mlir::SymbolTable::getSymbolAttrName())
			continue;

		if (index++ > 0)
			printer << ", ";

		printer << attribute.first;
		printer << ": ";
		printer << attribute.second;
	}

	printer << "}";
}

mlir::LogicalResult DerFunctionOp::verify()
{
	auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
	auto* base = module.lookupSymbol(derivedFunction());

	if (base == nullptr)
		return emitOpError("requires the function to be derived to exist within the module");

	return mlir::success();
}

mlir::Region* DerFunctionOp::getCallableRegion()
{
	// The function body will be created by an appropriate pass, and at the same
	// type the der function will be converted to a standard function. Thus,
	// this operation is like an external function.

	return nullptr;
}

llvm::ArrayRef<mlir::Type> DerFunctionOp::getCallableResults()
{
	auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
	return mlir::cast<mlir::CallableOpInterface>(module.lookupSymbol(derivedFunction())).getCallableResults();
}

llvm::StringRef DerFunctionOp::name()
{
	return getOperation()->getAttrOfType<mlir::StringAttr>(mlir::SymbolTable::getSymbolAttrName()).getValue();
}

llvm::StringRef DerFunctionOp::derivedFunction()
{
	return getOperation()->getAttrOfType<mlir::StringAttr>("derived_function").getValue();
}

llvm::ArrayRef<mlir::Attribute> DerFunctionOp::independentVariables()
{
	return getOperation()->getAttrOfType<mlir::ArrayAttr>("independent_variables").getValue();
}

//===----------------------------------------------------------------------===//
// Modelica::ConstantOp
//===----------------------------------------------------------------------===//

void ConstantOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Attribute attribute)
{
	state.addAttribute("value", attribute);
	state.addTypes(attribute.getType());
}

mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::Attribute value;

	if (parser.parseAttribute(value))
		return mlir::failure();

	result.attributes.append("value", value);
	result.addTypes(value.getType());
	return mlir::success();
}

void ConstantOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << value();
}

mlir::OpFoldResult ConstantOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
{
	assert(operands.empty() && "constant has no operands");
	return value();
}

void ConstantOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	mlir::Value zero = builder.create<ConstantOp>(getLoc(), RealAttribute::get(RealType::get(getContext()), 0));
	derivatives.map(getResult(), zero);
}

mlir::Attribute ConstantOp::value()
{
	return getOperation()->getAttr("value");
}

mlir::Type ConstantOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

//===----------------------------------------------------------------------===//
// Modelica::CastOp
//===----------------------------------------------------------------------===//

mlir::Value CastOpAdaptor::value()
{
	return getValues()[0];
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

mlir::LogicalResult CastOp::verify()
{
	mlir::Type inputType = value().getType();

	if (!inputType.isa<mlir::IndexType, BooleanType, IntegerType, RealType>())
		return emitOpError("requires the value to be an index, boolean, integer or real");

	if (!resultType().isa<mlir::IndexType, BooleanType, IntegerType, RealType>())
		return emitOpError("requires the result type to be index, boolean, integer or real");

	return mlir::success();
}

mlir::OpFoldResult CastOp::fold(mlir::ArrayRef<mlir::Attribute> operands)
{
	if (value().getType() == resultType())
		return value();

	return nullptr;
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
// Modelica::AssignmentOp
//===----------------------------------------------------------------------===//

mlir::Value AssignmentOpAdaptor::source()
{
	return getValues()[0];
}

mlir::Value AssignmentOpAdaptor::destination()
{
	return getValues()[1];
}

void AssignmentOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value source, mlir::Value destination)
{
	state.addOperands({ source, destination });
}

void AssignmentOp::print(mlir::OpAsmPrinter& printer)
{
	mlir::Value source = this->source();
	mlir::Value destination = this->destination();
	printer << "modelica.assign " << source << " to " << destination << " : " << source.getType() << ", " << destination.getType();
}

void AssignmentOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (source().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), source(), mlir::SideEffects::DefaultResource::get());

	if (destination().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Write::get(), source(), mlir::SideEffects::DefaultResource::get());
}

mlir::Value AssignmentOp::source()
{
	return Adaptor(*this).source();
}

mlir::Value AssignmentOp::destination()
{
	return Adaptor(*this).destination();
}

//===----------------------------------------------------------------------===//
// Modelica::CallOp
//===----------------------------------------------------------------------===//

mlir::ValueRange CallOpAdaptor::args()
{
	return getValues();
}

void CallOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::StringRef callee, mlir::TypeRange results, mlir::ValueRange args, unsigned int movedResults)
{
	state.addAttribute("callee", builder.getSymbolRefAttr(callee));
	state.addOperands(args);
	state.addTypes(results);
	state.addAttribute("moved_results", builder.getUI32IntegerAttr(movedResults));
}

mlir::ParseResult CallOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	auto& builder = parser.getBuilder();

	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> args;
	llvm::SmallVector<mlir::Type, 3> argsTypes;

	llvm::SmallVector<mlir::Type, 3> resultsTypes;

	mlir::FlatSymbolRefAttr calleeAttr;
	if (parser.parseAttribute(calleeAttr, builder.getType<mlir::NoneType>(), "callee", result.attributes))
		return mlir::failure();

	llvm::SMLoc argsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(args, mlir::OpAsmParser::Delimiter::Paren))
		return mlir::failure();

	mlir::Attribute movedResultsAttr;

	if (mlir::succeeded(parser.parseOptionalLBrace()))
	{
		if (parser.parseAttribute(movedResultsAttr, "moved_results", result.attributes) ||
				parser.parseRBrace())
			return mlir::failure();
	}

	if (!result.attributes.getNamed("moved_results").hasValue())
		result.attributes.append("moved_results", builder.getUI32IntegerAttr(0));

	if (parser.parseColon() ||
			parser.parseLParen())
		return mlir::failure();

	if (mlir::failed(parser.parseOptionalRParen()))
	{
		if (parser.parseTypeList(argsTypes) ||
				parser.parseRParen())
			return mlir::failure();
	}

	if (parser.parseArrow() ||
			parser.parseLParen())
		return mlir::failure();

	if (mlir::failed(parser.parseOptionalRParen()))
	{
		if (parser.parseTypeList(resultsTypes) ||
				parser.parseRParen())
			return mlir::failure();
	}

	if (args.size() != argsTypes.size())
		return parser.emitError(argsLoc)
				<< "expected as many args types as args "
				<< "(expected " << args.size() << " got "
				<< argsTypes.size() << ")";

	if (parser.resolveOperands(args, argsTypes, argsLoc, result.operands))
		return mlir::failure();

	result.addTypes(resultsTypes);
	return mlir::success();
}

void CallOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName()
					<< " @" << callee() << "(" << args() << ")";

	if (movedResults() != 0)
		printer << " {moved_results = " << movedResults() << "}";

	printer << " : (" << args().getTypes() << ") -> (" << getResultTypes() << ")";
}

void CallOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	// The callee may have no arguments and no results, but still have side
	// effects (i.e. an external function writing elsewhere). Thus we need to
	// consider the call itself as if it is has side effects and prevent the
	// CSE pass to erase it.
	effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::SideEffects::DefaultResource::get());

	// Declare the side effects on the array arguments.
	unsigned int movedResultsCount = movedResults();
	unsigned int nativeArgsCount = args().size() - movedResultsCount;

	for (size_t i = 0; i < nativeArgsCount; ++i)
		if (args()[i].getType().isa<PointerType>())
			effects.emplace_back(mlir::MemoryEffects::Read::get(), args()[i], mlir::SideEffects::DefaultResource::get());

	// Declare the side effects on the static array results that have been
	// promoted to arguments.

	for (size_t i = 0; i < movedResultsCount; ++i)
		effects.emplace_back(mlir::MemoryEffects::Write::get(), args()[nativeArgsCount + i], mlir::SideEffects::DefaultResource::get());

	for (const auto& [value, type] : llvm::zip(getResults(), getResultTypes()))
	{
		// The result arrays, which will be allocated by the callee on the heap,
		// must be seen as if they were allocated by the function call. This way,
		// the deallocation pass can free them.

		if (auto pointerType = type.dyn_cast<PointerType>(); pointerType && pointerType.getAllocationScope() == BufferAllocationScope::heap)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), value, mlir::SideEffects::DefaultResource::get());

		/*
		// Records are considered as resources to be freed, because they
		// potentially have subtypes that need to be freed.

		if (type.isa<StructType>())
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), value, mlir::SideEffects::DefaultResource::get());
		 */
	}
}

mlir::CallInterfaceCallable CallOp::getCallableForCallee() {
	return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
}

mlir::Operation::operand_range CallOp::getArgOperands()
{
	return getOperands();
}

mlir::LogicalResult CallOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	if (getNumResults() != 1)
		return emitError("The callee must have one and only one result");

	if (argumentIndex >= args().size())
		return emitError("Index out of bounds: " + std::to_string(argumentIndex));

	if (auto size = currentResult.size(); size != 1)
		return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");

	mlir::Value toNest = currentResult[0];

	auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
	auto callee = module.lookupSymbol<FunctionOp>(this->callee());

	if (!callee->hasAttr("inverse"))
		return emitError("Function " + callee->getName().getStringRef() + " is not invertible");

	auto inverseAnnotation = callee->getAttrOfType<InverseFunctionsAttribute>("inverse");

	if (!inverseAnnotation.isInvertible(argumentIndex))
		return emitError("Function " + callee->getName().getStringRef() + " is not invertible for argument " + std::to_string(argumentIndex));

	size_t argsSize = args().size();
	llvm::SmallVector<mlir::Value, 3> args;

	for (auto arg : inverseAnnotation.getArgumentsIndexes(argumentIndex))
	{
		if (arg < argsSize)
		{
			args.push_back(this->args()[arg]);
		}
		else
		{
			assert(arg == argsSize);
			args.push_back(toNest);
		}
	}

	auto invertedCall = builder.create<CallOp>(getLoc(), inverseAnnotation.getFunction(argumentIndex), this->args()[argumentIndex].getType(), args);

	getResult(0).replaceAllUsesWith(this->args()[argumentIndex]);
	erase();

	for (auto& use : toNest.getUses())
		if (use.getOwner() != invertedCall)
			use.set(invertedCall.getResult(0));

	return mlir::success();
}

mlir::StringRef CallOp::callee()
{
	return getOperation()->getAttrOfType<mlir::FlatSymbolRefAttr>("callee").getValue();
}

mlir::ValueRange CallOp::args()
{
	return Adaptor(*this).args();
}

unsigned int CallOp::movedResults()
{
	auto attr = getOperation()->getAttrOfType<mlir::IntegerAttr>("moved_results");
	return attr.getUInt();
}

//===----------------------------------------------------------------------===//
// Modelica::MemberCreateOp
//===----------------------------------------------------------------------===//

mlir::ValueRange MemberCreateOpAdaptor::dynamicDimensions()
{
	return getValues();
}

void MemberCreateOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, llvm::StringRef name, mlir::Type type, mlir::ValueRange dynamicDimensions, mlir::NamedAttrList attributes)
{
	state.addAttribute("name", builder.getStringAttr(name));
	state.addTypes(type);
	state.addOperands(dynamicDimensions);
	state.attributes.append(attributes);
}

mlir::ParseResult MemberCreateOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 2> dynamicDimensions;
	mlir::StringAttr nameAttr;
	llvm::SmallVector<mlir::Type, 2> dynamicDimensionsTypes;
	mlir::Type resultType;

	mlir::NamedAttrList attributes;

	if (parser.parseOptionalAttrDict(attributes))
		return mlir::failure();

	result.attributes.append(attributes);

	llvm::SMLoc dynamicDimensionsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(dynamicDimensions))
		return mlir::failure();

	if (parser.parseColon())
		return mlir::failure();

	if (mlir::succeeded(parser.parseOptionalLParen()))
	{
		if (parser.parseTypeList(dynamicDimensionsTypes) ||
				parser.parseRParen() ||
				parser.parseArrow())
			return mlir::failure();
	}

	if (parser.parseType(resultType))
		return mlir::failure();

	if (mlir::succeeded(parser.parseOptionalArrow()))
	{
		dynamicDimensionsTypes.push_back(resultType);

		if (parser.parseType(resultType))
			return mlir::failure();
	}

	if (dynamicDimensions.size() != dynamicDimensionsTypes.size())
		return parser.emitError(dynamicDimensionsLoc)
				<< "expected as many dimensions types as dimensions "
				<< "(expected " << dynamicDimensions.size() << " got "
				<< dynamicDimensionsTypes.size() << ")";

	if (parser.resolveOperands(dynamicDimensions, dynamicDimensionsTypes, dynamicDimensionsLoc, result.operands))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void MemberCreateOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName();

	for (const auto& dynamicDimension : llvm::enumerate(dynamicDimensions()))
	{
		if (dynamicDimension.index() > 0)
			printer << ", ";

		printer << dynamicDimension.value();
	}

	printer << " " << getOperation()->getAttrDictionary();
	printer << " : ";

	if (auto dimensions = dynamicDimensions().getTypes(); !dimensions.empty())
	{
		if (dimensions.size() > 1)
			printer << "(";

		printer << dimensions;

		if (dimensions.size() > 1)
			printer << ")";

		printer << " -> ";
	}

	printer << resultType();
}

mlir::LogicalResult MemberCreateOp::verify()
{
	if (!resultType().isa<MemberType>())
		return emitOpError("requires the result to be a member type");

	return mlir::success();
}

void MemberCreateOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());
}

llvm::StringRef MemberCreateOp::name()
{
	return getOperation()->getAttrOfType<mlir::StringAttr>("name").getValue();
}

mlir::Type MemberCreateOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::ValueRange MemberCreateOp::dynamicDimensions()
{
	return Adaptor(*this).dynamicDimensions();
}

//===----------------------------------------------------------------------===//
// Modelica::MemberLoadOp
//===----------------------------------------------------------------------===//

mlir::Value MemberLoadOpAdaptor::member()
{
	return getValues()[0];
}

void MemberLoadOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value member)
{
	state.addTypes(resultType);
	state.addOperands(member);
}

mlir::ParseResult MemberLoadOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::OpAsmParser::OperandType member;
	mlir::Type resultType;

	if (parser.parseOperand(member) ||
			parser.parseColon())
		return mlir::failure();

	llvm::SMLoc memberTypeLoc = parser.getCurrentLocation();

	if (parser.parseType(resultType))
		return mlir::failure();

	auto memberType = resultType.isa<PointerType>() ?
										MemberType::get(resultType.cast<PointerType>()) :
										MemberType::get(parser.getBuilder().getContext(), MemberAllocationScope::stack, resultType);

	if (parser.resolveOperand(member, memberType, result.operands))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void MemberLoadOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << member() << " : " << resultType();
}

mlir::LogicalResult MemberLoadOp::verify()
{
	if (!member().getType().isa<MemberType>())
		return emitOpError("requires a source with member type");

	return mlir::success();
}

void MemberLoadOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Read::get(), member(), mlir::SideEffects::DefaultResource::get());
}

void MemberLoadOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	auto derivedOp = builder.create<MemberLoadOp>(getLoc(), resultType(), derivatives.lookup(member()));
	derivatives.map(getResult(), derivedOp.getResult());
}

mlir::Type MemberLoadOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value MemberLoadOp::member()
{
	return Adaptor(*this).member();
}

//===----------------------------------------------------------------------===//
// Modelica::MemberStoreOp
//===----------------------------------------------------------------------===//

mlir::Value MemberStoreOpAdaptor::member()
{
	return getValues()[0];
}

mlir::Value MemberStoreOpAdaptor::value()
{
	return getValues()[1];
}

void MemberStoreOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value member, mlir::Value value)
{
	state.addOperands(member);
	state.addOperands(value);
}

mlir::ParseResult MemberStoreOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 2> operands;
	mlir::Type memberType;

	if (parser.parseOperandList(operands, 2) ||
			parser.parseColon())
		return mlir::failure();

	llvm::SMLoc resultTypeLoc = parser.getCurrentLocation();

	if (parser.parseType(memberType))
		return mlir::failure();

	if (!memberType.isa<MemberType>())
		return parser.emitError(resultTypeLoc)
				<< "specified type must be a member type";

	if (parser.resolveOperand(operands[0], memberType, result.operands) ||
			parser.resolveOperand(operands[1], memberType.cast<MemberType>().getElementType(), result.operands))
		return mlir::failure();

	return mlir::success();
}

void MemberStoreOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName()
					<< " " << member() << ", " << value()
					<< " : " << member().getType();
}

mlir::LogicalResult MemberStoreOp::verify()
{
	if (!member().getType().isa<MemberType>())
		return emitOpError("requires the destination to have member type");

	auto memberType = member().getType().cast<MemberType>();
	mlir::Type valueType = value().getType();

	if (valueType.isa<PointerType>())
	{
		auto pointerType = valueType.cast<PointerType>();

		for (const auto& [valueDimension, memberDimension] : llvm::zip(pointerType.getShape(), memberType.getShape()))
			if (valueDimension != -1 && memberDimension != -1 && valueDimension != memberDimension)
				return emitOpError("requires the shapes to be compatible");
	}

	return mlir::success();
}

void MemberStoreOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Read::get(), value(), mlir::SideEffects::DefaultResource::get());
	effects.emplace_back(mlir::MemoryEffects::Write::get(), member(), mlir::SideEffects::DefaultResource::get());
}

void MemberStoreOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	// Store operations should be derived only if they store a value into
	// a member whose derivative is created by the current function. Otherwise,
	// we would create a double store into that derived member.

	assert(derivatives.contains(member()) && "Derived member not found");
	mlir::Value derivedMember = derivatives.lookup(member());

	if (!derivatives.contains(derivedMember))
		builder.create<MemberStoreOp>(getLoc(), derivedMember, derivatives.lookup(value()));
}

mlir::Value MemberStoreOp::member()
{
	return Adaptor(*this).member();
}

mlir::Value MemberStoreOp::value()
{
	return Adaptor(*this).value();
}

//===----------------------------------------------------------------------===//
// Modelica::AllocaOp
//===----------------------------------------------------------------------===//

mlir::ValueRange AllocaOpAdaptor::dynamicDimensions()
{
	return getValues();
}

void AllocaOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type elementType, llvm::ArrayRef<long> shape, mlir::ValueRange dimensions, bool constant)
{
	state.addTypes(PointerType::get(state.getContext(), BufferAllocationScope::stack, elementType, shape));
	state.addOperands(dimensions);
	state.addAttribute("constant", builder.getBoolAttr(constant));
}

mlir::ParseResult AllocaOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	auto& builder = parser.getBuilder();

	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> indexes;
	llvm::SmallVector<mlir::Type, 3> indexesTypes;

	mlir::Type resultType;

	llvm::SMLoc indexesLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(indexes))
		return mlir::failure();

	// TODO: parse constant attribute
	result.addAttribute("constant", builder.getBoolAttr(false));

	if (parser.parseColon())
		return mlir::failure();

	if (!indexes.empty())
	{
		if (mlir::succeeded(parser.parseOptionalLParen()))
		{
			if (parser.parseTypeList(indexesTypes) ||
					parser.parseRParen())
				return mlir::failure();
		}
		else if (parser.parseTypeList(indexesTypes))
		{
			return mlir::failure();
		}

		if (parser.parseArrow())
			return mlir::failure();
	}

	if (indexes.size() != indexesTypes.size())
		return parser.emitError(indexesLoc)
				<< "expected as many indexes types as indexes "
				<< "(expected " << indexes.size() << " got "
				<< indexesTypes.size() << ")";

	if (parser.parseType(resultType))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void AllocaOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName()
					<< " " << dynamicDimensions()
					<< ": ";

	if (isConstant())
		printer << "{constant = true} ";

	if (auto dimensionsTypes = dynamicDimensions().getTypes(); !dimensionsTypes.empty())
	{
		if (dimensionsTypes.size() > 1)
			printer << "(";

		printer << dimensionsTypes;

		if (dimensionsTypes.size() > 1)
			printer << ")";

		printer << " -> ";
	}

	printer << resultType();
}

mlir::LogicalResult AllocaOp::verify()
{
	auto shape = resultType().getShape();
	unsigned int unknownSizes = 0;

	for (const auto& size : shape)
		if (size == -1)
			++unknownSizes;

	if (unknownSizes != dynamicDimensions().size())
		return emitOpError("requires the dynamic dimensions amount (" +
											 std::to_string(dynamicDimensions().size()) +
											 ") to match the number of provided values (" +
											 std::to_string(unknownSizes) + ")");

	return mlir::success();
}

void AllocaOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());
}

PointerType AllocaOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<PointerType>();
}

mlir::ValueRange AllocaOp::dynamicDimensions()
{
	return Adaptor(*this).dynamicDimensions();
}

bool AllocaOp::isConstant()
{
	return getOperation()->getAttrOfType<mlir::BoolAttr>("constant").getValue();
}

//===----------------------------------------------------------------------===//
// Modelica::AllocOp
//===----------------------------------------------------------------------===//

mlir::ValueRange AllocOpAdaptor::dynamicDimensions()
{
	return getValues();
}

void AllocOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type elementType, llvm::ArrayRef<long> shape, mlir::ValueRange dimensions, bool shouldBeFreed, bool constant)
{
	state.addTypes(PointerType::get(state.getContext(), BufferAllocationScope::heap, elementType, shape));
	state.addOperands(dimensions);

	state.addAttribute(getAutoFreeAttrName(), builder.getBoolAttr(shouldBeFreed));
	state.addAttribute("constant", builder.getBoolAttr(constant));
}

void AllocOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.alloc ";
	printer.printOperands(getOperands());
	printer << ": ";
	printer.printType(getOperation()->getResultTypes()[0]);
}

mlir::LogicalResult AllocOp::verify()
{
	auto shape = resultType().getShape();
	unsigned int unknownSizes = 0;

	for (const auto& size : shape)
		if (size == -1)
			++unknownSizes;

	if (unknownSizes != dynamicDimensions().size())
		return emitOpError("requires the dynamic dimensions amount (" +
											 std::to_string(dynamicDimensions().size()) +
											 ") to match the number of provided values (" +
											 std::to_string(unknownSizes) + ")");

	return mlir::success();
}

void AllocOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (shouldBeFreed())
		effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
	else
	{
		// If the buffer is marked as manually freed, then we need to set the
		// operation to have a generic side effect, or the CSE pass would
		// otherwise consider all the allocs with the same structure as equal,
		// and thus would replace all the subsequent buffers with the first
		// allocated one.

		effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::SideEffects::DefaultResource::get());
	}
}

PointerType AllocOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<PointerType>();
}

mlir::ValueRange AllocOp::dynamicDimensions()
{
	return Adaptor(*this).dynamicDimensions();
}

bool AllocOp::isConstant()
{
	return getOperation()->getAttrOfType<mlir::BoolAttr>("constant").getValue();
}

//===----------------------------------------------------------------------===//
// Modelica::FreeOp
//===----------------------------------------------------------------------===//

mlir::Value FreeOpAdaptor::memory()
{
	return getValues()[0];
}

void FreeOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value memory)
{
	state.addOperands(memory);
}

void FreeOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.free " << memory();
}

mlir::LogicalResult FreeOp::verify()
{
	if (auto pointerType = memory().getType().dyn_cast<PointerType>(); pointerType)
	{
		if (pointerType.getAllocationScope() == BufferAllocationScope::heap)
			return mlir::success();

		return emitOpError("requires the memory to be allocated on the heap");
	}

	return emitOpError("requires operand to be a pointer to heap allocated memory");
}

void FreeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Free::get(), memory(), mlir::SideEffects::DefaultResource::get());
}

mlir::Value FreeOp::memory()
{
	return Adaptor(*this).memory();
}

//===----------------------------------------------------------------------===//
// Modelica::PtrCastOp
//===----------------------------------------------------------------------===//

mlir::Value PtrCastOpAdaptor::memory()
{
	return getValues()[0];
}

void PtrCastOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value memory, mlir::Type resultType)
{
	state.addOperands(memory);
	state.addTypes(resultType);
}

void PtrCastOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.ptr_cast " << memory() << " : " << resultType();
}

mlir::LogicalResult PtrCastOp::verify()
{
	mlir::Type sourceType = memory().getType();
	mlir::Type destinationType = resultType();

	if (auto source = sourceType.dyn_cast<PointerType>())
	{
		if (auto destination = destinationType.dyn_cast<PointerType>())
		{
			if (source.getElementType() != resultType().cast<PointerType>().getElementType())
				return emitOpError("requires the result pointer type to have the same element type of the operand");

			if (source.getRank() != resultType().cast<PointerType>().getRank())
				return emitOpError("requires the result pointer type to have the same rank as the operand");

			if (destination.getAllocationScope() != BufferAllocationScope::unknown &&
					destination.getAllocationScope() != source.getAllocationScope())
				return emitOpError("can change the allocation scope only to the unknown one");

			for (auto [source, destination] : llvm::zip(source.getShape(), destination.getShape()))
				if (destination != -1 && source != destination)
					return emitOpError("can change the dimensions size only to an unknown one");

			return mlir::success();
		}

		if (auto destination = destinationType.dyn_cast<UnsizedPointerType>())
		{
			if (source.getElementType() != destination.getElementType())
				return emitOpError("requires the result pointer type to have the same element type of the operand");

			return mlir::success();
		}

		if (destinationType.isa<OpaquePointerType>())
			return mlir::success();
	}

	if (auto source = sourceType.dyn_cast<OpaquePointerType>())
	{
		if (auto destination = destinationType.dyn_cast<PointerType>())
		{
			// If the destination is a non-opaque pointer type, its shape must not
			// contain dynamic sizes.

			for (auto size : destination.getShape())
				if (size == -1)
					return emitOpError("requires the non-opaque pointer type to have fixed shape");

			return mlir::success();
		}
	}

	return emitOpError("requires a compatible conversion");
}

mlir::Value PtrCastOp::memory()
{
	return Adaptor(*this).memory();
}

mlir::Type PtrCastOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

//===----------------------------------------------------------------------===//
// Modelica::DimOp
//===----------------------------------------------------------------------===//

mlir::Value DimOpAdaptor::memory()
{
	return getValues()[0];
}

mlir::Value DimOpAdaptor::dimension()
{
	return getValues()[1];
}

void DimOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value memory, mlir::Value dimension)
{
	state.addOperands(memory);
	state.addOperands(dimension);
	state.addTypes(builder.getIndexType());
}

mlir::ParseResult DimOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	auto& builder = parser.getBuilder();

	mlir::OpAsmParser::OperandType array;
	mlir::OpAsmParser::OperandType dimension;

	mlir::Type arrayType;

	if (parser.parseOperand(array) ||
			parser.parseComma() ||
			parser.parseOperand(dimension) ||
			parser.parseColonType(arrayType))
		return mlir::failure();

	if (parser.resolveOperand(array, arrayType, result.operands) ||
			parser.resolveOperand(dimension, builder.getIndexType(), result.operands))
		return mlir::failure();

	result.addTypes(builder.getIndexType());
	return mlir::success();
}

void DimOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName()
					<< " " << memory() << ", " << dimension()
					<< " : " << memory().getType();
}

mlir::LogicalResult DimOp::verify()
{
	if (!memory().getType().isa<PointerType>())
		return emitOpError("requires the operand to be a pointer to an array");

	if (!dimension().getType().isa<mlir::IndexType>())
		return emitOpError("requires the dimension to be an index");

	return mlir::success();
}

mlir::OpFoldResult DimOp::fold(mlir::ArrayRef<mlir::Attribute> operands)
{
	if (operands[1] == nullptr)
		return nullptr;

	if (auto attribute = operands[1].dyn_cast<mlir::IntegerAttr>(); attribute)
	{
		auto pointerType = memory().getType().cast<PointerType>();
		auto shape = pointerType.getShape();

		size_t index = attribute.getInt();

		if (shape[index] != -1)
			return mlir::IntegerAttr::get(mlir::IndexType::get(getContext()), shape[index]);
	}

	return nullptr;
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
// Modelica::SubscriptionOp
//===----------------------------------------------------------------------===//

mlir::Value SubscriptionOpAdaptor::source()
{
	return getValues()[0];
}

mlir::ValueRange SubscriptionOpAdaptor::indexes()
{
	return mlir::ValueRange(std::next(getValues().begin(), 1), getValues().end());
}

void SubscriptionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value source, mlir::ValueRange indexes)
{
	state.addOperands(source);
	state.addOperands(indexes);

	auto sourcePointerType = source.getType().cast<PointerType>();
	mlir::Type resultType = sourcePointerType.slice(indexes.size());
	state.addTypes(resultType);
}

mlir::ParseResult SubscriptionOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::OpAsmParser::OperandType source;
	mlir::Type sourceType;

	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> indexes;
	llvm::SmallVector<mlir::Type, 3> indexesTypes;

	llvm::SMLoc sourceLoc = parser.getCurrentLocation();

	if (parser.parseOperand(source))
		return mlir::failure();

	llvm::SMLoc indexesLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(indexes, mlir::OpAsmParser::Delimiter::Square))
		return mlir::failure();

	if (parser.parseColonType(sourceType) ||
			parser.resolveOperand(source, sourceType, result.operands))
		return mlir::failure();

	if (!sourceType.isa<PointerType>())
		return parser.emitError(sourceLoc, "the source must have a pointer type");

	result.addTypes(sourceType.cast<PointerType>().slice(indexes.size()));

	if (!indexes.empty())
	{
		if (parser.parseComma() ||
				parser.parseTypeList(indexesTypes))
			return mlir::failure();
	}

	if (indexes.size() != indexesTypes.size())
		return parser.emitError(indexesLoc)
				<< "expected as many indexes types as indexes "
				<< "(expected " << indexes.size() << " got "
				<< indexesTypes.size() << ")";

	if (parser.resolveOperands(indexes, indexesTypes, indexesLoc, result.operands))
		return mlir::failure();

	return mlir::success();
}

void SubscriptionOp::print(mlir::OpAsmPrinter& printer)
{
 	printer << getOperationName()
					<< " " << source() << "[" << indexes() << "]"
					<< " : " << source().getType();

	if (auto ind = indexes(); !ind.empty())
	{
		for (const auto& index : ind)
			printer << ", " << index.getType();
	}
}

mlir::Value SubscriptionOp::getViewSource()
{
	return source();
}

void SubscriptionOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	auto derivedOp = builder.create<SubscriptionOp>(getLoc(), derivatives.lookup(source()), indexes());
	derivatives.map(getResult(), derivedOp.getResult());
}

PointerType SubscriptionOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<PointerType>();
}

mlir::Value SubscriptionOp::source()
{
	return Adaptor(*this).source();
}

mlir::ValueRange SubscriptionOp::indexes()
{
	return Adaptor(*this).indexes();
}

//===----------------------------------------------------------------------===//
// Modelica::LoadOp
//===----------------------------------------------------------------------===//

mlir::Value LoadOpAdaptor::memory()
{
	return getValues()[0];
}

mlir::ValueRange LoadOpAdaptor::indexes()
{
	return mlir::ValueRange(std::next(getValues().begin(), 1), getValues().end());
}

void LoadOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value memory, mlir::ValueRange indexes)
{
	state.addOperands(memory);
	state.addOperands(indexes);
	state.addTypes(memory.getType().cast<PointerType>().getElementType());
}

mlir::ParseResult LoadOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	auto& builder = parser.getBuilder();

	mlir::OpAsmParser::OperandType array;
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> indexes;
	mlir::Type arrayType;

	if (parser.parseOperand(array) ||
			parser.parseOperandList(indexes, mlir::OpAsmParser::Delimiter::Square))
		return mlir::failure();

	llvm::SMLoc arrayTypeLoc = parser.getCurrentLocation();

	if (parser.parseColonType(arrayType))
		return mlir::failure();

	if (parser.resolveOperand(array, arrayType, result.operands) ||
			parser.resolveOperands(indexes, builder.getIndexType(), result.operands))
		return mlir::failure();

	if (!arrayType.isa<PointerType>())
		return parser.emitError(arrayTypeLoc, "the array type must be a pointer type");

	result.addTypes(arrayType.cast<PointerType>().getElementType());
	return mlir::success();
}

void LoadOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName()
					<< " " << memory()
					<< "[" << indexes() << "]"
					<< " : " << memory().getType();
}

mlir::LogicalResult LoadOp::verify()
{
	auto pointerType = memory().getType().cast<PointerType>();

	if (pointerType.getRank() != indexes().size())
		return emitOpError("requires the indexes amount (" +
											 std::to_string(indexes().size()) +
											 ") to match the array rank (" +
											 std::to_string(pointerType.getRank()) + ")");

	return mlir::success();
}

void LoadOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Read::get(), memory(), mlir::SideEffects::DefaultResource::get());
}

void LoadOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	auto derivedOp = builder.create<LoadOp>(getLoc(), derivatives.lookup(memory()), indexes());
	derivatives.map(getResult(), derivedOp.getResult());
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

mlir::Value StoreOpAdaptor::value()
{
	return getValues()[0];
}

mlir::Value StoreOpAdaptor::memory()
{
	return getValues()[1];
}

mlir::ValueRange StoreOpAdaptor::indexes()
{
	return mlir::ValueRange(std::next(getValues().begin(), 2), getValues().end());
}

void StoreOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value value, mlir::Value memory, mlir::ValueRange indexes)
{
	state.addOperands(value);
	state.addOperands(memory);
	state.addOperands(indexes);
}

mlir::ParseResult StoreOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	auto& builder = parser.getBuilder();

	mlir::OpAsmParser::OperandType value;
	mlir::OpAsmParser::OperandType array;
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 2> indexes;
	mlir::Type arrayType;

	if (parser.parseOperand(value) ||
			parser.parseComma() ||
			parser.parseOperand(array) ||
			parser.parseOperandList(indexes, mlir::OpAsmParser::Delimiter::Square) ||
			parser.parseColon())
		return mlir::failure();

	llvm::SMLoc arrayTypeLoc = parser.getCurrentLocation();

	if (parser.parseType(arrayType))
		return mlir::failure();

	if (!arrayType.isa<PointerType>())
		return parser.emitError(arrayTypeLoc)
				<< "destination type must be a pointer type";

	if (parser.resolveOperand(value, arrayType.cast<PointerType>().getElementType(), result.operands) ||
			parser.resolveOperand(array, arrayType, result.operands) ||
			parser.resolveOperands(indexes, builder.getIndexType(), result.operands))
		return mlir::failure();

	return mlir::success();
}

void StoreOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << value() << ", " << memory() << "[";
	printer.printOperands(indexes());
	printer << "] : ";
	printer.printType(memory().getType());
}

mlir::LogicalResult StoreOp::verify()
{
	auto pointerType = memory().getType().cast<PointerType>();

	if (pointerType.getElementType() != value().getType())
		return emitOpError("requires the value to have the same type of the array elements");

	if (pointerType.getRank() != indexes().size())
		return emitOpError("requires the indexes amount (" +
											 std::to_string(indexes().size()) +
											 ") to match the array rank (" +
											 std::to_string(pointerType.getRank()) + ")");

	return mlir::success();
}

void StoreOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), memory(), mlir::SideEffects::DefaultResource::get());
}

void StoreOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	auto derivedOp = builder.create<StoreOp>(
			getLoc(), derivatives.lookup(value()), derivatives.lookup(memory()), indexes());
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
// Modelica::ArrayCloneOp
//===----------------------------------------------------------------------===//

mlir::Value ArrayCloneOpAdaptor::source()
{
	return getValues()[0];
}

void ArrayCloneOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value source, PointerType resultType, bool shouldBeFreed)
{
	state.addOperands(source);
	state.addTypes(resultType);
	state.addAttribute(getAutoFreeAttrName(), builder.getBoolAttr(shouldBeFreed));
}

void ArrayCloneOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << source();
	printer	<< " : " << source().getType() << " -> " << resultType();
}

mlir::LogicalResult ArrayCloneOp::verify()
{
	auto pointerType = resultType();

	if (auto scope = pointerType.getAllocationScope();
			scope != BufferAllocationScope::stack && scope != BufferAllocationScope::heap)
		return emitOpError("requires the result array type to be stack or heap allocated");

	return mlir::success();
}

void ArrayCloneOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Read::get(), source(), mlir::SideEffects::DefaultResource::get());

	auto scope = resultType().getAllocationScope();

	if (scope == BufferAllocationScope::heap && shouldBeFreed())
		effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
	else if (scope == BufferAllocationScope::stack)
		effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());

	effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
}

PointerType ArrayCloneOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<PointerType>();
}

mlir::Value ArrayCloneOp::source()
{
	return Adaptor(*this).source();
}

//===----------------------------------------------------------------------===//
// Modelica::IfOp
//===----------------------------------------------------------------------===//

mlir::Value IfOpAdaptor::condition()
{
	return getValues()[0];
}

void IfOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::TypeRange resultTypes, mlir::Value condition, bool withElseRegion)
{
	mlir::OpBuilder::InsertionGuard guard(builder);
	state.addTypes(resultTypes);
	state.addOperands(condition);

	// "Then" region
	auto* thenRegion = state.addRegion();
	builder.createBlock(thenRegion);

	// "Else" region
	auto* elseRegion = state.addRegion();

	if (withElseRegion)
		builder.createBlock(elseRegion);
}

void IfOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value condition, bool withElseRegion)
{
	build(builder, state, llvm::None, condition, withElseRegion);
}

mlir::ParseResult IfOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::OpAsmParser::OperandType condition;
	mlir::Type conditionType;

	if (parser.parseLParen() ||
			parser.parseOperand(condition) ||
			parser.parseColonType(conditionType) ||
			parser.parseRParen() ||
			parser.resolveOperand(condition, conditionType, result.operands))
		return mlir::failure();

	if (mlir::succeeded(parser.parseOptionalArrow()))
	{
		mlir::Type resultType;
		bool paren = mlir::succeeded(parser.parseOptionalLParen());

		if (mlir::succeeded(*parser.parseOptionalType(resultType)))
			result.addTypes(resultType);

		if (paren)
			if (parser.parseRParen())
				return mlir::failure();
	}

	mlir::Region* thenRegion = result.addRegion();

	if (parser.parseRegion(*thenRegion))
		return mlir::failure();

	mlir::Region* elseRegion = result.addRegion();

	if (mlir::succeeded(parser.parseOptionalKeyword("else")))
	{
		if (parser.parseRegion(*elseRegion))
			return mlir::failure();
	}

	return mlir::success();
}

void IfOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName()
					<< " (" << condition() << " : " << condition().getType() << ")";

	if (!resultTypes().empty())
		printer << " -> " << " (" << resultTypes() << ")";

	printer.printRegion(thenRegion());

	if (!elseRegion().empty())
	{
		printer << " else";
		printer.printRegion(elseRegion());
	}
}

mlir::LogicalResult IfOp::verify()
{
	if (!condition().getType().isa<BooleanType>())
		return emitOpError("requires the condition to be a boolean");

	return mlir::success();
}

void IfOp::getSuccessorRegions(llvm::Optional<unsigned int> index, llvm::ArrayRef<mlir::Attribute> operands, llvm::SmallVectorImpl<mlir::RegionSuccessor>& regions)
{
	// The "then" and the "else" region branch back to the parent operation
	if (index.hasValue())
	{
		regions.push_back(mlir::RegionSuccessor(getOperation()->getResults()));
		return;
	}

	// Don't consider the else region if it is empty
	mlir::Region* elseRegion = &this->elseRegion();

	if (elseRegion->empty())
		elseRegion = nullptr;

	// Otherwise, the successor is dependent on the condition
	bool condition = false;

	if (auto condAttr = operands.front().dyn_cast_or_null<BooleanAttribute>())
	{
		condition = condAttr.getValue() == true;
	}
	else
	{
		// If the condition isn't constant, both regions may be executed
		regions.push_back(mlir::RegionSuccessor(&thenRegion()));

		// If the else region does not exist, it is not a viable successor
		if (elseRegion)
			regions.push_back(mlir::RegionSuccessor(elseRegion));

		return;
	}

	// Add the successor regions using the condition
	regions.push_back(mlir::RegionSuccessor(condition ? &thenRegion() : elseRegion));
}

mlir::TypeRange IfOp::resultTypes()
{
	return getResultTypes();
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

mlir::ValueRange ForOpAdaptor::args()
{
	return getValues();
}

void ForOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange args)
{
	mlir::OpBuilder::InsertionGuard guard(builder);
	state.addOperands(args);

	// Condition block
	builder.createBlock(state.addRegion(), {}, args.getTypes());

	// Step block
	builder.createBlock(state.addRegion(), {}, args.getTypes());

	// Body block
	builder.createBlock(state.addRegion(), {}, args.getTypes());
}

void ForOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.for " << args();

	if (!args().empty())
		printer << " ";

	printer << "condition";
	printer.printRegion(condition(), true);
	printer << " body";
	printer.printRegion(body(), true);
	printer << " step";
	printer.printRegion(step(), true);
}

void ForOp::getSuccessorRegions(llvm::Optional<unsigned int> index, llvm::ArrayRef<mlir::Attribute> operands, llvm::SmallVectorImpl<mlir::RegionSuccessor>& regions)
{
	if (!index.hasValue())
	{
		regions.push_back(mlir::RegionSuccessor(&condition(), condition().getArguments()));
		return;
	}

	assert(*index < 3 && "There are only three regions in a ForOp");

	if (*index == 0)
	{
		regions.emplace_back(&body(), body().getArguments());
		regions.emplace_back(getOperation()->getResults());
	}
	else if (*index == 1)
	{
		regions.emplace_back(&step(), step().getArguments());
	}
	else if (*index == 2)
	{
		regions.emplace_back(&condition(), condition().getArguments());
	}
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

mlir::ValueRange ForOp::args()
{
	return Adaptor(*this).args();
}

//===----------------------------------------------------------------------===//
// Modelica::BreakableForOp
//===----------------------------------------------------------------------===//

mlir::Value BreakableForOpAdaptor::breakCondition()
{
	return getValues()[0];
}

mlir::Value BreakableForOpAdaptor::returnCondition()
{
	return getValues()[1];
}

mlir::ValueRange BreakableForOpAdaptor::args()
{
	return mlir::ValueRange(std::next(getValues().begin(), 2), getValues().end());
}

void BreakableForOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value breakCondition, mlir::Value returnCondition, mlir::ValueRange args)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	state.addOperands(breakCondition);
	state.addOperands(returnCondition);
	state.addOperands(args);

	// Condition block
	builder.createBlock(state.addRegion(), {}, args.getTypes());

	// Step block
	builder.createBlock(state.addRegion(), {}, args.getTypes());

	// Body block
	builder.createBlock(state.addRegion(), {}, args.getTypes());
}

void BreakableForOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " (";
	printer << "break = " << breakCondition() << " : " << breakCondition().getType() << ", ";
	printer << "return = " << returnCondition() << " : " << returnCondition().getType() << ")";

	if (!args().empty())
		printer << "(" << args() << " : " << args().getTypes() << ")";

	printer << " condition";
	printer.printRegion(condition(), false);
	printer << " body";
	printer.printRegion(body(), true);
	printer << " step";
	printer.printRegion(step(), true);
}

mlir::LogicalResult BreakableForOp::verify()
{
	if (auto breakPtr = breakCondition().getType().dyn_cast<PointerType>();
			!breakPtr || !breakPtr.getElementType().isa<BooleanType>() || breakPtr.getRank() != 0)
		return emitOpError("requires the break condition to be a pointer to a single boolean value");

	if (auto returnPtr = breakCondition().getType().dyn_cast<PointerType>();
			!returnPtr || !returnPtr.getElementType().isa<BooleanType>() || returnPtr.getRank() != 0)
		return emitOpError("requires the return condition to be a pointer to a single boolean value");

	return mlir::success();
}

bool BreakableForOp::isDefinedOutsideOfLoop(mlir::Value value)
{
	return !body().isAncestor(value.getParentRegion());
}

mlir::Region& BreakableForOp::getLoopBody()
{
	return body();
}

mlir::LogicalResult BreakableForOp::moveOutOfLoop(llvm::ArrayRef<mlir::Operation*> ops)
{
	for (auto* op : ops)
		op->moveBefore(*this);

	return mlir::success();
}

void BreakableForOp::getSuccessorRegions(llvm::Optional<unsigned int> index, llvm::ArrayRef<mlir::Attribute> operands, llvm::SmallVectorImpl<mlir::RegionSuccessor>& regions)
{
	if (!index.hasValue())
	{
		regions.push_back(mlir::RegionSuccessor(&condition(), condition().getArguments()));
		return;
	}

	assert(*index < 3 && "There are only three regions in a BreakableForOp");

	if (*index == 0)
	{
		regions.emplace_back(&body(), body().getArguments());
		regions.emplace_back(getOperation()->getResults());
	}
	else if (*index == 1)
	{
		regions.emplace_back(&step(), step().getArguments());
	}
	else if (*index == 2)
	{
		regions.emplace_back(&condition(), condition().getArguments());
	}
}

mlir::Region& BreakableForOp::condition()
{
	return getOperation()->getRegion(0);
}

mlir::Region& BreakableForOp::step()
{
	return getOperation()->getRegion(1);
}

mlir::Region& BreakableForOp::body()
{
	return getOperation()->getRegion(2);
}

mlir::Value BreakableForOp::breakCondition()
{
	return Adaptor(*this).breakCondition();
}

mlir::Value BreakableForOp::returnCondition()
{
	return Adaptor(*this).returnCondition();
}

mlir::ValueRange BreakableForOp::args()
{
	return Adaptor(*this).args();
}

//===----------------------------------------------------------------------===//
// Modelica::BreakableWhileOp
//===----------------------------------------------------------------------===//

mlir::Value BreakableWhileOpAdaptor::breakCondition()
{
	return getValues()[0];
}

mlir::Value BreakableWhileOpAdaptor::returnCondition()
{
	return getValues()[1];
}

void BreakableWhileOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value breakCondition, mlir::Value returnCondition)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	state.addOperands(breakCondition);
	state.addOperands(returnCondition);

	// Condition block
	builder.createBlock(state.addRegion());

	// Body block
	builder.createBlock(state.addRegion());
}

mlir::ParseResult BreakableWhileOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::OpAsmParser::OperandType breakCondition;
	mlir::Type breakConditionType;

	mlir::OpAsmParser::OperandType returnCondition;
	mlir::Type returnConditionType;

	if (parser.parseLParen())
		return mlir::failure();

	if (parser.parseKeyword("break") ||
			parser.parseEqual() ||
			parser.parseOperand(breakCondition) ||
			parser.parseColonType(breakConditionType) ||
			parser.resolveOperand(breakCondition, breakConditionType, result.operands))
		return mlir::failure();

	if (parser.parseComma())
		return mlir::failure();

	if (parser.parseKeyword("return") ||
			parser.parseEqual() ||
			parser.parseOperand(returnCondition) ||
			parser.parseColonType(returnConditionType) ||
			parser.resolveOperand(returnCondition, returnConditionType, result.operands))
		return mlir::failure();

	if (parser.parseRParen())
		return mlir::failure();

	mlir::Region* conditionRegion = result.addRegion();

	if (parser.parseRegion(*conditionRegion))
		return mlir::failure();

	if (parser.parseKeyword("do"))
		return mlir::failure();

	mlir::Region* bodyRegion = result.addRegion();

	if (parser.parseRegion(*bodyRegion))
		return mlir::failure();

	return mlir::success();
}

void BreakableWhileOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " (";
	printer << "break = " << breakCondition() << " : " << breakCondition().getType() << ", ";
	printer << "return = " << returnCondition() << " : " << returnCondition().getType() << ")";
	printer.printRegion(condition(), false);
	printer << " do";
	printer.printRegion(body(), false);
}

mlir::LogicalResult BreakableWhileOp::verify()
{
	if (auto breakPtr = breakCondition().getType().dyn_cast<PointerType>();
			!breakPtr || !breakPtr.getElementType().isa<BooleanType>() || breakPtr.getRank() != 0)
		return emitOpError("requires the break condition to be a pointer to a single boolean value");

	if (auto returnPtr = breakCondition().getType().dyn_cast<PointerType>();
			!returnPtr || !returnPtr.getElementType().isa<BooleanType>() || returnPtr.getRank() != 0)
		return emitOpError("requires the return condition to be a pointer to a single boolean value");

	return mlir::success();
}

bool BreakableWhileOp::isDefinedOutsideOfLoop(mlir::Value value)
{
	return !body().isAncestor(value.getParentRegion());
}

mlir::Region& BreakableWhileOp::getLoopBody()
{
	return body();
}

mlir::LogicalResult BreakableWhileOp::moveOutOfLoop(llvm::ArrayRef<mlir::Operation*> ops)
{
	for (auto* op : ops)
		op->moveBefore(*this);

	return mlir::success();
}

void BreakableWhileOp::getSuccessorRegions(llvm::Optional<unsigned int> index, llvm::ArrayRef<mlir::Attribute> operands, llvm::SmallVectorImpl<mlir::RegionSuccessor>& regions)
{
	if (!index.hasValue())
	{
		regions.emplace_back(&condition(), condition().getArguments());
		return;
	}

	assert(*index < 2 && "There are only two regions in a BreakableWhileOp");

	if (*index == 0)
	{
		regions.emplace_back(&body(), body().getArguments());
		regions.emplace_back(getOperation()->getResults());
	}
	else if (*index == 1)
	{
		regions.emplace_back(&condition(), condition().getArguments());
	}
}

mlir::Region& BreakableWhileOp::condition()
{
	return getOperation()->getRegion(0);
}

mlir::Region& BreakableWhileOp::body()
{
	return getOperation()->getRegion(1);
}

mlir::Value BreakableWhileOp::breakCondition()
{
	return Adaptor(*this).breakCondition();
}

mlir::Value BreakableWhileOp::returnCondition()
{
	return Adaptor(*this).returnCondition();
}

//===----------------------------------------------------------------------===//
// Modelica::ConditionOp
//===----------------------------------------------------------------------===//

mlir::Value ConditionOpAdaptor::condition()
{
	return getValues()[0];
}

mlir::ValueRange ConditionOpAdaptor::args()
{
	return mlir::ValueRange(std::next(getValues().begin()), getValues().end());
}

void ConditionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value condition, mlir::ValueRange args)
{
	state.addOperands(condition);
	state.addOperands(args);
}

mlir::ParseResult ConditionOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::OpAsmParser::OperandType condition;
	mlir::Type conditionType;

	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> args;
	llvm::SmallVector<mlir::Type, 3> argsTypes;

	mlir::Type resultType;

	if (parser.parseLParen() ||
			parser.parseOperand(condition) ||
			parser.parseColonType(conditionType) ||
			parser.parseRParen() ||
			parser.resolveOperand(condition, conditionType, result.operands))
		return mlir::failure();

	llvm::SMLoc argsLoc = parser.getCurrentLocation();

	if (mlir::failed(parser.parseOptionalLParen()))
		return mlir::success();

	if (parser.parseOperandList(args) ||
			parser.parseColonTypeList(argsTypes) ||
			parser.resolveOperands(args, argsTypes, argsLoc, result.operands) ||
			parser.parseRParen())
		return mlir::failure();

	if (args.size() != argsTypes.size())
		return parser.emitError(argsLoc)
				<< "expected as many args types as args "
				<< "(expected " << args.size() << " got "
				<< argsTypes.size() << ")";

	return mlir::success();
}

void ConditionOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " (" << condition() << " : " << condition().getType() << ")";

	if (auto arguments = args(); !arguments.empty())
		printer << " (" << arguments << " : " << arguments.getTypes() << ")";
}

mlir::LogicalResult ConditionOp::verify()
{
	return mlir::success(isNumeric(condition()));
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

mlir::ValueRange YieldOpAdaptor::values()
{
	return getValues();
}

void YieldOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange operands)
{
	state.addOperands(operands);
}

mlir::ParseResult YieldOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> values;
	llvm::SmallVector<mlir::Type, 3> valuesTypes;
	mlir::Type resultType;

	llvm::SMLoc valuesLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(values))
		return mlir::failure();

	if (!values.empty())
		if (parser.parseOptionalColonTypeList(valuesTypes))
			return mlir::failure();

	if (values.size() != valuesTypes.size())
		return parser.emitError(valuesLoc)
				<< "expected as many types as values "
				<< "(expected " << values.size() << " got "
				<< valuesTypes.size() << ")";

	if (parser.resolveOperands(values, valuesTypes, valuesLoc, result.operands))
		return mlir::failure();

	return mlir::success();
}

void YieldOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << values();

	if (auto types = values().getTypes(); !types.empty())
		printer << " : " << types;
}

mlir::ValueRange YieldOp::values()
{
	return Adaptor(*this).values();
}

//===----------------------------------------------------------------------===//
// Modelica::NotOp
//===----------------------------------------------------------------------===//

mlir::Value NotOpAdaptor::operand()
{
	return getValues()[0];
}

void NotOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand)
{
	state.addTypes(resultType);
	state.addOperands(operand);
}

mlir::ParseResult NotOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void NotOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " "
					<< operand() << " : "
					<< operand().getType() << " -> " << resultType();
}

mlir::LogicalResult NotOp::verify()
{
	if (!operand().getType().isa<BooleanType>())
		if (auto pointerType = operand().getType().dyn_cast<PointerType>(); !pointerType || !pointerType.getElementType().isa<BooleanType>())
			return emitOpError("requires the operand to be a boolean or an array of booleans");

	return mlir::success();
}

void NotOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (operand().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());

	if (auto pointerType = resultType().dyn_cast<PointerType>())
	{
		if (pointerType.getAllocationScope() == BufferAllocationScope::stack)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());
		else if (pointerType.getAllocationScope() == BufferAllocationScope::heap)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());

		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
	}
}

mlir::Type NotOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value NotOp::operand()
{
	return Adaptor(*this).operand();
}

//===----------------------------------------------------------------------===//
// Modelica::AndOp
//===----------------------------------------------------------------------===//

mlir::Value AndOpAdaptor::lhs()
{
	return getValues()[0];
}

mlir::Value AndOpAdaptor::rhs()
{
	return getValues()[1];
}

void AndOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::ParseResult AndOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void AndOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " "
					<< lhs() << ", " << rhs() << " : ("
					<< lhs().getType() << ", " << rhs().getType() << ") -> "
					<< resultType();
}

mlir::LogicalResult AndOp::verify()
{
	mlir::Type lhsType = lhs().getType();
	mlir::Type rhsType = rhs().getType();

	if (lhsType.isa<BooleanType>() && rhsType.isa<BooleanType>())
		return mlir::success();

	if (lhsType.isa<PointerType>() && rhsType.isa<PointerType>())
		if (lhsType.cast<PointerType>().getElementType().isa<BooleanType>() &&
		    rhsType.cast<PointerType>().getElementType().isa<BooleanType>())
			return mlir::success();

	return emitOpError("requires the operands to be booleans or arrays of booleans");
}

void AndOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (lhs().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());

	if (rhs().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());

	if (auto pointerType = resultType().dyn_cast<PointerType>())
	{
		auto scope = pointerType.getAllocationScope();

		if (scope == BufferAllocationScope::stack)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());
		else if (scope == BufferAllocationScope::heap)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());

		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
	}
}

mlir::Type AndOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value AndOp::lhs()
{
	return Adaptor(*this).lhs();
}

mlir::Value AndOp::rhs()
{
	return Adaptor(*this).rhs();
}

//===----------------------------------------------------------------------===//
// Modelica::OrOp
//===----------------------------------------------------------------------===//

mlir::Value OrOpAdaptor::lhs()
{
	return getValues()[0];
}

mlir::Value OrOpAdaptor::rhs()
{
	return getValues()[1];
}

void OrOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::ParseResult OrOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void OrOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " "
					<< lhs() << ", " << rhs() << " : ("
					<< lhs().getType() << ", " << rhs().getType() << ") -> "
					<< resultType();
}

mlir::LogicalResult OrOp::verify()
{
	mlir::Type lhsType = lhs().getType();
	mlir::Type rhsType = rhs().getType();

	if (lhsType.isa<BooleanType>() && rhsType.isa<BooleanType>())
		return mlir::success();

	if (lhsType.isa<PointerType>() && rhsType.isa<PointerType>())
		if (lhsType.cast<PointerType>().getElementType().isa<BooleanType>() &&
				rhsType.cast<PointerType>().getElementType().isa<BooleanType>())
			return mlir::success();

	return emitOpError("requires the operands to be booleans or arrays of booleans");
}

void OrOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (lhs().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());

	if (rhs().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());

	if (auto pointerType = resultType().dyn_cast<PointerType>())
	{
		auto scope = pointerType.getAllocationScope();

		if (scope == BufferAllocationScope::stack)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());
		else if (scope == BufferAllocationScope::heap)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());

		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
	}
}

mlir::Type OrOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value OrOp::lhs()
{
	return Adaptor(*this).lhs();
}

mlir::Value OrOp::rhs()
{
	return Adaptor(*this).rhs();
}

//===----------------------------------------------------------------------===//
// Modelica::EqOp
//===----------------------------------------------------------------------===//

mlir::Value EqOpAdaptor::lhs()
{
	return getValues()[0];
}

mlir::Value EqOpAdaptor::rhs()
{
	return getValues()[1];
}

void EqOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::ParseResult EqOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void EqOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << getOperands() << " : ("
					<< lhs().getType() << ", "
					<< rhs().getType() << ") -> "
					<< getOperation()->getResultTypes()[0];
}

mlir::LogicalResult EqOp::verify()
{
	if (!isNumeric(lhs()) || !isNumeric(rhs()))
		return emitOpError("requires the operands to be scalars of simple types");

	return mlir::success();
}

mlir::Type EqOp::resultType()
{
	return getOperation()->getResultTypes()[0];
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

mlir::Value NotEqOpAdaptor::lhs()
{
	return getValues()[0];
}

mlir::Value NotEqOpAdaptor::rhs()
{
	return getValues()[1];
}

void NotEqOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::ParseResult NotEqOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void NotEqOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << getOperands() << " : ("
					<< lhs().getType() << ", "
					<< rhs().getType() << ") -> "
					<< getOperation()->getResultTypes()[0];
}

mlir::LogicalResult NotEqOp::verify()
{
	if (!isNumeric(lhs()) || !isNumeric(rhs()))
		return emitOpError("requires the operands to be scalars of simple types");

	return mlir::success();
}

mlir::Type NotEqOp::resultType()
{
	return getOperation()->getResultTypes()[0];
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

mlir::Value GtOpAdaptor::lhs()
{
	return getValues()[0];
}

mlir::Value GtOpAdaptor::rhs()
{
	return getValues()[1];
}

void GtOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::ParseResult GtOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void GtOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << getOperands() << " : ("
					<< lhs().getType() << ", "
					<< rhs().getType() << ") -> "
					<< getOperation()->getResultTypes()[0];
}

mlir::LogicalResult GtOp::verify()
{
	if (!isNumeric(lhs()) || !isNumeric(rhs()))
		return emitOpError("requires the operands to be scalars of simple types");

	return mlir::success();
}

mlir::Type GtOp::resultType()
{
	return getOperation()->getResultTypes()[0];
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

mlir::Value GteOpAdaptor::lhs()
{
	return getValues()[0];
}

mlir::Value GteOpAdaptor::rhs()
{
	return getValues()[1];
}

void GteOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::ParseResult GteOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void GteOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << getOperands() << " : ("
					<< lhs().getType() << ", "
					<< rhs().getType() << ") -> "
					<< getOperation()->getResultTypes()[0];
}

mlir::LogicalResult GteOp::verify()
{
	if (!isNumeric(lhs()) || !isNumeric(rhs()))
		return emitOpError("requires the operands to be scalars of simple types");

	return mlir::success();
}

mlir::Type GteOp::resultType()
{
	return getOperation()->getResultTypes()[0];
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

mlir::Value LtOpAdaptor::lhs()
{
	return getValues()[0];
}

mlir::Value LtOpAdaptor::rhs()
{
	return getValues()[1];
}

void LtOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::ParseResult LtOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void LtOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << getOperands() << " : ("
					<< lhs().getType() << ", "
					<< rhs().getType() << ") -> "
					<< getOperation()->getResultTypes()[0];
}

mlir::LogicalResult LtOp::verify()
{
	if (!isNumeric(lhs()) || !isNumeric(rhs()))
		return emitOpError("requires the operands to be scalars of simple types");

	return mlir::success();
}

mlir::Type LtOp::resultType()
{
	return getOperation()->getResultTypes()[0];
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

mlir::Value LteOpAdaptor::lhs()
{
	return getValues()[0];
}

mlir::Value LteOpAdaptor::rhs()
{
	return getValues()[1];
}

void LteOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::ParseResult LteOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void LteOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << getOperands() << " : ("
					<< lhs().getType() << ", "
					<< rhs().getType() << ") -> "
					<< getOperation()->getResultTypes()[0];
}

mlir::LogicalResult LteOp::verify()
{
	if (!isNumeric(lhs()) || !isNumeric(rhs()))
		return emitOpError("requires the operands to be scalars of simple types");

	return mlir::success();
}

mlir::Type LteOp::resultType()
{
	return getOperation()->getResultTypes()[0];
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
// Modelica::NegateOp
//===----------------------------------------------------------------------===//

mlir::Value NegateOpAdaptor::operand()
{
	return getValues()[0];
}

void NegateOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value value)
{
	state.addTypes(resultType);
	state.addOperands(value);
}

mlir::ParseResult NegateOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void NegateOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << operand() << " : "
					<< operand().getType() << " -> " << resultType();
}

void NegateOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (operand().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());

	if (auto pointerType = resultType().dyn_cast<PointerType>())
	{
		auto scope = pointerType.getAllocationScope();

		if (scope == BufferAllocationScope::stack)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());
		else if (scope == BufferAllocationScope::heap)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());

		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
	}
}

mlir::LogicalResult NegateOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	if (argumentIndex > 0)
		return emitError("Index out of bounds: " + std::to_string(argumentIndex));

	if (auto size = currentResult.size(); size != 1)
		return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");

	mlir::Value toNest = currentResult[0];

	mlir::Value nestedOperand = readValue(builder, toNest);
	auto right = builder.create<NegateOp>(getLoc(), nestedOperand.getType(), nestedOperand);

	for (auto& use : toNest.getUses())
		if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
			use.set(right.getResult());

	replaceAllUsesWith(operand());
	getOperation()->remove();

	return mlir::success();
}

mlir::Value NegateOp::distribute(mlir::OpBuilder& builder)
{
	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	if (auto childOp = mlir::dyn_cast<NegateOpDistributionInterface>(operand().getDefiningOp()))
		return childOp.distributeNegateOp(builder, resultType());

	// The operation can't be propagated because the child doesn't
	// know how to distribute the multiplication to its children.
	return getResult();
}

mlir::Value NegateOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeNegateOp(builder, resultType);

		return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
	};

	mlir::Value operand = distributeFn(this->operand());

	return builder.create<NegateOp>(getLoc(), resultType, operand);
}

mlir::Value NegateOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeMulOp(builder, resultType, value);

		return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value operand = distributeFn(this->operand());

	return builder.create<NegateOp>(getLoc(), resultType, operand);
}

mlir::Value NegateOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeMulOp(builder, resultType, value);

		return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value operand = distributeFn(this->operand());

	return builder.create<NegateOp>(getLoc(), resultType, operand);
}

void NegateOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	auto derivedOp = builder.create<NegateOp>(getLoc(), resultType(), operand());
	derivatives.map(getResult(), derivedOp.getResult());
}

mlir::Type NegateOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value NegateOp::operand()
{
	return Adaptor(*this).operand();
}

//===----------------------------------------------------------------------===//
// Modelica::AddOp
//===----------------------------------------------------------------------===//

mlir::Value AddOpAdaptor::lhs()
{
	return getValues()[0];
}

mlir::Value AddOpAdaptor::rhs()
{
	return getValues()[1];
}

void AddOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	if (auto pointerType = resultType.dyn_cast<PointerType>())
		if (pointerType.getAllocationScope() == BufferAllocationScope::unknown)
			resultType = pointerType.toMinAllowedAllocationScope();

	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::ParseResult AddOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> operands;
	llvm::SmallVector<mlir::Type, 3> operandsTypes;
	mlir::Type resultType;

	llvm::SMLoc operandsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(operands, 2) ||
			parser.parseColon() || parser.parseLParen() ||
			parser.parseTypeList(operandsTypes) ||
			parser.parseRParen() || parser.parseArrow() ||
			parser.parseType(resultType) ||
			parser.resolveOperands(operands, operandsTypes, operandsLoc, result.operands))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void AddOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << lhs() << ", " << rhs() << " : ("
					<< lhs().getType() << ", " << rhs().getType()
					<< ") -> " << resultType();
}

void AddOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (lhs().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());

	if (rhs().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());

	if (auto pointerType = resultType().dyn_cast<PointerType>())
	{
		auto scope = pointerType.getAllocationScope();

		if (scope == BufferAllocationScope::stack)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());
		else if (scope == BufferAllocationScope::heap)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());

		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
	}
}

mlir::LogicalResult AddOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	if (auto size = currentResult.size(); size != 1)
		return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");

	mlir::Value toNest = currentResult[0];

	if (argumentIndex == 0)
	{
		mlir::Value nestedOperand = readValue(builder, toNest);
		auto right = builder.create<SubOp>(getLoc(), nestedOperand.getType(), nestedOperand, rhs());

		for (auto& use : toNest.getUses())
			if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
				use.set(right.getResult());

		replaceAllUsesWith(lhs());
		erase();

		return mlir::success();
	}

	if (argumentIndex == 1)
	{
		mlir::Value nestedOperand = readValue(builder, toNest);
		auto right = builder.create<SubOp>(getLoc(), nestedOperand.getType(), nestedOperand, lhs());

		for (auto& use : toNest.getUses())
			if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
				use.set(right.getResult());

		replaceAllUsesWith(rhs());
		erase();

		return mlir::success();
	}

	return emitError("Index out of bounds: " + std::to_string(argumentIndex));
}

mlir::Value AddOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeNegateOp(builder, resultType);

		return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = distributeFn(this->rhs());

	return builder.create<AddOp>(getLoc(), resultType, lhs, rhs);
}

mlir::Value AddOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeMulOp(builder, resultType, value);

		return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = distributeFn(this->rhs());

	return builder.create<AddOp>(getLoc(), resultType, lhs, rhs);
}

mlir::Value AddOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeMulOp(builder, resultType, value);

		return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = distributeFn(this->rhs());

	return builder.create<AddOp>(getLoc(), resultType, lhs, rhs);
}

void AddOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	mlir::Value derivedLhs = derivatives.lookup(lhs());
	mlir::Value derivedRhs = derivatives.lookup(rhs());

	auto derivedOp = builder.create<AddOp>(getLoc(), resultType(), derivedLhs, derivedRhs);
	derivatives.map(getResult(), derivedOp.getResult());
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

//===----------------------------------------------------------------------===//
// Modelica::SubOp
//===----------------------------------------------------------------------===//

mlir::Value SubOpAdaptor::lhs()
{
	return getValues()[0];
}

mlir::Value SubOpAdaptor::rhs()
{
	return getValues()[1];
}

void SubOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	if (auto pointerType = resultType.dyn_cast<PointerType>())
		if (pointerType.getAllocationScope() == BufferAllocationScope::unknown)
			resultType = pointerType.toMinAllowedAllocationScope();

	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::ParseResult SubOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> operands;
	llvm::SmallVector<mlir::Type, 3> operandsTypes;
	mlir::Type resultType;

	llvm::SMLoc operandsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(operands, 2) ||
			parser.parseColon() || parser.parseLParen() ||
			parser.parseTypeList(operandsTypes) ||
			parser.parseRParen() || parser.parseArrow() || parser.parseLParen() ||
			parser.parseType(resultType) || parser.parseRParen() ||
			parser.resolveOperands(operands, operandsTypes, operandsLoc, result.operands))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void SubOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << lhs() << ", " << rhs() << " : ("
					<< lhs().getType() << ", " << rhs().getType()
					<< ") -> " << resultType();
}

void SubOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (lhs().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());

	if (rhs().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());

	if (auto pointerType = resultType().dyn_cast<PointerType>())
	{
		auto scope = pointerType.getAllocationScope();

		if (scope == BufferAllocationScope::stack)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());
		else if (scope == BufferAllocationScope::heap)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());

		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
	}
}

mlir::LogicalResult SubOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	if (auto size = currentResult.size(); size != 1)
		return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");

	mlir::Value toNest = currentResult[0];

	if (argumentIndex == 0)
	{
		mlir::Value nestedOperand = readValue(builder, toNest);
		auto right = builder.create<AddOp>(getLoc(), nestedOperand.getType(), nestedOperand, rhs());

		for (auto& use : toNest.getUses())
			if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
				use.set(right.getResult());

		replaceAllUsesWith(lhs());
		erase();

		return mlir::success();
	}

	if (argumentIndex == 1)
	{
		mlir::Value nestedOperand = readValue(builder, toNest);
		auto right = builder.create<SubOp>(getLoc(), nestedOperand.getType(), lhs(), nestedOperand);

		for (auto& use : toNest.getUses())
			if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
				use.set(right.getResult());

		replaceAllUsesWith(rhs());
		erase();

		return mlir::success();
	}

	return emitError("Index out of bounds: " + std::to_string(argumentIndex));
}

mlir::Value SubOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeNegateOp(builder, resultType);

		return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = distributeFn(this->rhs());

	return builder.create<AddOp>(getLoc(), resultType, lhs, rhs);
}

mlir::Value SubOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeMulOp(builder, resultType, value);

		return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = distributeFn(this->rhs());

	return builder.create<SubOp>(getLoc(), resultType, lhs, rhs);
}

mlir::Value SubOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeMulOp(builder, resultType, value);

		return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = distributeFn(this->rhs());

	return builder.create<SubOp>(getLoc(), resultType, lhs, rhs);
}

void SubOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	mlir::Value derivedLhs = derivatives.lookup(lhs());
	mlir::Value derivedRhs = derivatives.lookup(rhs());

	auto derivedOp = builder.create<SubOp>(getLoc(), resultType(), derivedLhs, derivedRhs);
	derivatives.map(getResult(), derivedOp.getResult());
}

mlir::Type SubOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value SubOp::lhs()
{
	return Adaptor(*this).lhs();
}

mlir::Value SubOp::rhs()
{
	return Adaptor(*this).rhs();
}

//===----------------------------------------------------------------------===//
// Modelica::MulOp
//===----------------------------------------------------------------------===//

mlir::Value MulOpAdaptor::lhs()
{
	return getValues()[0];
}

mlir::Value MulOpAdaptor::rhs()
{
	return getValues()[1];
}

void MulOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	if (auto pointerType = resultType.dyn_cast<PointerType>())
		if (pointerType.getAllocationScope() == BufferAllocationScope::unknown)
			resultType = pointerType.toMinAllowedAllocationScope();

	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::ParseResult MulOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> operands;
	llvm::SmallVector<mlir::Type, 3> operandsTypes;
	mlir::Type resultType;

	llvm::SMLoc operandsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(operands, 2) ||
			parser.parseColon() || parser.parseLParen() ||
			parser.parseTypeList(operandsTypes) ||
			parser.parseRParen() || parser.parseArrow() ||
			parser.parseType(resultType) ||
			parser.resolveOperands(operands, operandsTypes, operandsLoc, result.operands))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void MulOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << lhs() << ", " << rhs() << " : ("
					<< lhs().getType() << ", " << rhs().getType()
					<< ") -> " << resultType();
}

void MulOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (lhs().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());

	if (rhs().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());

	if (auto pointerType = resultType().dyn_cast<PointerType>())
	{
		auto scope = pointerType.getAllocationScope();

		if (scope == BufferAllocationScope::stack)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());
		else if (scope == BufferAllocationScope::heap)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());

		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
	}
}

mlir::LogicalResult MulOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	if (auto size = currentResult.size(); size != 1)
		return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");

	mlir::Value toNest = currentResult[0];

	if (argumentIndex == 0)
	{
		mlir::Value nestedOperand = readValue(builder, toNest);
		auto right = builder.create<DivOp>(getLoc(), nestedOperand.getType(), nestedOperand, rhs());

		for (auto& use : toNest.getUses())
			if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
				use.set(right.getResult());

		replaceAllUsesWith(lhs());
		erase();

		return mlir::success();
	}

	if (argumentIndex == 1)
	{
		mlir::Value nestedOperand = readValue(builder, toNest);
		auto right = builder.create<DivOp>(getLoc(), nestedOperand.getType(), nestedOperand, lhs());

		for (auto& use : toNest.getUses())
			if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
				use.set(right.getResult());

		getResult().replaceAllUsesWith(rhs());
		erase();

		return mlir::success();
	}

	return emitError("Index out of bounds: " + std::to_string(argumentIndex));
}

mlir::Value MulOp::distribute(mlir::OpBuilder& builder)
{
	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	if (!mlir::isa<MulOpDistributionInterface>(lhs().getDefiningOp()) &&
			!mlir::isa<MulOpDistributionInterface>(rhs().getDefiningOp()))
	{
		// The operation can't be propagated because none of the children
		// know how to distribute the multiplication to their children.
		return getResult();
	}

	MulOpDistributionInterface childOp = mlir::isa<MulOpDistributionInterface>(lhs().getDefiningOp()) ?
	    mlir::cast<MulOpDistributionInterface>(lhs().getDefiningOp()) :
	        mlir::cast<MulOpDistributionInterface>(rhs().getDefiningOp());

	mlir::Value toDistribute = mlir::isa<MulOpDistributionInterface>(lhs().getDefiningOp()) ? rhs() : lhs();

	return childOp.distributeMulOp(builder, resultType(), toDistribute);
}

mlir::Value MulOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeNegateOp(builder, resultType);

		return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = this->rhs();

	return builder.create<MulOp>(getLoc(), resultType, lhs, rhs);
}

mlir::Value MulOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeMulOp(builder, resultType, value);

		return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = this->rhs();

	return builder.create<MulOp>(getLoc(), resultType, lhs, rhs);
}

mlir::Value MulOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeMulOp(builder, resultType, value);

		return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = this->rhs();

	return builder.create<MulOp>(getLoc(), resultType, lhs, rhs);
}

void MulOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	mlir::Value derivedLhs = derivatives.lookup(lhs());
	mlir::Value derivedRhs = derivatives.lookup(rhs());

	mlir::Value firstMul = builder.create<MulOp>(getLoc(), resultType(), derivedLhs, rhs());
	mlir::Value secondMul = builder.create<MulOp>(getLoc(), resultType(), lhs(), derivedRhs);

	auto derivedOp = builder.create<AddOp>(getLoc(), resultType(), firstMul, secondMul);
	derivatives.map(getResult(), derivedOp.getResult());
}

mlir::Type MulOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value MulOp::lhs()
{
	return Adaptor(*this).lhs();
}

mlir::Value MulOp::rhs()
{
	return Adaptor(*this).rhs();
}

//===----------------------------------------------------------------------===//
// Modelica::DivOp
//===----------------------------------------------------------------------===//

mlir::Value DivOpAdaptor::lhs()
{
	return getValues()[0];
}

mlir::Value DivOpAdaptor::rhs()
{
	return getValues()[1];
}

void DivOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	if (auto pointerType = resultType.dyn_cast<PointerType>())
		if (pointerType.getAllocationScope() == BufferAllocationScope::unknown)
			resultType = pointerType.toMinAllowedAllocationScope();

	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::ParseResult DivOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> operands;
	llvm::SmallVector<mlir::Type, 3> operandsTypes;
	mlir::Type resultType;

	llvm::SMLoc operandsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(operands, 2) ||
			parser.parseColon() || parser.parseLParen() ||
			parser.parseTypeList(operandsTypes) ||
			parser.parseRParen() || parser.parseArrow() || parser.parseLParen() ||
			parser.parseType(resultType) || parser.parseRParen() ||
			parser.resolveOperands(operands, operandsTypes, operandsLoc, result.operands))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void DivOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << lhs() << ", " << rhs() << " : ("
					<< lhs().getType() << ", " << rhs().getType()
					<< ") -> " << resultType();
}

void DivOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (lhs().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());

	if (rhs().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());

	if (auto pointerType = resultType().dyn_cast<PointerType>())
	{
		auto scope = pointerType.getAllocationScope();

		if (scope == BufferAllocationScope::stack)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());
		else if (scope == BufferAllocationScope::heap)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());

		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
	}
}

mlir::LogicalResult DivOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	if (auto size = currentResult.size(); size != 1)
		return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");

	mlir::Value toNest = currentResult[0];

	if (argumentIndex == 0)
	{
		mlir::Value nestedOperand = readValue(builder, toNest);
		auto right = builder.create<MulOp>(getLoc(), nestedOperand.getType(), nestedOperand, rhs());

		for (auto& use : toNest.getUses())
			if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
				use.set(right.getResult());

		getResult().replaceAllUsesWith(lhs());
		erase();

		return mlir::success();
	}

	if (argumentIndex == 1)
	{
		mlir::Value nestedOperand = readValue(builder, toNest);
		auto right = builder.create<DivOp>(getLoc(), nestedOperand.getType(), lhs(), nestedOperand);

		for (auto& use : toNest.getUses())
			if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
				use.set(right.getResult());

		getResult().replaceAllUsesWith(rhs());
		erase();

		return mlir::success();
	}

	return emitError("Index out of bounds: " + std::to_string(argumentIndex));
}

mlir::Value DivOp::distribute(mlir::OpBuilder& builder)
{
	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	if (!mlir::isa<DivOpDistributionInterface>(lhs().getDefiningOp()) &&
			!mlir::isa<DivOpDistributionInterface>(rhs().getDefiningOp()))
	{
		// The operation can't be propagated because none of the children
		// know how to distribute the multiplication to their children.
		return getResult();
	}

	DivOpDistributionInterface childOp = mlir::isa<DivOpDistributionInterface>(lhs().getDefiningOp()) ?
																			 mlir::cast<DivOpDistributionInterface>(lhs().getDefiningOp()) :
																			 mlir::cast<DivOpDistributionInterface>(rhs().getDefiningOp());

	mlir::Value toDistribute = mlir::isa<DivOpDistributionInterface>(lhs().getDefiningOp()) ? rhs() : lhs();

	return childOp.distributeDivOp(builder, resultType(), toDistribute);
}

mlir::Value DivOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeNegateOp(builder, resultType);

		return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = this->rhs();

	return builder.create<MulOp>(getLoc(), resultType, lhs, rhs);
}

mlir::Value DivOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeMulOp(builder, resultType, value);

		return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = this->rhs();

	return builder.create<DivOp>(getLoc(), resultType, lhs, rhs);
}

mlir::Value DivOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeMulOp(builder, resultType, value);

		return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = this->rhs();

	return builder.create<DivOp>(getLoc(), resultType, lhs, rhs);
}

void DivOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	// TODO: DivOp derivation
}

mlir::Type DivOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value DivOp::lhs()
{
	return Adaptor(*this).lhs();
}

mlir::Value DivOp::rhs()
{
	return Adaptor(*this).rhs();
}

//===----------------------------------------------------------------------===//
// Modelica::PowOp
//===----------------------------------------------------------------------===//

/**
 * Pow optimizations:
 *  - if the exponent is 1, the result is just the base value.
 *  - if the exponent is 2, change the pow operation into a multiplication.
 */
struct PowOpOptimizationPattern : public mlir::OpRewritePattern<PowOp>
{
	using OpRewritePattern<PowOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(PowOp op, mlir::PatternRewriter& rewriter) const override
	{
		auto* exponentOp = op.exponent().getDefiningOp();

		// If the exponent is a block argument, then it has no defining op
		// and the method will return a nullptr.

		while (exponentOp != nullptr && mlir::isa<CastOp>(exponentOp))
			exponentOp = mlir::cast<CastOp>(exponentOp).value().getDefiningOp();

		if (exponentOp == nullptr || !mlir::isa<ConstantOp>(exponentOp))
			return rewriter.notifyMatchFailure(op, "Exponent is not a constant");

		auto constant = mlir::cast<ConstantOp>(exponentOp);
		mlir::Attribute attribute = constant.value();
		long exponent = -1;

		if (attribute.isa<mlir::IntegerAttr>())
			exponent = attribute.cast<mlir::IntegerAttr>().getSInt();
		else if (attribute.isa<BooleanAttribute>())
			exponent = attribute.cast<BooleanAttribute>().getValue() ? 1 : 0;
		else if (attribute.isa<IntegerAttribute>())
			exponent = attribute.cast<IntegerAttribute>().getValue();
		else if (attribute.isa<RealAttribute>())
		{
			double value = attribute.cast<RealAttribute>().getValue();

			if (auto ceiled = ceil(value); ceiled == value)
				exponent = ceiled;
		}

		if (exponent == 1)
		{
			rewriter.replaceOpWithNewOp<CastOp>(op, op.base(), op.resultType());
			return mlir::success();
		}

		if (exponent == 2)
		{
			rewriter.replaceOpWithNewOp<MulOp>(op, op.resultType(), op.base(), op.base());
			return mlir::success();
		}

		return rewriter.notifyMatchFailure(op, "No optimization can be applied");
	}
};

mlir::Value PowOpAdaptor::base()
{
	return getValues()[0];
}

mlir::Value PowOpAdaptor::exponent()
{
	return getValues()[1];
}

void PowOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value base, mlir::Value exponent)
{
	if (auto pointerType = resultType.dyn_cast<PointerType>())
		if (pointerType.getAllocationScope() == BufferAllocationScope::unknown)
			resultType = pointerType.toMinAllowedAllocationScope();

	state.addTypes(resultType);
	state.addOperands({ base, exponent });
}

mlir::ParseResult PowOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> operands;
	llvm::SmallVector<mlir::Type, 3> operandsTypes;
	mlir::Type resultType;

	llvm::SMLoc operandsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(operands, 2) ||
			parser.parseColon() || parser.parseLParen() ||
			parser.parseTypeList(operandsTypes) ||
			parser.parseRParen() || parser.parseArrow() || parser.parseLParen() ||
			parser.parseType(resultType) || parser.parseRParen() ||
			parser.resolveOperands(operands, operandsTypes, operandsLoc, result.operands))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void PowOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << base() << ", " << exponent() << " : ("
					<< base().getType() << ", " << exponent().getType()
					<< ") -> " << resultType();
}

void PowOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context)
{
	patterns.insert<PowOpOptimizationPattern>(context);
}

void PowOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (base().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), base(), mlir::SideEffects::DefaultResource::get());

	if (exponent().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), exponent(), mlir::SideEffects::DefaultResource::get());

	if (auto pointerType = resultType().dyn_cast<PointerType>())
	{
		auto scope = pointerType.getAllocationScope();

		if (scope == BufferAllocationScope::stack)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());
		else if (scope == BufferAllocationScope::heap)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());

		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
	}
}

mlir::Type PowOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value PowOp::base()
{
	return Adaptor(*this).base();
}

mlir::Value PowOp::exponent()
{
	return Adaptor(*this).exponent();
}

//===----------------------------------------------------------------------===//
// Modelica::NDimsOp
//===----------------------------------------------------------------------===//

mlir::Value NDimsOpAdaptor::memory()
{
	return getValues()[0];
}

void NDimsOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value memory)
{
	state.addTypes(resultType);
	state.addOperands(memory);
}

mlir::ParseResult NDimsOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::OpAsmParser::OperandType array;
	mlir::Type arrayType;
	mlir::Type resultType;

	if (parser.parseOperand(array) ||
			parser.parseColon() ||
			parser.parseType(arrayType) ||
			parser.resolveOperand(array, arrayType, result.operands) ||
			parser.parseArrow() ||
			parser.parseType(resultType))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void NDimsOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << memory() << " : " << memory().getType() << " -> " << resultType();
}

mlir::Type NDimsOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value NDimsOp::memory()
{
	return Adaptor(*this).memory();
}

//===----------------------------------------------------------------------===//
// Modelica::SizeOp
//===----------------------------------------------------------------------===//

mlir::Value SizeOpAdaptor::memory()
{
	return getValues()[0];
}

mlir::Value SizeOpAdaptor::index()
{
	return getValues()[1];
}

void SizeOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value memory, mlir::Value index)
{
	if (auto pointerType = resultType.dyn_cast<PointerType>())
		resultType = pointerType.toAllocationScope(BufferAllocationScope::heap);

	state.addTypes(resultType);
	state.addOperands(memory);

	if (index != nullptr)
		state.addOperands(index);
}

void SizeOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.size " << memory();

	if (hasIndex())
		printer << "[" << index() << "]";

	printer << " : " << resultType();
}

mlir::LogicalResult SizeOp::verify()
{
	if (!memory().getType().isa<PointerType>())
		return emitOpError("requires the operand to be an array");

	return mlir::success();
}

void SizeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (auto pointerType = resultType().dyn_cast<PointerType>())
		if (pointerType.getAllocationScope() == BufferAllocationScope::heap)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());

	effects.emplace_back(mlir::MemoryEffects::Read::get(), memory(), mlir::SideEffects::DefaultResource::get());
}

bool SizeOp::hasIndex()
{
	return getNumOperands() == 2;
}

mlir::Type SizeOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value SizeOp::memory()
{
	return Adaptor(*this).memory();
}

mlir::Value SizeOp::index()
{
	return Adaptor(*this).index();
}

//===----------------------------------------------------------------------===//
// Modelica::IdentityOp
//===----------------------------------------------------------------------===//

mlir::Value IdentityOpAdaptor::size()
{
	return getValues()[0];
}

void IdentityOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value size)
{
	state.addTypes(resultType);
	state.addOperands(size);
}

mlir::ParseResult IdentityOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::OpAsmParser::OperandType size;
	mlir::Type sizeType;
	mlir::Type resultType;

	if (parser.parseOperand(size) ||
			parser.parseColon() ||
			parser.parseType(sizeType) ||
			parser.resolveOperand(size, sizeType, result.operands) ||
			parser.parseArrow() ||
			parser.parseType(resultType))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void IdentityOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " "
					<< size() << " : " << size().getType() << " -> " << resultType();
}

mlir::LogicalResult IdentityOp::verify()
{
	if (!size().getType().isa<IntegerType, mlir::IndexType>())
		return emitOpError("requires the size to be an integer value");

	return mlir::success();
}

void IdentityOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (auto pointerType = resultType().dyn_cast<PointerType>())
		if (pointerType.getAllocationScope() == BufferAllocationScope::heap)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
}

mlir::Type IdentityOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value IdentityOp::size()
{
	return Adaptor(*this).size();
}

//===----------------------------------------------------------------------===//
// Modelica::DiagonalOp
//===----------------------------------------------------------------------===//

mlir::Value DiagonalOpAdaptor::values()
{
	return getValues()[0];
}

void DiagonalOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value values)
{
	state.addTypes(resultType);
	state.addOperands(values);
}

mlir::ParseResult DiagonalOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::OpAsmParser::OperandType array;
	mlir::Type arrayType;
	mlir::Type resultType;

	if (parser.parseOperand(array) ||
			parser.parseColon() ||
			parser.parseType(arrayType) ||
			parser.resolveOperand(array, arrayType, result.operands) ||
			parser.parseArrow() ||
			parser.parseType(resultType))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void DiagonalOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " "
					<< values() << " : " << values().getType() << " -> " << resultType();
}

mlir::LogicalResult DiagonalOp::verify()
{
	if (!values().getType().isa<PointerType>())
		return emitOpError("requires the values to be an array");

	if (auto pointerType = values().getType().cast<PointerType>(); pointerType.getRank() != 1)
		return emitOpError("requires the values array to have rank 1");

	return mlir::success();
}

void DiagonalOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (auto pointerType = resultType().dyn_cast<PointerType>())
		if (pointerType.getAllocationScope() == BufferAllocationScope::heap)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
}

mlir::Type DiagonalOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value DiagonalOp::values()
{
	return Adaptor(*this).values();
}

//===----------------------------------------------------------------------===//
// Modelica::ZerosOp
//===----------------------------------------------------------------------===//

mlir::ValueRange ZerosOpAdaptor::sizes()
{
	return getValues();
}

void ZerosOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::ValueRange sizes)
{
	state.addTypes(resultType);
	state.addOperands(sizes);
}

mlir::ParseResult ZerosOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> dimensions;
	llvm::SmallVector<mlir::Type, 3> dimensionsTypes;
	mlir::Type resultType;

	llvm::SMLoc dimensionsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(dimensions) ||
			parser.parseColon())
		return mlir::failure();

	llvm::SMLoc dimensionsTypesLoc = parser.getCurrentLocation();

	if (dimensions.size() > 1)
    if (parser.parseLParen())
			return mlir::failure();

	if (parser.parseTypeList(dimensionsTypes))
		return mlir::failure();

	if (dimensions.size() > 1)
		if (parser.parseRParen())
			return mlir::failure();

	if (parser.resolveOperands(dimensions, dimensionsTypes, dimensionsLoc, result.operands) ||
		 parser.parseArrow() ||
		 parser.parseType(resultType))
		return mlir::failure();

	if (dimensions.size() != dimensionsTypes.size())
		return parser.emitError(dimensionsTypesLoc)
				<< "expected as many input types as operands "
				<< "(expected " << dimensions.size() << " got "
				<< dimensionsTypes.size() << ")";

	result.addTypes(resultType);
	return mlir::success();
}

void ZerosOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << sizes() << " : ";
	auto sizesTypes = sizes().getTypes();

	if (sizesTypes.size() > 1)
		printer << "(";

	printer << sizesTypes;

	if (sizesTypes.size() > 1)
		printer << ")";

	printer << " -> " << resultType();
}

mlir::LogicalResult ZerosOp::verify()
{
	if (!resultType().isa<PointerType>())
		return emitOpError("requires the result to be an array");

	if (auto pointerType = resultType().cast<PointerType>(); pointerType.getRank() != sizes().size())
		return emitOpError("requires the rank of the result array to match the sizes amount");

	return mlir::success();
}

void ZerosOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (auto pointerType = resultType().dyn_cast<PointerType>())
		if (pointerType.getAllocationScope() == BufferAllocationScope::heap)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
}

mlir::Type ZerosOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::ValueRange ZerosOp::sizes()
{
	return Adaptor(*this).sizes();
}

//===----------------------------------------------------------------------===//
// Modelica::OnesOp
//===----------------------------------------------------------------------===//

mlir::ValueRange OnesOpAdaptor::sizes()
{
	return getValues();
}

void OnesOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::ValueRange sizes)
{
	state.addTypes(resultType);
	state.addOperands(sizes);
}

mlir::ParseResult OnesOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> dimensions;
	llvm::SmallVector<mlir::Type, 3> dimensionsTypes;
	mlir::Type resultType;

	llvm::SMLoc dimensionsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(dimensions) ||
			parser.parseColon())
		return mlir::failure();

	llvm::SMLoc dimensionsTypesLoc = parser.getCurrentLocation();

	if (dimensions.size() > 1)
		if (parser.parseLParen())
			return mlir::failure();

	if (parser.parseTypeList(dimensionsTypes))
		return mlir::failure();

	if (dimensions.size() > 1)
		if (parser.parseRParen())
			return mlir::failure();

	if (parser.resolveOperands(dimensions, dimensionsTypes, dimensionsLoc, result.operands) ||
			parser.parseArrow() ||
			parser.parseType(resultType))
		return mlir::failure();

	if (dimensions.size() != dimensionsTypes.size())
		return parser.emitError(dimensionsTypesLoc)
				<< "expected as many input types as operands "
				<< "(expected " << dimensions.size() << " got "
				<< dimensionsTypes.size() << ")";

	result.addTypes(resultType);
	return mlir::success();
}

void OnesOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << sizes() << " : ";
	auto sizesTypes = sizes().getTypes();

	if (sizesTypes.size() > 1)
		printer << "(";

	printer << sizesTypes;

	if (sizesTypes.size() > 1)
		printer << ")";

	printer << " -> " << resultType();
}

mlir::LogicalResult OnesOp::verify()
{
	if (!resultType().isa<PointerType>())
		return emitOpError("requires the result to be an array");

	if (auto pointerType = resultType().cast<PointerType>(); pointerType.getRank() != sizes().size())
		return emitOpError("requires the rank of the result array to match the sizes amount");

	return mlir::success();
}

void OnesOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (auto pointerType = resultType().dyn_cast<PointerType>())
		if (pointerType.getAllocationScope() == BufferAllocationScope::heap)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
}

mlir::Type OnesOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::ValueRange OnesOp::sizes()
{
	return Adaptor(*this).sizes();
}

//===----------------------------------------------------------------------===//
// Modelica::LinspaceOp
//===----------------------------------------------------------------------===//

mlir::Value LinspaceOpAdaptor::start()
{
	return getValues()[0];
}

mlir::Value LinspaceOpAdaptor::end()
{
	return getValues()[1];
}

mlir::Value LinspaceOpAdaptor::steps()
{
	return getValues()[2];
}

void LinspaceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value start, mlir::Value end, mlir::Value steps)
{
	state.addTypes(resultType);
	state.addOperands(start);
	state.addOperands(end);
	state.addOperands(steps);
}

mlir::ParseResult LinspaceOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> operands;
	llvm::SmallVector<mlir::Type, 3> operandsTypes;
	mlir::Type resultType;

	llvm::SMLoc operandsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(operands, 3) ||
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

void LinspaceOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " "
					<< start() << ", " << end() << ", " << steps() << " : ("
					<< start().getType() << ", "
					<< end().getType() << ", "
					<< steps().getType() << ") -> "
					<< resultType();
}

mlir::LogicalResult LinspaceOp::verify()
{
	if (!resultType().isa<PointerType>())
		return emitOpError("requires the result to be an array");

	if (auto pointerType = resultType().cast<PointerType>(); pointerType.getRank() != 1)
		return emitOpError("requires the result array to have rank 1");

	return mlir::success();
}

void LinspaceOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (auto pointerType = resultType().dyn_cast<PointerType>())
		if (pointerType.getAllocationScope() == BufferAllocationScope::heap)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
}

mlir::Type LinspaceOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value LinspaceOp::start()
{
	return Adaptor(*this).start();
}

mlir::Value LinspaceOp::end()
{
	return Adaptor(*this).end();
}

mlir::Value LinspaceOp::steps()
{
	return Adaptor(*this).steps();
}

//===----------------------------------------------------------------------===//
// Modelica::FillOp
//===----------------------------------------------------------------------===//

mlir::Value FillOpAdaptor::value()
{
	return getValues()[0];
}

mlir::Value FillOpAdaptor::memory()
{
	return getValues()[1];
}

void FillOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value value, mlir::Value memory)
{
	state.addOperands(value);
	state.addOperands(memory);
}

mlir::ParseResult FillOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 2> operands;
	llvm::SmallVector<mlir::Type, 2> operandsTypes;
	mlir::Type resultType;

	if (parser.parseOperandList(operands, 2) ||
			parser.parseColonTypeList(operandsTypes))
		return mlir::failure();

	return mlir::success();
}

void FillOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " "
					<< value() << ", " << memory() << " : "
					<< value().getType() << ", " << memory().getType();
}

mlir::Value FillOp::value()
{
	return Adaptor(*this).value();
}

mlir::Value FillOp::memory()
{
	return Adaptor(*this).memory();
}

//===----------------------------------------------------------------------===//
// Modelica::MinOp
//===----------------------------------------------------------------------===//

mlir::ValueRange MinOpAdaptor::values()
{
	return getValues();
}

void MinOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::ValueRange values)
{
	state.addTypes(resultType);
	state.addOperands(values);
}

mlir::ParseResult MinOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 2> operands;
	llvm::SmallVector<mlir::Type, 2> operandsTypes;
	mlir::Type resultType;

	llvm::SMLoc operandsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(operands) ||
			parser.parseColon())
		return mlir::failure();

	if (operands.empty())
		return parser.emitError(operandsLoc) << "expected at least one operand";

	llvm::SMLoc dimensionsTypesLoc = parser.getCurrentLocation();

	if (operands.size() > 1)
		if (parser.parseLParen())
			return mlir::failure();

	if (parser.parseTypeList(operandsTypes))
		return mlir::failure();

	if (operands.size() > 1)
		if (parser.parseRParen())
			return mlir::failure();

	if (parser.resolveOperands(operands, operandsTypes, operandsLoc, result.operands) ||
			parser.parseArrow() ||
			parser.parseType(resultType))
		return mlir::failure();

	if (operands.size() != operandsTypes.size())
		return parser.emitError(dimensionsTypesLoc)
				<< "expected as many input types as operands "
				<< "(expected " << operands.size() << " got "
				<< operandsTypes.size() << ")";

	result.addTypes(resultType);
	return mlir::success();
}

void MinOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << values() << " : ";
	auto valuesTypes = values().getTypes();

	if (valuesTypes.size() > 1)
		printer << "(";

	printer << valuesTypes;

	if (valuesTypes.size() > 1)
		printer << ")";

	printer << " -> " << resultType();
}

mlir::LogicalResult MinOp::verify()
{
	if (getNumOperands() == 1)
	{
		if (auto pointerType = values()[0].getType().dyn_cast<PointerType>();
				pointerType && isNumeric(pointerType.getElementType()))
			return mlir::success();
	}
	else if (getNumOperands() == 2)
	{
		if (isNumeric(values()[0]) && isNumeric(values()[1]))
			return mlir::success();
	}

	return emitOpError("requires the operands to be two scalars or one array");
}

mlir::Type MinOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::ValueRange MinOp::values()
{
	return Adaptor(*this).values();
}

//===----------------------------------------------------------------------===//
// Modelica::MaxOp
//===----------------------------------------------------------------------===//

mlir::ValueRange MaxOpAdaptor::values()
{
	return getValues();
}

void MaxOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::ValueRange values)
{
	state.addTypes(resultType);
	state.addOperands(values);
}

mlir::ParseResult MaxOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 2> operands;
	llvm::SmallVector<mlir::Type, 2> operandsTypes;
	mlir::Type resultType;

	llvm::SMLoc operandsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(operands) ||
			parser.parseColon())
		return mlir::failure();

	if (operands.empty())
		return parser.emitError(operandsLoc) << "expected at least one operand";

	llvm::SMLoc dimensionsTypesLoc = parser.getCurrentLocation();

	if (operands.size() > 1)
		if (parser.parseLParen())
			return mlir::failure();

	if (parser.parseTypeList(operandsTypes))
		return mlir::failure();

	if (operands.size() > 1)
		if (parser.parseRParen())
			return mlir::failure();

	if (parser.resolveOperands(operands, operandsTypes, operandsLoc, result.operands) ||
			parser.parseArrow() ||
			parser.parseType(resultType))
		return mlir::failure();

	if (operands.size() != operandsTypes.size())
		return parser.emitError(dimensionsTypesLoc)
				<< "expected as many input types as operands "
				<< "(expected " << operands.size() << " got "
				<< operandsTypes.size() << ")";

	result.addTypes(resultType);
	return mlir::success();
}

void MaxOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << values() << " : ";
	auto valuesTypes = values().getTypes();

	if (valuesTypes.size() > 1)
		printer << "(";

	printer << valuesTypes;

	if (valuesTypes.size() > 1)
		printer << ")";

	printer << " -> " << resultType();
}

mlir::LogicalResult MaxOp::verify()
{
	if (getNumOperands() == 1)
	{
		if (auto pointerType = values()[0].getType().dyn_cast<PointerType>();
				pointerType && isNumeric(pointerType.getElementType()))
			return mlir::success();
	}
	else if (getNumOperands() == 2)
	{
		if (isNumeric(values()[0]) && isNumeric(values()[1]))
			return mlir::success();
	}

	return emitOpError("requires the operands to be two scalars or one array");
}

mlir::Type MaxOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::ValueRange MaxOp::values()
{
	return Adaptor(*this).values();
}

//===----------------------------------------------------------------------===//
// Modelica::SumOp
//===----------------------------------------------------------------------===//

mlir::Value SumOpAdaptor::array()
{
	return getValues()[0];
}

void SumOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value array)
{
	state.addTypes(resultType);
	state.addOperands(array);
}

mlir::ParseResult SumOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::OpAsmParser::OperandType array;
	mlir::Type arrayType;
	mlir::Type resultType;

	if (parser.parseOperand(array) ||
			parser.parseColon() ||
			parser.parseType(resultType) ||
			parser.resolveOperand(array, arrayType, result.operands) ||
			parser.parseArrow() ||
			parser.parseType(resultType))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void SumOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << array() << " : "
					<< array().getType() << " -> " << resultType();
}

mlir::LogicalResult SumOp::verify()
{
	if (!array().getType().isa<PointerType>())
		return emitOpError("requires the operand to be an array");

	if (!isNumeric(array().getType().cast<PointerType>().getElementType()))
		return emitOpError("requires the operand to be an array of numeric values");

	return mlir::success();
}

mlir::Type SumOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value SumOp::array()
{
	return Adaptor(*this).array();
}

//===----------------------------------------------------------------------===//
// Modelica::ProductOp
//===----------------------------------------------------------------------===//

mlir::Value ProductOpAdaptor::array()
{
	return getValues()[0];
}

void ProductOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value array)
{
	state.addTypes(resultType);
	state.addOperands(array);
}

mlir::ParseResult ProductOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::OpAsmParser::OperandType array;
	mlir::Type arrayType;
	mlir::Type resultType;

	if (parser.parseOperand(array) ||
			parser.parseColon() ||
			parser.parseType(resultType) ||
			parser.resolveOperand(array, arrayType, result.operands) ||
			parser.parseArrow() ||
			parser.parseType(resultType))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void ProductOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << array() << " : "
					<< array().getType() << " -> " << resultType();
}

mlir::LogicalResult ProductOp::verify()
{
	if (!array().getType().isa<PointerType>())
		return emitOpError("requires the operand to be an array");

	if (!isNumeric(array().getType().cast<PointerType>().getElementType()))
		return emitOpError("requires the operand to be an array of numeric values");

	return mlir::success();
}

mlir::Type ProductOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value ProductOp::array()
{
	return Adaptor(*this).array();
}

//===----------------------------------------------------------------------===//
// Modelica::TransposeOp
//===----------------------------------------------------------------------===//

mlir::Value TransposeOpAdaptor::matrix()
{
	return getValues()[0];
}

void TransposeOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value matrix)
{
	state.addTypes(resultType);
	state.addOperands(matrix);
}

mlir::ParseResult TransposeOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::OpAsmParser::OperandType matrix;
	mlir::Type matrixType;
	mlir::Type resultType;

	if (parser.parseOperand(matrix) ||
			parser.parseColon() ||
			parser.parseType(matrixType) ||
			parser.resolveOperand(matrix, matrixType, result.operands) ||
			parser.parseArrow() ||
			parser.parseType(resultType))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void TransposeOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << matrix() << " : "
					<< matrix().getType() << " -> " << resultType();
}

mlir::LogicalResult TransposeOp::verify()
{
	if (!matrix().getType().isa<PointerType>())
		return emitOpError("requires the source to be an array");

	auto sourceType = matrix().getType().cast<PointerType>();

	if (sourceType.getRank() != 2)
		return emitOpError("requires the source to have rank 2");

	if (!resultType().isa<PointerType>())
		return emitOpError("requires the result to be an array");

	auto destinationType = resultType().cast<PointerType>();

	if (destinationType.getRank() != 2)
		return emitOpError("requires the destination to have rank 2");

	// Check if the dimensions are compatible
	auto sourceShape = sourceType.getShape();
	auto destinationShape = destinationType.getShape();

	if (sourceShape[0] != -1 && destinationShape[1] != -1 &&
			sourceShape[0] != destinationShape[1])
		return emitOpError("requires compatible shapes");

	if (sourceShape[1] != -1 && destinationShape[0] != -1 &&
			sourceShape[1] != destinationShape[0])
		return emitOpError("requires compatible shapes");

	return mlir::success();
}

void TransposeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (auto pointerType = resultType().dyn_cast<PointerType>())
		if (pointerType.getAllocationScope() == BufferAllocationScope::heap)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
}

mlir::Type TransposeOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value TransposeOp::matrix()
{
	return Adaptor(*this).matrix();
}

//===----------------------------------------------------------------------===//
// Modelica::SymmetricOp
//===----------------------------------------------------------------------===//

mlir::Value SymmetricOpAdaptor::matrix()
{
	return getValues()[0];
}

void SymmetricOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value matrix)
{
	state.addTypes(resultType);
	state.addOperands(matrix);
}

mlir::ParseResult SymmetricOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::OpAsmParser::OperandType matrix;
	mlir::Type matrixType;
	mlir::Type resultType;

	if (parser.parseOperand(matrix) ||
			parser.parseColon() ||
			parser.parseType(matrixType) ||
			parser.resolveOperand(matrix, matrixType, result.operands) ||
			parser.parseArrow() ||
			parser.parseType(resultType))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void SymmetricOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << matrix() << " : "
					<< matrix().getType() << " -> " << resultType();
}

mlir::LogicalResult SymmetricOp::verify()
{
	if (!matrix().getType().isa<PointerType>())
		return emitOpError("requires the source to be an array");

	auto sourceType = matrix().getType().cast<PointerType>();

	if (sourceType.getRank() != 2)
		return emitOpError("requires the source to have rank 2");

	if (!resultType().isa<PointerType>())
		return emitOpError("requires the result to be an array");

	auto destinationType = resultType().cast<PointerType>();

	if (destinationType.getRank() != 2)
		return emitOpError("requires the destination to have rank 2");

	// Check if the dimensions are compatible
	auto sourceShape = sourceType.getShape();
	auto destinationShape = destinationType.getShape();

	if (sourceShape[0] != -1 && destinationShape[0] != -1 &&
			sourceShape[0] != destinationShape[0])
		return emitOpError("requires compatible shapes");

	if (sourceShape[1] != -1 && destinationShape[1] != -1 &&
			sourceShape[1] != destinationShape[1])
		return emitOpError("requires compatible shapes");

	return mlir::success();
}

void SymmetricOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (auto pointerType = resultType().dyn_cast<PointerType>())
		if (pointerType.getAllocationScope() == BufferAllocationScope::heap)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::DefaultResource::get());
}

mlir::Type SymmetricOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value SymmetricOp::matrix()
{
	return Adaptor(*this).matrix();
}

//===----------------------------------------------------------------------===//
// Modelica::DerOp
//===----------------------------------------------------------------------===//

mlir::Value DerOpAdaptor::operand()
{
	return getValues()[0];
}

void DerOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand)
{
	state.addTypes(resultType);
	state.addOperands(operand);
}

mlir::ParseResult DerOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void DerOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << operand() << " : "
					<< operand().getType() << " -> " << resultType();
}

mlir::Type DerOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value DerOp::operand()
{
	return Adaptor(*this).operand();
}

//===----------------------------------------------------------------------===//
// Modelica::PrintOp
//===----------------------------------------------------------------------===//

mlir::ValueRange PrintOpAdaptor::values()
{
	return getValues();
}

void PrintOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange values)
{
	state.addOperands(values);
}

mlir::ParseResult PrintOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> operands;
	llvm::SmallVector<mlir::Type, 3> operandsTypes;
	mlir::Type resultType;

	llvm::SMLoc operandsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(operands) ||
			parser.parseColonTypeList(operandsTypes) ||
			parser.resolveOperands(operands, operandsTypes, operandsLoc, result.operands))
		return mlir::failure();

	return mlir::success();
}

void PrintOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << values() << " : " << values().getTypes();
}

mlir::ValueRange PrintOp::values()
{
	return Adaptor(*this).values();
}

void PrintOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::SideEffects::DefaultResource::get());

	for (mlir::Value value : values())
		if (value.getType().isa<PointerType>())
			effects.emplace_back(mlir::MemoryEffects::Read::get(), value, mlir::SideEffects::DefaultResource::get());
}
