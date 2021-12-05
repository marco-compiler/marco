#include <mlir/Conversion/Passes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/FunctionImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <marco/mlirlowerer/dialects/modelica/Attribute.h>
#include <marco/mlirlowerer/dialects/modelica/Ops.h>

using namespace marco::codegen::modelica;

static bool isNumeric(mlir::Type type)
{
	return type.isa<mlir::IndexType, BooleanType, IntegerType, RealType>();
}

static bool isNumeric(mlir::Value value)
{
	return isNumeric(value.getType());
}

static mlir::Type convertToRealType(mlir::Type type)
{
	if (auto arrayType = type.dyn_cast<ArrayType>())
		return arrayType.toElementType(RealType::get(type.getContext()));

	return RealType::get(type.getContext());
}

static mlir::Value readValue(mlir::OpBuilder& builder, mlir::Value operand)
{
	if (auto arrayType = operand.getType().dyn_cast<ArrayType>(); arrayType && arrayType.getRank() == 0)
		return builder.create<LoadOp>(operand.getLoc(), operand);

	return operand;
}

static void populateAllocationEffects(
        mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects,
        mlir::Value value,
        bool isManuallyDeallocated = false)
{
  if (auto arrayType = value.getType().dyn_cast<ArrayType>())
  {
    auto allocationScope = arrayType.getAllocationScope();
    assert(allocationScope == BufferAllocationScope::stack || allocationScope == BufferAllocationScope::heap);

    if (allocationScope == BufferAllocationScope::stack)
    {
      // Stack-allocated arrays are automatically deallocated when the
      // surrounding function ends.

      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), value, mlir::SideEffects::AutomaticAllocationScopeResource::get());
    }
    else if (allocationScope == BufferAllocationScope::heap)
    {
      // We need to check if there exists a clone operation with forwarding
      // enabled and whose result is manually deallocated. If that is the
      // case, then also the original buffer must not be deallocated, or a
      // double free would happen.

      bool isForwardedAsManuallyDeallocated = llvm::any_of(value.getUsers(), [](const auto& op) -> bool {
         if (auto cloneOp = mlir::dyn_cast<ArrayCloneOp>(op))
           return cloneOp.canSourceBeForwarded() && !cloneOp.shouldBeFreed();

         return false;
      });

      if (!isManuallyDeallocated && !isForwardedAsManuallyDeallocated)
      {
        // Mark the value as heap-allocated so that the deallocation pass can
        // place the deallocation instruction.

        effects.emplace_back(mlir::MemoryEffects::Allocate::get(), value, mlir::SideEffects::DefaultResource::get());
      }
      else
      {
        // If the buffer is marked as manually deallocated, then we need to
        // set the operation to have a generic side effect, or the CSE pass
        // would otherwise consider all the allocations with the same
        // structure as equal, and thus would replace all the subsequent
        // buffers with the first allocated one.

        effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::SideEffects::DefaultResource::get());
      }
    }
  }
}

static std::string getPartialDerFunctionName(llvm::StringRef baseName)
{
	return "pder_" + baseName.str();
}

static mlir::Type getMostGenericType(mlir::Type x, mlir::Type y)
{
	if (x.isa<BooleanType>())
		return y;

	if (y.isa<BooleanType>())
		return x;

	if (x.isa<RealType>())
		return x;

	if (y.isa<RealType>())
		return y;

	if (x.isa<mlir::IndexType>())
		return x;

	if (y.isa<mlir::IndexType>())
		return y;

	if (x.isa<IntegerType>())
		return x;

	return y;
}

static double getAttributeValue(mlir::Attribute attribute)
{
	if (IntegerAttribute integer = attribute.dyn_cast<IntegerAttribute>())
		return integer.getValue();

	if (RealAttribute real = attribute.dyn_cast<RealAttribute>())
		return real.getValue();

	assert(attribute.getType().isa<mlir::IndexType>());

	return attribute.cast<mlir::IntegerAttr>().getInt();
}

static mlir::Attribute getAttribute(mlir::OpBuilder& builder, mlir::Type type, double value)
{
	if (type.isa<BooleanType>())
		return BooleanAttribute::get(type, value > 0);

	if (type.isa<IntegerType>())
		return IntegerAttribute::get(type, value);

	if (type.isa<RealType>())
		return RealAttribute::get(type, value);

	return builder.getIndexAttr(value);
}

static bool isOperandFoldable(mlir::Value operand)
{
	if (operand.isa<mlir::BlockArgument>())
		return false;

	if (!mlir::isa<ConstantOp>(operand.getDefiningOp()))
		return false;

	return true;
}

//===----------------------------------------------------------------------===//
// Modelica::PackOp
//===----------------------------------------------------------------------===//

mlir::ValueRange PackOpAdaptor::values()
{
	return getValues();
}

llvm::ArrayRef<llvm::StringRef> PackOp::getAttributeNames()
{
	return {};
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

llvm::ArrayRef<llvm::StringRef> ExtractOp::getAttributeNames()
{
	static llvm::StringRef attrNames[] = {llvm::StringRef("index")};
	return llvm::makeArrayRef(attrNames);
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

llvm::ArrayRef<llvm::StringRef> SimulationOp::getAttributeNames()
{
	static llvm::StringRef attrNames[] = {llvm::StringRef("variableNames"), llvm::StringRef("startTime"), llvm::StringRef("endTime"), llvm::StringRef("timeStep")};
	return llvm::makeArrayRef(attrNames);
}

mlir::ArrayAttr SimulationOp::variableNames() {
    return getOperation()->getAttrOfType<mlir::ArrayAttr>("variableNames");
}

void SimulationOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ArrayAttr variableNames, RealAttribute startTime,
						 RealAttribute endTime, RealAttribute timeStep, RealAttribute relTol, RealAttribute absTol, mlir::TypeRange vars)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	state.addAttribute("variableNames", variableNames);
	state.addAttribute("startTime", startTime);
	state.addAttribute("endTime", endTime);
	state.addAttribute("timeStep", timeStep);
	state.addAttribute("relativeTolerance", relTol);
	state.addAttribute("absoluteTolerance", absTol);

	// Init block
	builder.createBlock(state.addRegion());

	// Body block
	builder.createBlock(state.addRegion(), {}, vars);
}

void SimulationOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.simulation ("
					<< "start: " << startTime().getValue()
					<< ", end: " << endTime().getValue()
					<< ", step: " << timeStep().getValue()
					<< ", variables: " << variableNames()
					<< ", relTol: " << relTol().getValue()
					<< ", absTol: " << absTol().getValue() << ")";

	printer << " init";
	printer.printRegion(init(), false);

	printer << " step";
	printer.printRegion(body(), true);
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

RealAttribute SimulationOp::relTol()
{
	return getOperation()->getAttrOfType<RealAttribute>("relativeTolerance");
}

RealAttribute SimulationOp::absTol()
{
	return getOperation()->getAttrOfType<RealAttribute>("absoluteTolerance");
}

mlir::Region& SimulationOp::init()
{
	return getOperation()->getRegion(0);
}

mlir::Region& SimulationOp::body()
{
	return getOperation()->getRegion(1);
}

mlir::Value SimulationOp::getVariableAllocation(mlir::Value var)
{
	unsigned int index = var.dyn_cast<mlir::BlockArgument>().getArgNumber();
	return mlir::cast<YieldOp>(init().back().getTerminator()).values()[index];
}

mlir::Value SimulationOp::time()
{
	return body().getArgument(0);
}

//===----------------------------------------------------------------------===//
// Modelica::EquationOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> EquationOp::getAttributeNames()
{
	return {};
}

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

llvm::ArrayRef<llvm::StringRef> ForEquationOp::getAttributeNames()
{
	return {};
}

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
	return mlir::cast<YieldOp>(inductionsBlock()->getTerminator()).values();
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

llvm::ArrayRef<llvm::StringRef> InductionOp::getAttributeNames()
{
	static llvm::StringRef attrNames[] = {llvm::StringRef("start"), llvm::StringRef("end")};
	return llvm::makeArrayRef(attrNames);
}

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

llvm::ArrayRef<llvm::StringRef> EquationSidesOp::getAttributeNames()
{
	static llvm::StringRef attrNames[] = {llvm::StringRef("lhs"), llvm::StringRef("rhs")};
	return llvm::makeArrayRef(attrNames);
}

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

llvm::ArrayRef<llvm::StringRef> FunctionOp::getAttributeNames()
{
	static llvm::StringRef attrNames[] = {llvm::StringRef("args_names"), llvm::StringRef("results_names")};
	return llvm::makeArrayRef(attrNames);
}

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
	build(builder, state, name, type,
				builder.getStrArrayAttr(argsNames),
				builder.getStrArrayAttr(resultsNames));
}

void FunctionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, llvm::StringRef name, mlir::FunctionType type, mlir::ArrayAttr argsNames, mlir::ArrayAttr resultsNames)
{
	state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
	state.addAttribute(getTypeAttrName(), mlir::TypeAttr::get(type));

	state.addAttribute("args_names", argsNames);
	state.addAttribute("results_names", resultsNames);

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
	if (mlir::function_like_impl::parseFunctionArgumentList(parser, false, false, args, argsTypes, argsAttrs, isVariadic))
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

void FunctionOp::getMembers(llvm::SmallVectorImpl<mlir::Value>& members, llvm::SmallVectorImpl<llvm::StringRef>& names)
{
	for (const auto& [arg, name] : llvm::zip(getArguments(), argsNames()))
	{
		members.push_back(arg);
		names.push_back(name.cast<mlir::StringAttr>().getValue());
	}

	getBody().walk([&members, &names](MemberCreateOp op) {
		members.push_back(op.getResult());
		names.push_back(op.name());
	});
}

//===----------------------------------------------------------------------===//
// Modelica::FunctionTerminatorOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> FunctionTerminatorOp::getAttributeNames()
{
	return {};
}

void FunctionTerminatorOp::build(mlir::OpBuilder& builder, mlir::OperationState& state)
{
}

mlir::ParseResult FunctionTerminatorOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return mlir::success();
}

void FunctionTerminatorOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName();
}

//===----------------------------------------------------------------------===//
// Modelica::DerFunctionOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> DerFunctionOp::getAttributeNames()
{
	static llvm::StringRef attrNames[] = {llvm::StringRef("derived_function"), llvm::StringRef("independent_vars")};
	return llvm::makeArrayRef(attrNames);
}

void DerFunctionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, llvm::StringRef name, llvm::StringRef derivedFunction, llvm::ArrayRef<llvm::StringRef> independentVariables)
{
	state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
	state.addAttribute("derived_function", builder.getStringAttr(derivedFunction));
	state.addAttribute("independent_vars", builder.getStrArrayAttr(independentVariables));
}

void DerFunctionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, llvm::StringRef name, llvm::StringRef derivedFunction, llvm::ArrayRef<mlir::Attribute> independentVariables)
{
	state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
	state.addAttribute("derived_function", builder.getStringAttr(derivedFunction));
	state.addAttribute("independent_vars", builder.getArrayAttr(independentVariables));
}

mlir::ParseResult DerFunctionOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::StringAttr nameAttr;
	if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes))
		return mlir::failure();

	mlir::NamedAttrList attributes;

	if (parser.parseOptionalAttrDict(attributes))
		return mlir::failure();

	result.attributes.append(attributes);
	return mlir::success();
}

void DerFunctionOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " @" << name();
	printer << " {";

	size_t index = 0;

	for (auto attribute : getOperation()->getAttrs())
	{
		if (attribute.first == mlir::SymbolTable::getSymbolAttrName())
			continue;

		if (index++ > 0)
			printer << ", ";

		printer << attribute.first;
		printer << " = ";
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
	return getOperation()->getAttrOfType<mlir::ArrayAttr>("independent_vars").getValue();
}

//===----------------------------------------------------------------------===//
// Modelica::ConstantOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> ConstantOp::getAttributeNames()
{
	static llvm::StringRef attrNames[] = {llvm::StringRef("value")};
	return llvm::makeArrayRef(attrNames);
}

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

mlir::ValueRange ConstantOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	auto derivedOp = builder.create<ConstantOp>(getLoc(), RealAttribute::get(getContext(), 0));
	return derivedOp->getResults();
}

void ConstantOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{

}

void ConstantOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

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

llvm::ArrayRef<llvm::StringRef> CastOp::getAttributeNames()
{
	return {};
}

mlir::Value CastOpAdaptor::value()
{
	return getValues()[0];
}

void CastOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value value, mlir::Type resultType)
{
	state.addOperands(value);
	state.addTypes(resultType);
}

mlir::ParseResult CastOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::OpAsmParser::OperandType value;
	mlir::Type valueType;
	mlir::Type resultType;

	if (parser.parseOperand(value) ||
			parser.parseColon() ||
			parser.parseType(valueType) ||
			parser.resolveOperand(value, valueType, result.operands) ||
			parser.parseArrow() ||
			parser.parseType(resultType))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void CastOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << value()
					<< " : " << value().getType() << " -> " << resultType();
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

mlir::ValueRange CastOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	return derivatives.lookup(value());
}

void CastOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(value());
}

void CastOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

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

llvm::ArrayRef<llvm::StringRef> AssignmentOp::getAttributeNames()
{
	return {};
}

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
	state.addOperands(source);
	state.addOperands(destination);
}

mlir::ParseResult AssignmentOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::OpAsmParser::OperandType source;
	mlir::OpAsmParser::OperandType destination;

	mlir::Type sourceType;
	mlir::Type destinationType;

	if (parser.parseOperand(source) ||
			parser.parseComma() ||
			parser.parseOperand(destination) ||
			parser.parseColon() ||
			parser.parseType(sourceType) ||
			parser.parseComma() ||
			parser.parseType(destinationType) ||
			parser.resolveOperand(source, sourceType, result.operands) ||
			parser.resolveOperand(destination, destinationType, result.operands))
		return mlir::failure();

	return mlir::success();
}

void AssignmentOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName()
					<< " " << source() << ", " << destination()
					<< " : " << source().getType() << ", " << destination().getType();
}

void AssignmentOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (source().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), source(), mlir::SideEffects::DefaultResource::get());

	if (destination().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Write::get(), source(), mlir::SideEffects::DefaultResource::get());
}

mlir::ValueRange AssignmentOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	mlir::Location loc = getLoc();

	mlir::Value derivedSource = derivatives.lookup(source());
	mlir::Value derivedDestination = derivatives.lookup(destination());

	builder.create<AssignmentOp>(loc, derivedSource, derivedDestination);
	return llvm::None;
}

void AssignmentOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{

}

void AssignmentOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

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

llvm::ArrayRef<llvm::StringRef> CallOp::getAttributeNames()
{
	static llvm::StringRef attrNames[] = {llvm::StringRef("callee"), llvm::StringRef("moved_results")};
	return llvm::makeArrayRef(attrNames);
}

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
  /*
	// The callee may have no arguments and no results, but still have side
	// effects (i.e. an external function writing elsewhere). Thus we need to
	// consider the call itself as if it is has side effects and prevent the
	// CSE pass to erase it.
	effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::SideEffects::DefaultResource::get());
   */

	// Declare the side effects on the array arguments.
	unsigned int movedResultsCount = movedResults();
	unsigned int nativeArgsCount = args().size() - movedResultsCount;

	for (size_t i = 0; i < nativeArgsCount; ++i)
    effects.emplace_back(mlir::MemoryEffects::Read::get(), args()[i], mlir::SideEffects::DefaultResource::get());

	// Declare the side effects on the static array results that have been
	// promoted to arguments.

	for (size_t i = 0; i < movedResultsCount; ++i)
		effects.emplace_back(mlir::MemoryEffects::Write::get(), args()[nativeArgsCount + i], mlir::SideEffects::DefaultResource::get());

  for (const auto& result : getResults())
  {
    populateAllocationEffects(effects, result);

    if (result.getType().isa<ArrayType>())
      effects.emplace_back(mlir::MemoryEffects::Write::get(), result, mlir::SideEffects::DefaultResource::get());
  }
}

mlir::CallInterfaceCallable CallOp::getCallableForCallee() {
	return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
}

mlir::Operation::operand_range CallOp::getArgOperands()
{
	return getOperands();
}

mlir::ValueRange CallOp::getArgs()
{
	return args();
}

unsigned int CallOp::getArgExpectedRank(unsigned int argIndex)
{
	auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
	auto function = module.lookupSymbol<FunctionOp>(callee());

	if (function == nullptr)
	{
		// If the function is not declare, then assume that the arguments types
		// already match its hypothetical signature.

		mlir::Type argType = getArgs()[argIndex].getType();

		if (auto arrayType = argType.dyn_cast<ArrayType>())
			return arrayType.getRank();

		return 0;
	}

	mlir::Type argType = function.getArgument(argIndex).getType();

	if (auto arrayType = argType.dyn_cast<ArrayType>())
		return arrayType.getRank();

	return 0;
}

mlir::ValueRange CallOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
{
	llvm::SmallVector<mlir::Type, 3> newResultsTypes;

	for (mlir::Type type : getResultTypes())
	{
		mlir::Type newResultType = type.cast<ArrayType>().slice(indexes.size());

		if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
			newResultType = arrayType.getElementType();

		newResultsTypes.push_back(newResultType);
	}

	llvm::SmallVector<mlir::Value, 3> newArgs;

	for (mlir::Value arg : args())
	{
		assert(arg.getType().isa<ArrayType>());
		mlir::Value newArg = builder.create<SubscriptionOp>(getLoc(), arg, indexes);

		if (auto arrayType = newArg.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
			newArg = builder.create<LoadOp>(getLoc(), newArg);

		newArgs.push_back(newArg);
	}

	auto op = builder.create<CallOp>(getLoc(), callee(), newResultsTypes, newArgs);
	return op->getResults();
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

mlir::ValueRange CallOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	assert(args().size() == 1 && resultTypes().size() == 1 &&
		"CallOp differentiation with multiple arguments or multiple return values is not supported yet");

	std::string pderName = getPartialDerFunctionName(callee());
	CallOp pderCall = builder.create<CallOp>(getLoc(), pderName, resultTypes(), args(), movedResults());

	MulOp mulOp = builder.create<MulOp>(getLoc(), resultTypes()[0], derivatives.lookup(args()[0]), pderCall.getResult(0));
	return mulOp->getResults();
}

void CallOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	std::string pderName = getPartialDerFunctionName(callee());
	mlir::ModuleOp moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();

	// Create the partial derivative function if it does not exist already.
	if (moduleOp.lookupSymbol<FunctionOp>(pderName) == nullptr &&
			moduleOp.lookupSymbol<DerFunctionOp>(pderName) == nullptr)
	{
		FunctionOp base = moduleOp.lookupSymbol<FunctionOp>(callee());
		assert(base != nullptr);
		assert(base.argsNames().size() == 1 && base.resultsNames().size() == 1 &&
			"CallOp differentiation with multiple arguments or multiple return values is not supported yet");

		mlir::OpBuilder builder(moduleOp.getContext());
		builder.setInsertionPointAfter(base);
		mlir::Attribute independentVariable = base.argsNames()[0];
		builder.create<DerFunctionOp>(base.getLoc(), pderName, base.getName(), independentVariable);
	}

	for (mlir::Value arg : args())
		toBeDerived.push_back(arg);
}

void CallOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

mlir::StringRef CallOp::callee()
{
	return getOperation()->getAttrOfType<mlir::FlatSymbolRefAttr>("callee").getValue();
}

mlir::TypeRange CallOp::resultTypes()
{
	return getOperation()->getResultTypes();
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

llvm::ArrayRef<llvm::StringRef> MemberCreateOp::getAttributeNames()
{
	static llvm::StringRef attrNames[] = {llvm::StringRef("name")};
	return llvm::makeArrayRef(attrNames);
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
	llvm::SmallVector<mlir::Type, 2> dynamicDimensionsTypes;
	mlir::Type resultType;

	llvm::SMLoc dynamicDimensionsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(dynamicDimensions))
		return mlir::failure();

	mlir::NamedAttrList attributes;

	if (parser.parseOptionalAttrDict(attributes))
		return mlir::failure();

	result.attributes.append(attributes);

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
			printer << ",";

		printer << " " << dynamicDimension.value();
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

llvm::ArrayRef<llvm::StringRef> MemberLoadOp::getAttributeNames()
{
	return {};
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

	if (parser.parseType(resultType))
		return mlir::failure();

	auto memberType = resultType.isa<ArrayType>() ?
										MemberType::get(resultType.cast<ArrayType>()) :
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

mlir::ValueRange MemberLoadOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	auto derivedOp = builder.create<MemberLoadOp>(getLoc(), convertToRealType(resultType()), derivatives.lookup(member()));
	return derivedOp->getResults();
}

void MemberLoadOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(member());
}

void MemberLoadOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

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

llvm::ArrayRef<llvm::StringRef> MemberStoreOp::getAttributeNames()
{
	return {};
}

void MemberStoreOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value member, mlir::Value value)
{
	state.addOperands(member);
	state.addOperands(value);
}

mlir::ParseResult MemberStoreOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 2> operands;
	mlir::Type valueType;
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

	if (auto castedMemberType = memberType.cast<MemberType>(); castedMemberType.getRank() != 0)
		valueType = castedMemberType.toArrayType();
	else
		valueType = castedMemberType.getElementType();

	if (parser.resolveOperand(operands[0], memberType, result.operands) ||
			parser.resolveOperand(operands[1], valueType, result.operands))
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

	if (valueType.isa<ArrayType>())
	{
		auto arrayType = valueType.cast<ArrayType>();

		for (const auto& [valueDimension, memberDimension] : llvm::zip(arrayType.getShape(), memberType.getShape()))
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

mlir::ValueRange MemberStoreOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	// Store operations should be derived only if they store a value into
	// a member whose derivative is created by the current function. Otherwise,
	// we would create a double store into that derived member.

	assert(derivatives.contains(member()) && "Derived member not found");
	mlir::Value derivedMember = derivatives.lookup(member());

	if (!derivatives.contains(derivedMember))
		builder.create<MemberStoreOp>(getLoc(), derivedMember, derivatives.lookup(value()));

	return llvm::None;
}

void MemberStoreOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(value());
	toBeDerived.push_back(member());
}

void MemberStoreOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

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

llvm::ArrayRef<llvm::StringRef> AllocaOp::getAttributeNames()
{
	static llvm::StringRef attrNames[] = {llvm::StringRef("constant")};
	return llvm::makeArrayRef(attrNames);
}

void AllocaOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type elementType, llvm::ArrayRef<long> shape, mlir::ValueRange dimensions, bool constant)
{
	state.addTypes(ArrayType::get(state.getContext(), BufferAllocationScope::stack, elementType, shape));
	state.addOperands(dimensions);
	state.addAttribute("constant", builder.getBoolAttr(constant));
}

mlir::ParseResult AllocaOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	auto& builder = parser.getBuilder();

	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> indexes;
	llvm::SmallVector<mlir::Type, 3> indexesTypes;

	mlir::Type resultType;

	if (parser.parseOperandList(indexes))
		return mlir::failure();

	mlir::NamedAttrList attributes;

	if (parser.parseOptionalAttrDict(attributes))
		return mlir::failure();

	result.attributes.append(attributes);

	if (!result.attributes.getNamed("constant"))
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

	if (parser.parseType(resultType))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void AllocaOp::print(mlir::OpAsmPrinter& printer)
{
	auto dimensions = dynamicDimensions();
	printer << getOperationName();

	if (!dimensions.empty())
		printer << " " << dimensions;

	printer.printOptionalAttrDict(getOperation()->getAttrs());
	printer << " : ";

	if (dimensions.size() > 1)
		printer << "(";

	printer << dimensions.getTypes();

	if (dimensions.size() > 1)
		printer << ")";

	if (!dimensions.empty())
		printer << " -> ";

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
  populateAllocationEffects(effects, getResult());
}

ArrayType AllocaOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<ArrayType>();
}

mlir::ValueRange AllocaOp::dynamicDimensions()
{
	return Adaptor(*this).dynamicDimensions();
}

bool AllocaOp::isConstant()
{
	return getOperation()->getAttrOfType<mlir::BoolAttr>("constant").getValue();
}

mlir::ValueRange AllocaOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	return builder.clone(*getOperation())->getResults();
}

void AllocaOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{

}

void AllocaOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

//===----------------------------------------------------------------------===//
// Modelica::AllocOp
//===----------------------------------------------------------------------===//

mlir::ValueRange AllocOpAdaptor::dynamicDimensions()
{
	return getValues();
}

bool AllocOpAdaptor::isConstant()
{
	auto attr = getAttrs().getNamed("constant");

	if (!attr.hasValue())
		return false;

	return attr->second.cast<mlir::BoolAttr>().getValue();
}

llvm::ArrayRef<llvm::StringRef> AllocOp::getAttributeNames()
{
	static llvm::StringRef attrNames[] = {llvm::StringRef("constant")};
	return llvm::makeArrayRef(attrNames);
}

void AllocOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type elementType, llvm::ArrayRef<long> shape, mlir::ValueRange dimensions, bool shouldBeFreed, bool constant)
{
	state.addTypes(ArrayType::get(state.getContext(), BufferAllocationScope::heap, elementType, shape));
	state.addOperands(dimensions);

	state.addAttribute(getAutoFreeAttrName(), builder.getBoolAttr(shouldBeFreed));
	state.addAttribute("constant", builder.getBoolAttr(constant));
}

mlir::ParseResult AllocOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> dimensions;
	llvm::SmallVector<mlir::Type, 3> dimensionsTypes;
	mlir::Type resultType;

	llvm::SMLoc dimensionsLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(dimensions))
		return mlir::failure();

	mlir::NamedAttrList attributes;

	if (parser.parseOptionalAttrDict(attributes))
		return mlir::failure();

	result.attributes.append(attributes);

	if (parser.parseColon())
		return mlir::failure();

	if (dimensions.size() > 1)
		if (parser.parseLParen())
			return mlir::failure();

	if (parser.parseTypeList(dimensionsTypes))
		return mlir::failure();

	if (dimensions.size() > 1)
		if (parser.parseRParen())
			return mlir::failure();

	if (parser.resolveOperands(dimensions, dimensionsTypes, dimensionsLoc, result.operands))
		return mlir::failure();

	if (!dimensions.empty())
		if (parser.parseArrow())
			return mlir::failure();

	if (parser.parseType(resultType))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void AllocOp::print(mlir::OpAsmPrinter& printer)
{
	auto dimensions = dynamicDimensions();
	printer << getOperationName();

	if (!dimensions.empty())
		printer << " " << dimensions;

	printer.printOptionalAttrDict(getOperation()->getAttrs());
	printer << " : ";

	if (dimensions.size() > 1)
		printer << "(";

	printer << dimensions.getTypes();

	if (dimensions.size() > 1)
		printer << ")";

	if (!dimensions.empty())
		printer << " -> ";

	printer << resultType();
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
  populateAllocationEffects(effects, getResult(), !shouldBeFreed());
}

ArrayType AllocOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<ArrayType>();
}

mlir::ValueRange AllocOp::dynamicDimensions()
{
	return Adaptor(*this).dynamicDimensions();
}

bool AllocOp::isConstant()
{
	return Adaptor(*this).isConstant();
}

mlir::ValueRange AllocOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	return builder.clone(*getOperation())->getResults();
}

void AllocOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{

}

void AllocOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

//===----------------------------------------------------------------------===//
// Modelica::FreeOp
//===----------------------------------------------------------------------===//

mlir::Value FreeOpAdaptor::memory()
{
	return getValues()[0];
}

llvm::ArrayRef<llvm::StringRef> FreeOp::getAttributeNames()
{
	return {};
}

void FreeOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value memory)
{
	state.addOperands(memory);
}

mlir::ParseResult FreeOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::OpAsmParser::OperandType memory;
	mlir::Type type;

	if (parser.parseOperand(memory) ||
			parser.parseColonType(type) ||
			parser.resolveOperand(memory, type, result.operands))
		return mlir::failure();

	return mlir::success();
}

void FreeOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName()
					<< " " << memory()
					<< " : " << memory().getType();
}

mlir::LogicalResult FreeOp::verify()
{
	if (auto arrayType = memory().getType().dyn_cast<ArrayType>(); arrayType)
	{
		if (arrayType.getAllocationScope() == BufferAllocationScope::heap)
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

mlir::ValueRange FreeOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	builder.create<FreeOp>(getLoc(), derivatives.lookup(memory()));
	return llvm::None;
}

void FreeOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(memory());
}

void FreeOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

//===----------------------------------------------------------------------===//
// Modelica::ArrayCastOp
//===----------------------------------------------------------------------===//

mlir::Value ArrayCastOpAdaptor::memory()
{
	return getValues()[0];
}

llvm::ArrayRef<llvm::StringRef> ArrayCastOp::getAttributeNames()
{
	return {};
}

void ArrayCastOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value memory, mlir::Type resultType)
{
	state.addOperands(memory);
	state.addTypes(resultType);
}

void ArrayCastOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << memory() << " : " << resultType();
}

mlir::LogicalResult ArrayCastOp::verify()
{
	mlir::Type sourceType = memory().getType();
	mlir::Type destinationType = resultType();

	if (auto source = sourceType.dyn_cast<ArrayType>())
	{
		if (auto destination = destinationType.dyn_cast<ArrayType>())
		{
			if (source.getElementType() != resultType().cast<ArrayType>().getElementType())
				return emitOpError("requires the result pointer type to have the same element type of the operand");

			if (source.getRank() != resultType().cast<ArrayType>().getRank())
				return emitOpError("requires the result pointer type to have the same rank as the operand");

			if (destination.getAllocationScope() != BufferAllocationScope::unknown &&
					source.getAllocationScope() != BufferAllocationScope::unknown &&
					destination.getAllocationScope() != source.getAllocationScope())
				return emitOpError("can change the allocation scope only to the unknown one");

			for (auto [source, destination] : llvm::zip(source.getShape(), destination.getShape()))
				if (destination != -1 && source != destination)
					return emitOpError("can change the dimensions size only to an unknown one");

			return mlir::success();
		}

		if (auto destination = destinationType.dyn_cast<UnsizedArrayType>())
		{
			if (source.getElementType() != destination.getElementType())
				return emitOpError("requires the result pointer type to have the same element type of the operand");

			return mlir::success();
		}
	}

	return emitOpError("requires a compatible conversion");
}

mlir::Value ArrayCastOp::getViewSource()
{
	return memory();
}

mlir::Value ArrayCastOp::memory()
{
	return Adaptor(*this).memory();
}

mlir::Type ArrayCastOp::resultType()
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

llvm::ArrayRef<llvm::StringRef> DimOp::getAttributeNames()
{
	return {};
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
	if (!memory().getType().isa<ArrayType>())
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
		auto arrayType = memory().getType().cast<ArrayType>();
		auto shape = arrayType.getShape();

		size_t index = attribute.getInt();

		if (shape[index] != -1)
			return mlir::IntegerAttr::get(mlir::IndexType::get(getContext()), shape[index]);
	}

	return nullptr;
}

ArrayType DimOp::getArrayType()
{
	return memory().getType().cast<ArrayType>();
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

llvm::ArrayRef<llvm::StringRef> SubscriptionOp::getAttributeNames()
{
	return {};
}

void SubscriptionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value source, mlir::ValueRange indexes)
{
	state.addOperands(source);
	state.addOperands(indexes);

	auto sourceArrayType = source.getType().cast<ArrayType>();
	mlir::Type resultType = sourceArrayType.slice(indexes.size());
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

	if (!sourceType.isa<ArrayType>())
		return parser.emitError(sourceLoc, "the source must have a pointer type");

	result.addTypes(sourceType.cast<ArrayType>().slice(indexes.size()));

	if (!indexes.empty())
	{
		if (parser.parseComma() ||
				parser.parseTypeList(indexesTypes))
			return mlir::failure();
	}

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

mlir::ValueRange SubscriptionOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	auto derivedOp = builder.create<SubscriptionOp>(getLoc(), derivatives.lookup(source()), indexes());
	return derivedOp->getResults();
}

void SubscriptionOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(source());
}

void SubscriptionOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

ArrayType SubscriptionOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<ArrayType>();
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

llvm::ArrayRef<llvm::StringRef> LoadOp::getAttributeNames()
{
	return {};
}

void LoadOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value memory, mlir::ValueRange indexes)
{
	state.addOperands(memory);
	state.addOperands(indexes);
	state.addTypes(memory.getType().cast<ArrayType>().getElementType());
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

	if (!arrayType.isa<ArrayType>())
		return parser.emitError(arrayTypeLoc, "the array type must be a pointer type");

	result.addTypes(arrayType.cast<ArrayType>().getElementType());
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
	auto arrayType = memory().getType().cast<ArrayType>();

	if (arrayType.getRank() != indexes().size())
		return emitOpError("requires the indexes amount (" +
											 std::to_string(indexes().size()) +
											 ") to match the array rank (" +
											 std::to_string(arrayType.getRank()) + ")");

	return mlir::success();
}

void LoadOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Read::get(), memory(), mlir::SideEffects::DefaultResource::get());
}

mlir::ValueRange LoadOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	auto derivedOp = builder.create<LoadOp>(getLoc(), derivatives.lookup(memory()), indexes());
	return derivedOp->getResults();
}

void LoadOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(memory());
}

void LoadOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

ArrayType LoadOp::getArrayType()
{
	return memory().getType().cast<ArrayType>();
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

mlir::Value StoreOpAdaptor::memory()
{
	return getValues()[1];
}

mlir::ValueRange StoreOpAdaptor::indexes()
{
	return mlir::ValueRange(std::next(getValues().begin(), 2), getValues().end());
}

mlir::Value StoreOpAdaptor::value()
{
	return getValues()[0];
}

llvm::ArrayRef<llvm::StringRef> StoreOp::getAttributeNames()
{
	return {};
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

	mlir::OpAsmParser::OperandType array;
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 2> indexes;
	mlir::OpAsmParser::OperandType value;
	mlir::Type arrayType;

	if (parser.parseOperand(array))
		return mlir::failure();

	if (parser.parseOperandList(indexes, mlir::OpAsmParser::Delimiter::Square) ||
			parser.parseComma() ||
			parser.parseOperand(value) ||
			parser.parseColon())
		return mlir::failure();

	llvm::SMLoc arrayTypeLoc = parser.getCurrentLocation();

	if (parser.parseType(arrayType))
		return mlir::failure();

	if (!arrayType.isa<ArrayType>())
		return parser.emitError(arrayTypeLoc)
				<< "destination type must be a pointer type";

	if (parser.resolveOperand(value, arrayType.cast<ArrayType>().getElementType(), result.operands) ||
			parser.resolveOperand(array, arrayType, result.operands) ||
			parser.resolveOperands(indexes, builder.getIndexType(), result.operands))
		return mlir::failure();

	return mlir::success();
}

void StoreOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName()
					<< " " << memory() << "[" << indexes() << "], " << value()
					<< " : " << memory().getType();
}

mlir::LogicalResult StoreOp::verify()
{
	auto arrayType = memory().getType().cast<ArrayType>();

	if (arrayType.getElementType() != value().getType())
		return emitOpError("requires the value to have the same type of the array elements");

	if (arrayType.getRank() != indexes().size())
		return emitOpError("requires the indexes amount (" +
											 std::to_string(indexes().size()) +
											 ") to match the array rank (" +
											 std::to_string(arrayType.getRank()) + ")");

	return mlir::success();
}

void StoreOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Write::get(), memory(), mlir::SideEffects::DefaultResource::get());
}

mlir::ValueRange StoreOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	auto derivedOp = builder.create<StoreOp>(
			getLoc(), derivatives.lookup(value()), derivatives.lookup(memory()), indexes());

	return derivedOp->getResults();
}

void StoreOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(value());
	toBeDerived.push_back(memory());
}

void StoreOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

ArrayType StoreOp::getArrayType()
{
	return memory().getType().cast<ArrayType>();
}

mlir::Value StoreOp::memory()
{
	return Adaptor(*this).memory();
}

mlir::ValueRange StoreOp::indexes()
{
	return Adaptor(*this).indexes();
}

mlir::Value StoreOp::value()
{
	return Adaptor(*this).value();
}

//===----------------------------------------------------------------------===//
// Modelica::ArrayCloneOp
//===----------------------------------------------------------------------===//

mlir::Value ArrayCloneOpAdaptor::source()
{
	return getValues()[0];
}

bool ArrayCloneOpAdaptor::canSourceBeForwarded()
{
	auto attr = getAttrs().getNamed("forward");

	if (attr.hasValue())
		return attr->second.cast<mlir::BoolAttr>().getValue();

	return false;
}

llvm::ArrayRef<llvm::StringRef> ArrayCloneOp::getAttributeNames()
{
	static llvm::StringRef attrNames[] = {llvm::StringRef("forward")};
	return llvm::makeArrayRef(attrNames);
}

void ArrayCloneOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value source, ArrayType resultType, bool shouldBeFreed, bool canSourceBeForwarded)
{
	state.addOperands(source);
	state.addTypes(resultType);
	state.addAttribute(getAutoFreeAttrName(), builder.getBoolAttr(shouldBeFreed));
	state.addAttribute("forward", builder.getBoolAttr(canSourceBeForwarded));
}

mlir::ParseResult ArrayCloneOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::OpAsmParser::OperandType source;
	mlir::Type sourceType;
	mlir::Type resultType;

	if (parser.parseOperand(source))
		return mlir::failure();

	mlir::NamedAttrList attributes;

	if (parser.parseOptionalAttrDict(attributes))
		return mlir::failure();

	result.attributes.append(attributes);

	if (parser.parseColon() ||
			parser.parseType(sourceType) ||
			parser.resolveOperand(source, sourceType, result.operands) ||
			parser.parseArrow() ||
			parser.parseType(resultType))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void ArrayCloneOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << source();
	auto attributes = getOperation()->getAttrDictionary();

	if (!attributes.empty())
		printer << " " << attributes;

	printer << " : " << source().getType() << " -> " << resultType();
}

mlir::LogicalResult ArrayCloneOp::verify()
{
	auto arrayType = resultType();

	if (auto scope = arrayType.getAllocationScope();
			scope != BufferAllocationScope::stack && scope != BufferAllocationScope::heap)
		return emitOpError("requires the result array type to be stack or heap allocated");

	return mlir::success();
}

void ArrayCloneOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Read::get(), source(), mlir::SideEffects::DefaultResource::get());
  populateAllocationEffects(effects, getResult(), !shouldBeFreed());
	effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
}

ArrayType ArrayCloneOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<ArrayType>();
}

mlir::Value ArrayCloneOp::source()
{
	return Adaptor(*this).source();
}

bool ArrayCloneOp::canSourceBeForwarded()
{
	return Adaptor(*this).canSourceBeForwarded();
}

//===----------------------------------------------------------------------===//
// Modelica::IfOp
//===----------------------------------------------------------------------===//

mlir::Value IfOpAdaptor::condition()
{
	return getValues()[0];
}

llvm::ArrayRef<llvm::StringRef> IfOp::getAttributeNames()
{
	return {};
}

void IfOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value condition, bool withElseRegion)
{
	mlir::OpBuilder::InsertionGuard guard(builder);
	state.addOperands(condition);

	// "Then" region
	auto* thenRegion = state.addRegion();
	builder.createBlock(thenRegion);

	// "Else" region
	auto* elseRegion = state.addRegion();

	if (withElseRegion)
		builder.createBlock(elseRegion);
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
		condition = condAttr.getValue();
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

mlir::ValueRange IfOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	return llvm::None;
}

void IfOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{

}

void IfOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{
	regions.push_back(&thenRegion());
	regions.push_back(&elseRegion());
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

llvm::ArrayRef<llvm::StringRef> ForOp::getAttributeNames()
{
	return {};
}

void ForOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange args)
{
	mlir::OpBuilder::InsertionGuard guard(builder);
	state.addOperands(args);

	// Condition block
	builder.createBlock(state.addRegion(), {}, args.getTypes());

	// Body block
	builder.createBlock(state.addRegion(), {}, args.getTypes());

	// Step block
	builder.createBlock(state.addRegion(), {}, args.getTypes());
}

mlir::ParseResult ForOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::Region* conditionRegion = result.addRegion();

	if (mlir::succeeded(parser.parseOptionalLParen()))
	{
		if (mlir::failed(parser.parseOptionalRParen()))
		{
			do {
				mlir::OpAsmParser::OperandType arg;
				mlir::Type argType;

				if (parser.parseOperand(arg) ||
						parser.parseColonType(argType) ||
						parser.resolveOperand(arg, argType, result.operands))
					return mlir::failure();
			} while (mlir::succeeded(parser.parseOptionalComma()));
		}

		if (parser.parseRParen())
			return mlir::failure();
	}

	if (parser.parseKeyword("condition"))
		return mlir::failure();

	if (parser.parseRegion(*conditionRegion))
		return mlir::failure();

	if (parser.parseKeyword("body"))
		return mlir::failure();

	mlir::Region* bodyRegion = result.addRegion();

	if (parser.parseRegion(*bodyRegion))
		return mlir::failure();

	if (parser.parseKeyword("step"))
		return mlir::failure();

	mlir::Region* stepRegion = result.addRegion();

	if (parser.parseRegion(*stepRegion))
		return mlir::failure();

	return mlir::success();
}

void ForOp::print(mlir::OpAsmPrinter& printer)
{
	auto values = args();
	printer << "modelica.for";

	if (!values.empty())
	{
		printer << " (";

		for (auto arg : llvm::enumerate(values))
		{
			if (arg.index() != 0)
				printer << ", ";

			printer << arg.value() << " : " << arg.value().getType();
		}

		printer << ")";
	}

	printer << " condition";
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

mlir::ValueRange ForOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	return llvm::None;
}

void ForOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{

}

void ForOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{
	regions.push_back(&body());
}

mlir::Region& ForOp::condition()
{
	return getOperation()->getRegion(0);
}

mlir::Region& ForOp::body()
{
	return getOperation()->getRegion(1);
}

mlir::Region& ForOp::step()
{
	return getOperation()->getRegion(2);
}

mlir::ValueRange ForOp::args()
{
	return Adaptor(*this).args();
}

//===----------------------------------------------------------------------===//
// Modelica::WhileOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> WhileOp::getAttributeNames()
{
	return {};
}

void WhileOp::build(mlir::OpBuilder& builder, mlir::OperationState& state)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	// Condition block
	builder.createBlock(state.addRegion());

	// Body block
	builder.createBlock(state.addRegion());
}

mlir::ParseResult WhileOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
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

void WhileOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName();
	printer.printRegion(condition(), false);
	printer << " do";
	printer.printRegion(body(), false);
}

mlir::LogicalResult WhileOp::verify()
{
	return mlir::success();
}

bool WhileOp::isDefinedOutsideOfLoop(mlir::Value value)
{
	return !body().isAncestor(value.getParentRegion());
}

mlir::Region& WhileOp::getLoopBody()
{
	return body();
}

mlir::LogicalResult WhileOp::moveOutOfLoop(llvm::ArrayRef<mlir::Operation*> ops)
{
	for (auto* op : ops)
		op->moveBefore(*this);

	return mlir::success();
}

void WhileOp::getSuccessorRegions(llvm::Optional<unsigned int> index, llvm::ArrayRef<mlir::Attribute> operands, llvm::SmallVectorImpl<mlir::RegionSuccessor>& regions)
{
	if (!index.hasValue())
	{
		regions.emplace_back(&condition(), condition().getArguments());
		return;
	}

	assert(*index < 2 && "There are only two regions in a WhileOp");

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

mlir::ValueRange WhileOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	return llvm::None;
}

void WhileOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{

}

void WhileOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{
	regions.push_back(&body());
}

mlir::Region& WhileOp::condition()
{
	return getOperation()->getRegion(0);
}

mlir::Region& WhileOp::body()
{
	return getOperation()->getRegion(1);
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

llvm::ArrayRef<llvm::StringRef> ConditionOp::getAttributeNames()
{
	return {};
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

llvm::ArrayRef<llvm::StringRef> YieldOp::getAttributeNames()
{
	return {};
}

void YieldOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange operands)
{
	state.addOperands(operands);
}

mlir::ParseResult YieldOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	llvm::SmallVector<mlir::OpAsmParser::OperandType, 3> values;
	llvm::SmallVector<mlir::Type, 3> valuesTypes;

	llvm::SMLoc valuesLoc = parser.getCurrentLocation();

	if (parser.parseOperandList(values))
		return mlir::failure();

	if (!values.empty())
		if (parser.parseOptionalColonTypeList(valuesTypes))
			return mlir::failure();

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
// Modelica::BreakOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> BreakOp::getAttributeNames()
{
	return {};
}

void BreakOp::build(mlir::OpBuilder& builder, mlir::OperationState& state)
{
}

mlir::ParseResult BreakOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return mlir::success();
}

void BreakOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName();
}

//===----------------------------------------------------------------------===//
// Modelica::ReturnOp
//===----------------------------------------------------------------------===//

llvm::ArrayRef<llvm::StringRef> ReturnOp::getAttributeNames()
{
	return {};
}

void ReturnOp::build(mlir::OpBuilder& builder, mlir::OperationState& state)
{
}

mlir::ParseResult ReturnOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	return mlir::success();
}

void ReturnOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName();
}

//===----------------------------------------------------------------------===//
// Modelica::NotOp
//===----------------------------------------------------------------------===//

mlir::Value NotOpAdaptor::operand()
{
	return getValues()[0];
}

llvm::ArrayRef<llvm::StringRef> NotOp::getAttributeNames()
{
	return {};
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
		if (auto arrayType = operand().getType().dyn_cast<ArrayType>(); !arrayType || !arrayType.getElementType().isa<BooleanType>())
			return emitOpError("requires the operand to be a boolean or an array of booleans");

	return mlir::success();
}

void NotOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (operand().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());

  populateAllocationEffects(effects, getResult());

	if (resultType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
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

llvm::ArrayRef<llvm::StringRef> AndOp::getAttributeNames()
{
	return {};
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

	if (lhsType.isa<ArrayType>() && rhsType.isa<ArrayType>())
		if (lhsType.cast<ArrayType>().getElementType().isa<BooleanType>() &&
		    rhsType.cast<ArrayType>().getElementType().isa<BooleanType>())
			return mlir::success();

	return emitOpError("requires the operands to be booleans or arrays of booleans");
}

void AndOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (lhs().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());

	if (rhs().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());

  populateAllocationEffects(effects, getResult());

	if (resultType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
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

llvm::ArrayRef<llvm::StringRef> OrOp::getAttributeNames()
{
	return {};
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

	if (lhsType.isa<ArrayType>() && rhsType.isa<ArrayType>())
		if (lhsType.cast<ArrayType>().getElementType().isa<BooleanType>() &&
				rhsType.cast<ArrayType>().getElementType().isa<BooleanType>())
			return mlir::success();

	return emitOpError("requires the operands to be booleans or arrays of booleans");
}

void OrOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (lhs().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());

	if (rhs().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());

  populateAllocationEffects(effects, getResult());

	if (resultType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
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

llvm::ArrayRef<llvm::StringRef> EqOp::getAttributeNames()
{
	return {};
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

llvm::ArrayRef<llvm::StringRef> NotEqOp::getAttributeNames()
{
	return {};
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

llvm::ArrayRef<llvm::StringRef> GtOp::getAttributeNames()
{
	return {};
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

llvm::ArrayRef<llvm::StringRef> GteOp::getAttributeNames()
{
	return {};
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

llvm::ArrayRef<llvm::StringRef> LtOp::getAttributeNames()
{
	return {};
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

llvm::ArrayRef<llvm::StringRef> LteOp::getAttributeNames()
{
	return {};
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

llvm::ArrayRef<llvm::StringRef> NegateOp::getAttributeNames()
{
	return {};
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
	if (operand().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());

  populateAllocationEffects(effects, getResult());

	if (resultType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
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
	erase();

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
	return builder.clone(*operand().getDefiningOp())->getResult(0);
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
		if (auto casted = mlir::dyn_cast<DivOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeDivOp(builder, resultType, value);

		return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value operand = distributeFn(this->operand());

	return builder.create<NegateOp>(getLoc(), resultType, operand);
}

mlir::ValueRange NegateOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	mlir::Value derivedOperand = derivatives.lookup(operand());
	auto derivedOp = builder.create<NegateOp>(getLoc(), convertToRealType(resultType()), derivedOperand);
	return derivedOp->getResults();
}

void NegateOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(operand());
}

void NegateOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

void NegateOp::foldConstants(mlir::OpBuilder& builder)
{
	if (!isOperandFoldable(operand()))
		return;

	ConstantOp operandOp = mlir::cast<ConstantOp>(operand().getDefiningOp());

	double operand = getAttributeValue(operandOp.value());

	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	mlir::Value newOp = builder.create<ConstantOp>(getLoc(), RealAttribute::get(getContext(), -operand));

	replaceAllUsesWith(newOp);
	erase();
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

llvm::ArrayRef<llvm::StringRef> AddOp::getAttributeNames()
{
	return {};
}

void AddOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	if (auto arrayType = resultType.dyn_cast<ArrayType>())
		if (arrayType.getAllocationScope() == BufferAllocationScope::unknown)
			resultType = arrayType.toMinAllowedAllocationScope();

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
	if (lhs().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());

	if (rhs().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());

  populateAllocationEffects(effects, getResult());

	if (resultType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
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
		mlir::Type subType = getMostGenericType(rhs().getType(), nestedOperand.getType());
		auto right = builder.create<SubOp>(getLoc(), subType, nestedOperand, rhs());

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
		mlir::Type subType = getMostGenericType(lhs().getType(), nestedOperand.getType());
		auto right = builder.create<SubOp>(getLoc(), subType, nestedOperand, lhs());

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
		if (auto casted = mlir::dyn_cast<DivOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeDivOp(builder, resultType, value);

		return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = distributeFn(this->rhs());

	return builder.create<AddOp>(getLoc(), resultType, lhs, rhs);
}

mlir::ValueRange AddOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	mlir::Location loc = getLoc();

	mlir::Value derivedLhs = derivatives.lookup(lhs());
	mlir::Value derivedRhs = derivatives.lookup(rhs());

	auto derivedOp = builder.create<AddOp>(
			loc, convertToRealType(resultType()), derivedLhs, derivedRhs);

	return derivedOp->getResults();
}

void AddOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(lhs());
	toBeDerived.push_back(rhs());
}

void AddOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

void AddOp::foldConstants(mlir::OpBuilder& builder)
{
	if (isOperandFoldable(lhs()) && getAttributeValue(mlir::cast<ConstantOp>(lhs().getDefiningOp()).value()) == 0.0)
	{
		replaceAllUsesWith(rhs());
		erase();
		return;
	}

	if (isOperandFoldable(rhs()) && getAttributeValue(mlir::cast<ConstantOp>(rhs().getDefiningOp()).value()) == 0.0)
	{
		replaceAllUsesWith(lhs());
		erase();
		return;
	}

	if (!isOperandFoldable(lhs()) || !isOperandFoldable(rhs()))
		return;

	// Note: this constant folding is done also on Subscription indexes.
	ConstantOp leftOp = mlir::cast<ConstantOp>(lhs().getDefiningOp());
	ConstantOp rightOp = mlir::cast<ConstantOp>(rhs().getDefiningOp());

	double left = getAttributeValue(leftOp.value());
	double right = getAttributeValue(rightOp.value());

	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	mlir::Type type = getMostGenericType(leftOp.resultType(), rightOp.resultType());
	mlir::Value newOp = builder.create<ConstantOp>(getLoc(), getAttribute(builder, type, left + right));

	replaceAllUsesWith(newOp);
	erase();
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
// Modelica::AddElementWiseOp
//===----------------------------------------------------------------------===//

mlir::Value AddElementWiseOpAdaptor::lhs()
{
	return getValues()[0];
}

mlir::Value AddElementWiseOpAdaptor::rhs()
{
	return getValues()[1];
}

llvm::ArrayRef<llvm::StringRef> AddElementWiseOp::getAttributeNames()
{
	return {};
}

void AddElementWiseOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	if (auto arrayType = resultType.dyn_cast<ArrayType>())
		if (arrayType.getAllocationScope() == BufferAllocationScope::unknown)
			resultType = arrayType.toMinAllowedAllocationScope();

	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::ParseResult AddElementWiseOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void AddElementWiseOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << lhs() << ", " << rhs() << " : ("
					<< lhs().getType() << ", " << rhs().getType()
					<< ") -> " << resultType();
}

void AddElementWiseOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (lhs().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());

	if (rhs().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());

  populateAllocationEffects(effects, getResult());

	if (resultType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
}

mlir::LogicalResult AddElementWiseOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	if (auto size = currentResult.size(); size != 1)
		return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");

	mlir::Value toNest = currentResult[0];

	if (argumentIndex == 0)
	{
		mlir::Value nestedOperand = readValue(builder, toNest);
		mlir::Type subType = getMostGenericType(rhs().getType(), nestedOperand.getType());
		auto right = builder.create<SubElementWiseOp>(getLoc(), subType, nestedOperand, rhs());

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
		mlir::Type subType = getMostGenericType(lhs().getType(), nestedOperand.getType());
		auto right = builder.create<SubElementWiseOp>(getLoc(), subType, nestedOperand, lhs());

		for (auto& use : toNest.getUses())
			if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
				use.set(right.getResult());

		replaceAllUsesWith(rhs());
		erase();

		return mlir::success();
	}

	return emitError("Index out of bounds: " + std::to_string(argumentIndex));
}

mlir::Value AddElementWiseOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeNegateOp(builder, resultType);

		return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = distributeFn(this->rhs());

	return builder.create<AddElementWiseOp>(getLoc(), resultType, lhs, rhs);
}

mlir::Value AddElementWiseOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeMulOp(builder, resultType, value);

		return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = distributeFn(this->rhs());

	return builder.create<AddElementWiseOp>(getLoc(), resultType, lhs, rhs);
}

mlir::Value AddElementWiseOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<DivOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeDivOp(builder, resultType, value);

		return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = distributeFn(this->rhs());

	return builder.create<AddElementWiseOp>(getLoc(), resultType, lhs, rhs);
}

mlir::ValueRange AddElementWiseOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	mlir::Location loc = getLoc();

	mlir::Value derivedLhs = derivatives.lookup(lhs());
	mlir::Value derivedRhs = derivatives.lookup(rhs());

	auto derivedOp = builder.create<AddElementWiseOp>(
			loc, convertToRealType(resultType()), derivedLhs, derivedRhs);

	return derivedOp->getResults();
}

void AddElementWiseOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(lhs());
	toBeDerived.push_back(rhs());
}

void AddElementWiseOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

mlir::Type AddElementWiseOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value AddElementWiseOp::lhs()
{
	return Adaptor(*this).lhs();
}

mlir::Value AddElementWiseOp::rhs()
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

llvm::ArrayRef<llvm::StringRef> SubOp::getAttributeNames()
{
	return {};
}

void SubOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	if (auto arrayType = resultType.dyn_cast<ArrayType>())
		if (arrayType.getAllocationScope() == BufferAllocationScope::unknown)
			resultType = arrayType.toMinAllowedAllocationScope();

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
			parser.parseRParen() || parser.parseArrow() ||
			parser.parseType(resultType) ||
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
	if (lhs().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());

	if (rhs().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());

  populateAllocationEffects(effects, getResult());

	if (resultType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
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
		mlir::Type addType = getMostGenericType(rhs().getType(), nestedOperand.getType());
		auto right = builder.create<AddOp>(getLoc(), addType, nestedOperand, rhs());

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
		mlir::Type addType = getMostGenericType(lhs().getType(), nestedOperand.getType());
		auto right = builder.create<SubOp>(getLoc(), addType, lhs(), nestedOperand);

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
	return builder.create<SubOp>(getLoc(), resultType, rhs(), lhs());
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
		if (auto casted = mlir::dyn_cast<DivOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeDivOp(builder, resultType, value);

		return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = distributeFn(this->rhs());

	return builder.create<SubOp>(getLoc(), resultType, lhs, rhs);
}

mlir::ValueRange SubOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	mlir::Location loc = getLoc();

	mlir::Value derivedLhs = derivatives.lookup(lhs());
	mlir::Value derivedRhs = derivatives.lookup(rhs());

	auto derivedOp = builder.create<SubOp>(
			loc, convertToRealType(resultType()), derivedLhs, derivedRhs);

	return derivedOp->getResults();
}

void SubOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(lhs());
	toBeDerived.push_back(rhs());
}

void SubOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

void SubOp::foldConstants(mlir::OpBuilder& builder)
{
	if (isOperandFoldable(lhs()) && getAttributeValue(mlir::cast<ConstantOp>(lhs().getDefiningOp()).value()) == 0.0)
	{
		mlir::OpBuilder::InsertionGuard guard(builder);
		builder.setInsertionPoint(*this);
		mlir::Value newOp = builder.create<NegateOp>(getLoc(), resultType(), rhs());

		replaceAllUsesWith(newOp);
		erase();
		return;
	}

	if (isOperandFoldable(rhs()) && getAttributeValue(mlir::cast<ConstantOp>(rhs().getDefiningOp()).value()) == 0.0)
	{
		replaceAllUsesWith(lhs());
		erase();
		return;
	}

	if (!isOperandFoldable(lhs()) || !isOperandFoldable(rhs()))
		return;

	// Note: this constant folding is done also on Subscription indexes.
	ConstantOp leftOp = mlir::cast<ConstantOp>(lhs().getDefiningOp());
	ConstantOp rightOp = mlir::cast<ConstantOp>(rhs().getDefiningOp());

	double left = getAttributeValue(leftOp.value());
	double right = getAttributeValue(rightOp.value());

	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	mlir::Type type = getMostGenericType(leftOp.resultType(), rightOp.resultType());
	mlir::Value newOp = builder.create<ConstantOp>(getLoc(), getAttribute(builder, type, left - right));

	replaceAllUsesWith(newOp);
	erase();
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
// Modelica::SubElementWiseOp
//===----------------------------------------------------------------------===//

mlir::Value SubElementWiseOpAdaptor::lhs()
{
	return getValues()[0];
}

mlir::Value SubElementWiseOpAdaptor::rhs()
{
	return getValues()[1];
}

llvm::ArrayRef<llvm::StringRef> SubElementWiseOp::getAttributeNames()
{
	return {};
}

void SubElementWiseOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	if (auto arrayType = resultType.dyn_cast<ArrayType>())
		if (arrayType.getAllocationScope() == BufferAllocationScope::unknown)
			resultType = arrayType.toMinAllowedAllocationScope();

	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::ParseResult SubElementWiseOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void SubElementWiseOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << lhs() << ", " << rhs() << " : ("
					<< lhs().getType() << ", " << rhs().getType()
					<< ") -> " << resultType();
}

void SubElementWiseOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (lhs().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());

	if (rhs().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());

  populateAllocationEffects(effects, getResult());

	if (resultType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
}

mlir::LogicalResult SubElementWiseOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	if (auto size = currentResult.size(); size != 1)
		return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");

	mlir::Value toNest = currentResult[0];

	if (argumentIndex == 0)
	{
		mlir::Value nestedOperand = readValue(builder, toNest);
		mlir::Type addType = getMostGenericType(rhs().getType(), nestedOperand.getType());
		auto right = builder.create<AddElementWiseOp>(getLoc(), addType, nestedOperand, rhs());

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
		mlir::Type addType = getMostGenericType(lhs().getType(), nestedOperand.getType());
		auto right = builder.create<SubElementWiseOp>(getLoc(), addType, lhs(), nestedOperand);

		for (auto& use : toNest.getUses())
			if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
				use.set(right.getResult());

		replaceAllUsesWith(rhs());
		erase();

		return mlir::success();
	}

	return emitError("Index out of bounds: " + std::to_string(argumentIndex));
}

mlir::Value SubElementWiseOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
{
	mlir::OpBuilder::InsertionGuard guard(builder);
	return builder.create<SubElementWiseOp>(getLoc(), resultType, rhs(), lhs());
}

mlir::Value SubElementWiseOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeMulOp(builder, resultType, value);

		return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = distributeFn(this->rhs());

	return builder.create<SubElementWiseOp>(getLoc(), resultType, lhs, rhs);
}

mlir::Value SubElementWiseOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<DivOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeDivOp(builder, resultType, value);

		return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = distributeFn(this->rhs());

	return builder.create<SubElementWiseOp>(getLoc(), resultType, lhs, rhs);
}

mlir::ValueRange SubElementWiseOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	mlir::Location loc = getLoc();

	mlir::Value derivedLhs = derivatives.lookup(lhs());
	mlir::Value derivedRhs = derivatives.lookup(rhs());

	auto derivedOp = builder.create<SubElementWiseOp>(
			loc, convertToRealType(resultType()), derivedLhs, derivedRhs);

	return derivedOp->getResults();
}

void SubElementWiseOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(lhs());
	toBeDerived.push_back(rhs());
}

void SubElementWiseOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

mlir::Type SubElementWiseOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value SubElementWiseOp::lhs()
{
	return Adaptor(*this).lhs();
}

mlir::Value SubElementWiseOp::rhs()
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

llvm::ArrayRef<llvm::StringRef> MulOp::getAttributeNames()
{
	return {};
}

void MulOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	if (auto arrayType = resultType.dyn_cast<ArrayType>())
		if (arrayType.getAllocationScope() == BufferAllocationScope::unknown)
			resultType = arrayType.toMinAllowedAllocationScope();

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
	if (lhs().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());

	if (rhs().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());

  populateAllocationEffects(effects, getResult());

	if (resultType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
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
		mlir::Type divType = getMostGenericType(rhs().getType(), nestedOperand.getType());
		auto right = builder.create<DivOp>(getLoc(), divType, nestedOperand, rhs());

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
		mlir::Type divType = getMostGenericType(lhs().getType(), nestedOperand.getType());
		auto right = builder.create<DivOp>(getLoc(), divType, nestedOperand, lhs());

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
		if (auto casted = mlir::dyn_cast<DivOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeDivOp(builder, resultType, value);

		return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = this->rhs();

	return builder.create<MulOp>(getLoc(), resultType, lhs, rhs);
}

mlir::ValueRange MulOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	mlir::Location loc = getLoc();

	mlir::Value derivedLhs = derivatives.lookup(lhs());
	mlir::Value derivedRhs = derivatives.lookup(rhs());

	mlir::Type type = convertToRealType(resultType());

	mlir::Value firstMul = builder.create<MulOp>(loc, type, derivedLhs, rhs());
	mlir::Value secondMul = builder.create<MulOp>(loc, type, lhs(), derivedRhs);
	auto derivedOp = builder.create<AddOp>(loc, type, firstMul, secondMul);

	return derivedOp->getResults();
}

void MulOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(lhs());
	toBeDerived.push_back(rhs());
}

void MulOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

void MulOp::foldConstants(mlir::OpBuilder& builder)
{
	if (isOperandFoldable(lhs()))
	{
		double value = getAttributeValue(mlir::cast<ConstantOp>(lhs().getDefiningOp()).value());

		if (value == 0.0)
		{
			replaceAllUsesWith(lhs());
			erase();
			return;
		}

		if (value == 1.0)
		{
			replaceAllUsesWith(rhs());
			erase();
			return;
		}
	}

	if (isOperandFoldable(rhs()))
	{
		double value = getAttributeValue(mlir::cast<ConstantOp>(rhs().getDefiningOp()).value());

		if (value == 0.0)
		{
			replaceAllUsesWith(rhs());
			erase();
			return;
		}

		if (value == 1.0)
		{
			replaceAllUsesWith(lhs());
			erase();
			return;
		}
	}

	if (!isOperandFoldable(lhs()) || !isOperandFoldable(rhs()))
		return;

	ConstantOp leftOp = mlir::cast<ConstantOp>(lhs().getDefiningOp());
	ConstantOp rightOp = mlir::cast<ConstantOp>(rhs().getDefiningOp());

	double left = getAttributeValue(leftOp.value());
	double right = getAttributeValue(rightOp.value());

	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	mlir::Type type = getMostGenericType(leftOp.resultType(), rightOp.resultType());
	mlir::Value newOp = builder.create<ConstantOp>(getLoc(), getAttribute(builder, type, left * right));

	replaceAllUsesWith(newOp);
	erase();
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
// Modelica::MulElementWiseOp
//===----------------------------------------------------------------------===//

mlir::Value MulElementWiseOpAdaptor::lhs()
{
	return getValues()[0];
}

mlir::Value MulElementWiseOpAdaptor::rhs()
{
	return getValues()[1];
}

llvm::ArrayRef<llvm::StringRef> MulElementWiseOp::getAttributeNames()
{
	return {};
}

void MulElementWiseOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	if (auto arrayType = resultType.dyn_cast<ArrayType>())
		if (arrayType.getAllocationScope() == BufferAllocationScope::unknown)
			resultType = arrayType.toMinAllowedAllocationScope();

	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::ParseResult MulElementWiseOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void MulElementWiseOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << lhs() << ", " << rhs() << " : ("
					<< lhs().getType() << ", " << rhs().getType()
					<< ") -> " << resultType();
}

void MulElementWiseOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (lhs().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());

	if (rhs().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());

  populateAllocationEffects(effects, getResult());

	if (resultType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
}

mlir::LogicalResult MulElementWiseOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	if (auto size = currentResult.size(); size != 1)
		return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");

	mlir::Value toNest = currentResult[0];

	if (argumentIndex == 0)
	{
		mlir::Value nestedOperand = readValue(builder, toNest);
		mlir::Type divType = getMostGenericType(rhs().getType(), nestedOperand.getType());
		auto right = builder.create<DivElementWiseOp>(getLoc(), divType, nestedOperand, rhs());

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
		mlir::Type divType = getMostGenericType(lhs().getType(), nestedOperand.getType());
		auto right = builder.create<DivElementWiseOp>(getLoc(), divType, nestedOperand, lhs());

		for (auto& use : toNest.getUses())
			if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
				use.set(right.getResult());

		getResult().replaceAllUsesWith(rhs());
		erase();

		return mlir::success();
	}

	return emitError("Index out of bounds: " + std::to_string(argumentIndex));
}

mlir::Value MulElementWiseOp::distribute(mlir::OpBuilder& builder)
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

mlir::Value MulElementWiseOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeNegateOp(builder, resultType);

		return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = this->rhs();

	return builder.create<MulElementWiseOp>(getLoc(), resultType, lhs, rhs);
}

mlir::Value MulElementWiseOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeMulOp(builder, resultType, value);

		return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = this->rhs();

	return builder.create<MulElementWiseOp>(getLoc(), resultType, lhs, rhs);
}

mlir::Value MulElementWiseOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<DivOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeDivOp(builder, resultType, value);

		return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = this->rhs();

	return builder.create<MulElementWiseOp>(getLoc(), resultType, lhs, rhs);
}

mlir::ValueRange MulElementWiseOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	mlir::Location loc = getLoc();

	mlir::Value derivedLhs = derivatives.lookup(lhs());
	mlir::Value derivedRhs = derivatives.lookup(rhs());

	mlir::Type type = convertToRealType(resultType());

	mlir::Value firstMul = builder.create<MulElementWiseOp>(loc, type, derivedLhs, rhs());
	mlir::Value secondMul = builder.create<MulElementWiseOp>(loc, type, lhs(), derivedRhs);
	auto derivedOp = builder.create<AddElementWiseOp>(loc, type, firstMul, secondMul);

	return derivedOp->getResults();
}

void MulElementWiseOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(lhs());
	toBeDerived.push_back(rhs());
}

void MulElementWiseOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

mlir::Type MulElementWiseOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value MulElementWiseOp::lhs()
{
	return Adaptor(*this).lhs();
}

mlir::Value MulElementWiseOp::rhs()
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

llvm::ArrayRef<llvm::StringRef> DivOp::getAttributeNames()
{
	return {};
}

void DivOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	if (auto arrayType = resultType.dyn_cast<ArrayType>())
		if (arrayType.getAllocationScope() == BufferAllocationScope::unknown)
			resultType = arrayType.toMinAllowedAllocationScope();

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
			parser.parseRParen() || parser.parseArrow() ||
			parser.parseType(resultType) ||
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
	if (lhs().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());

	if (rhs().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());

  populateAllocationEffects(effects, getResult());

	if (resultType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
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
		mlir::Type mulType = getMostGenericType(rhs().getType(), nestedOperand.getType());
		auto right = builder.create<MulOp>(getLoc(), mulType, nestedOperand, rhs());

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
		mlir::Type mulType = getMostGenericType(lhs().getType(), nestedOperand.getType());
		auto right = builder.create<DivOp>(getLoc(), mulType, lhs(), nestedOperand);

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

	return builder.create<DivOp>(getLoc(), resultType, lhs, rhs);
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
		if (auto casted = mlir::dyn_cast<DivOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeDivOp(builder, resultType, value);

		return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = this->rhs();

	return builder.create<DivOp>(getLoc(), resultType, lhs, rhs);
}

mlir::ValueRange DivOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	mlir::Location loc = getLoc();

	mlir::Value derivedLhs = derivatives.lookup(lhs());
	mlir::Value derivedRhs = derivatives.lookup(rhs());

	mlir::Type type = convertToRealType(resultType());

	mlir::Value firstMul = builder.create<MulOp>(loc, type, derivedLhs, rhs());
	mlir::Value secondMul = builder.create<MulOp>(loc, type, lhs(), derivedRhs);
	mlir::Value numerator = builder.create<SubOp>(loc, type, firstMul, secondMul);
	mlir::Value denominator = builder.create<MulOp>(loc, convertToRealType(rhs().getType()), rhs(), rhs());
	auto derivedOp = builder.create<DivOp>(loc, type, numerator, denominator);

	return derivedOp->getResults();
}

void DivOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(lhs());
	toBeDerived.push_back(rhs());
}

void DivOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

void DivOp::foldConstants(mlir::OpBuilder& builder)
{
	if (isOperandFoldable(lhs()) && getAttributeValue(mlir::cast<ConstantOp>(lhs().getDefiningOp()).value()) == 0.0)
	{
		replaceAllUsesWith(lhs());
		erase();
		return;
	}

	if (isOperandFoldable(rhs()) && getAttributeValue(mlir::cast<ConstantOp>(rhs().getDefiningOp()).value()) == 1.0)
	{
		replaceAllUsesWith(lhs());
		erase();
		return;
	}

	if (!isOperandFoldable(lhs()) || !isOperandFoldable(rhs()))
		return;

	ConstantOp leftOp = mlir::cast<ConstantOp>(lhs().getDefiningOp());
	ConstantOp rightOp = mlir::cast<ConstantOp>(rhs().getDefiningOp());

	double left = getAttributeValue(leftOp.value());
	double right = getAttributeValue(rightOp.value());

	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	mlir::Type type = getMostGenericType(leftOp.resultType(), rightOp.resultType());
	mlir::Value newOp = builder.create<ConstantOp>(getLoc(), getAttribute(builder, type, left / right));

	replaceAllUsesWith(newOp);
	erase();
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
// Modelica::DivElementWiseOp
//===----------------------------------------------------------------------===//

mlir::Value DivElementWiseOpAdaptor::lhs()
{
	return getValues()[0];
}

mlir::Value DivElementWiseOpAdaptor::rhs()
{
	return getValues()[1];
}

llvm::ArrayRef<llvm::StringRef> DivElementWiseOp::getAttributeNames()
{
	return {};
}

void DivElementWiseOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	if (auto arrayType = resultType.dyn_cast<ArrayType>())
		if (arrayType.getAllocationScope() == BufferAllocationScope::unknown)
			resultType = arrayType.toMinAllowedAllocationScope();

	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

mlir::ParseResult DivElementWiseOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void DivElementWiseOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << lhs() << ", " << rhs() << " : ("
					<< lhs().getType() << ", " << rhs().getType()
					<< ") -> " << resultType();
}

void DivElementWiseOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (lhs().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());

	if (rhs().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());

  populateAllocationEffects(effects, getResult());

	if (resultType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
}

mlir::LogicalResult DivElementWiseOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	if (auto size = currentResult.size(); size != 1)
		return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");

	mlir::Value toNest = currentResult[0];

	if (argumentIndex == 0)
	{
		mlir::Value nestedOperand = readValue(builder, toNest);
		mlir::Type mulType = getMostGenericType(rhs().getType(), nestedOperand.getType());
		auto right = builder.create<MulElementWiseOp>(getLoc(), mulType, nestedOperand, rhs());

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
		mlir::Type mulType = getMostGenericType(lhs().getType(), nestedOperand.getType());
		auto right = builder.create<DivElementWiseOp>(getLoc(), mulType, lhs(), nestedOperand);

		for (auto& use : toNest.getUses())
			if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
				use.set(right.getResult());

		getResult().replaceAllUsesWith(rhs());
		erase();

		return mlir::success();
	}

	return emitError("Index out of bounds: " + std::to_string(argumentIndex));
}

mlir::Value DivElementWiseOp::distribute(mlir::OpBuilder& builder)
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

mlir::Value DivElementWiseOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeNegateOp(builder, resultType);

		return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = this->rhs();

	return builder.create<DivElementWiseOp>(getLoc(), resultType, lhs, rhs);
}

mlir::Value DivElementWiseOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeMulOp(builder, resultType, value);

		return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = this->rhs();

	return builder.create<DivElementWiseOp>(getLoc(), resultType, lhs, rhs);
}

mlir::Value DivElementWiseOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto distributeFn = [&](mlir::Value child) -> mlir::Value {
		if (auto casted = mlir::dyn_cast<DivOpDistributionInterface>(child.getDefiningOp()))
			return casted.distributeDivOp(builder, resultType, value);

		return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
	};

	mlir::Value lhs = distributeFn(this->lhs());
	mlir::Value rhs = this->rhs();

	return builder.create<DivElementWiseOp>(getLoc(), resultType, lhs, rhs);
}

mlir::ValueRange DivElementWiseOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	mlir::Location loc = getLoc();

	mlir::Value derivedLhs = derivatives.lookup(lhs());
	mlir::Value derivedRhs = derivatives.lookup(rhs());

	mlir::Type type = convertToRealType(resultType());

	mlir::Value firstMul = builder.create<MulElementWiseOp>(loc, type, derivedLhs, rhs());
	mlir::Value secondMul = builder.create<MulElementWiseOp>(loc, type, lhs(), derivedRhs);
	mlir::Value numerator = builder.create<SubElementWiseOp>(loc, type, firstMul, secondMul);
	mlir::Value denominator = builder.create<MulElementWiseOp>(loc, convertToRealType(rhs().getType()), rhs(), rhs());
	auto derivedOp = builder.create<DivElementWiseOp>(loc, type, numerator, denominator);

	return derivedOp->getResults();
}

void DivElementWiseOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(lhs());
	toBeDerived.push_back(rhs());
}

void DivElementWiseOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

mlir::Type DivElementWiseOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value DivElementWiseOp::lhs()
{
	return Adaptor(*this).lhs();
}

mlir::Value DivElementWiseOp::rhs()
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

llvm::ArrayRef<llvm::StringRef> PowOp::getAttributeNames()
{
	return {};
}

void PowOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value base, mlir::Value exponent)
{
	if (auto arrayType = resultType.dyn_cast<ArrayType>())
		if (arrayType.getAllocationScope() == BufferAllocationScope::unknown)
			resultType = arrayType.toMinAllowedAllocationScope();

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
			parser.parseRParen() || parser.parseArrow() ||
			parser.parseType(resultType) ||
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
	if (base().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), base(), mlir::SideEffects::DefaultResource::get());

	if (exponent().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), exponent(), mlir::SideEffects::DefaultResource::get());

  populateAllocationEffects(effects, getResult());

	if (resultType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
}

mlir::ValueRange PowOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	// D[x ^ y] = (x ^ y) * (y' * ln(x) + (y * x') / x)

	mlir::Location loc = getLoc();

	mlir::Value derivedBase = derivatives.lookup(base());
	mlir::Value derivedExponent = derivatives.lookup(exponent());

	mlir::Type type = convertToRealType(resultType());

	mlir::Value pow = builder.create<PowOp>(loc, type, base(), exponent());
	mlir::Value ln = builder.create<LogOp>(loc, type, base());
	mlir::Value firstOperand = builder.create<MulOp>(loc, type, derivedExponent, ln);
	mlir::Value numerator = builder.create<MulOp>(loc, type, exponent(), derivedBase);
	mlir::Value secondOperand = builder.create<DivOp>(loc, type, numerator, base());
	mlir::Value sum = builder.create<AddOp>(loc, type, firstOperand, secondOperand);
	auto derivedOp = builder.create<MulOp>(loc, type, pow, sum);

	return derivedOp->getResults();
}

void PowOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(base());
	toBeDerived.push_back(exponent());
}

void PowOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

void PowOp::foldConstants(mlir::OpBuilder& builder)
{
	if (isOperandFoldable(exponent()))
	{
		double value = getAttributeValue(mlir::cast<ConstantOp>(exponent().getDefiningOp()).value());

		if (value == 0.0)
		{
			mlir::OpBuilder::InsertionGuard guard(builder);
			builder.setInsertionPoint(*this);
			mlir::Value newOp = builder.create<ConstantOp>(getLoc(), RealAttribute::get(getContext(), 1));

			replaceAllUsesWith(newOp);
			erase();
			return;
		}

		if (value == 1.0)
		{
			replaceAllUsesWith(base());
			erase();
			return;
		}
	}

	if (isOperandFoldable(base()))
	{
		double value = getAttributeValue(mlir::cast<ConstantOp>(base().getDefiningOp()).value());

		if (value == 0.0 || value == 1.0)
		{
			replaceAllUsesWith(base());
			erase();
			return;
		}
	}

	if (!isOperandFoldable(base()) || !isOperandFoldable(exponent()))
		return;

	ConstantOp baseOp = mlir::cast<ConstantOp>(base().getDefiningOp());
	ConstantOp exponentOp = mlir::cast<ConstantOp>(exponent().getDefiningOp());

	double base = getAttributeValue(baseOp.value());
	double exponent = getAttributeValue(exponentOp.value());

	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	mlir::Value newOp = builder.create<ConstantOp>(getLoc(), RealAttribute::get(getContext(), pow(base, exponent)));

	replaceAllUsesWith(newOp);
	erase();
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
// Modelica::PowElementWiseOp
//===----------------------------------------------------------------------===//

mlir::Value PowElementWiseOpAdaptor::base()
{
	return getValues()[0];
}

mlir::Value PowElementWiseOpAdaptor::exponent()
{
	return getValues()[1];
}

llvm::ArrayRef<llvm::StringRef> PowElementWiseOp::getAttributeNames()
{
	return {};
}

void PowElementWiseOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value base, mlir::Value exponent)
{
	if (auto arrayType = resultType.dyn_cast<ArrayType>())
		if (arrayType.getAllocationScope() == BufferAllocationScope::unknown)
			resultType = arrayType.toMinAllowedAllocationScope();

	state.addTypes(resultType);
	state.addOperands({ base, exponent });
}

mlir::ParseResult PowElementWiseOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void PowElementWiseOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << base() << ", " << exponent() << " : ("
					<< base().getType() << ", " << exponent().getType()
					<< ") -> " << resultType();
}

void PowElementWiseOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (base().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), base(), mlir::SideEffects::DefaultResource::get());

	if (exponent().getType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), exponent(), mlir::SideEffects::DefaultResource::get());

  populateAllocationEffects(effects, getResult());

	if (resultType().isa<ArrayType>())
		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
}

mlir::ValueRange PowElementWiseOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	// D[x ^ y] = (x ^ y) * (y' * ln(x) + (y * x') / x)

	mlir::Location loc = getLoc();

	mlir::Value derivedBase = derivatives.lookup(base());
	mlir::Value derivedExponent = derivatives.lookup(exponent());

	mlir::Type type = convertToRealType(resultType());

	mlir::Value pow = builder.create<PowElementWiseOp>(loc, type, base(), exponent());
	mlir::Value ln = builder.create<LogOp>(loc, type, base());
	mlir::Value firstOperand = builder.create<MulElementWiseOp>(loc, type, derivedExponent, ln);
	mlir::Value numerator = builder.create<MulElementWiseOp>(loc, type, exponent(), derivedBase);
	mlir::Value secondOperand = builder.create<DivElementWiseOp>(loc, type, numerator, base());
	mlir::Value sum = builder.create<AddElementWiseOp>(loc, type, firstOperand, secondOperand);
	auto derivedOp = builder.create<MulElementWiseOp>(loc, type, pow, sum);

	return derivedOp->getResults();
}

void PowElementWiseOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(base());
	toBeDerived.push_back(exponent());
}

void PowElementWiseOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

mlir::Type PowElementWiseOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value PowElementWiseOp::base()
{
	return Adaptor(*this).base();
}

mlir::Value PowElementWiseOp::exponent()
{
	return Adaptor(*this).exponent();
}

//===----------------------------------------------------------------------===//
// Modelica::AbsOp
//===----------------------------------------------------------------------===//

mlir::Value AbsOpAdaptor::operand()
{
	return getValues()[0];
}

llvm::ArrayRef<llvm::StringRef> AbsOp::getAttributeNames()
{
	return {};
}

void AbsOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand)
{
	state.addTypes(resultType);
	state.addOperands(operand);
}

mlir::ParseResult AbsOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void AbsOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << operand() << " : " << operand().getType() << " -> " << resultType();
}

mlir::ValueRange AbsOp::getArgs()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

unsigned int AbsOp::getArgExpectedRank(unsigned int argIndex)
{
	return 0;
}

mlir::ValueRange AbsOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
{
	mlir::Type newResultType = resultType().cast<ArrayType>().slice(indexes.size());

	if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newResultType = arrayType.getElementType();

	mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

	if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newOperand = builder.create<LoadOp>(getLoc(), newOperand);

	auto op = builder.create<AbsOp>(getLoc(), newResultType, newOperand);
	return op->getResults();
}

mlir::ValueRange AbsOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	// D[abs(x)] = x' * sign(x)

	mlir::Location loc = getLoc();
	mlir::Value derivedOperand = derivatives.lookup(operand());
	mlir::Type type = convertToRealType(resultType());

	mlir::Value sign = builder.create<SignOp>(loc, type, operand());
	auto derivedOp = builder.create<MulElementWiseOp>(loc, type, derivedOperand, sign);
	return derivedOp->getResults();
}

void AbsOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{

}

void AbsOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

void AbsOp::foldConstants(mlir::OpBuilder& builder)
{
	if (!isOperandFoldable(operand()))
		return;

	ConstantOp operandOp = mlir::cast<ConstantOp>(operand().getDefiningOp());

	double operand = getAttributeValue(operandOp.value());

	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	mlir::Value newOp = builder.create<ConstantOp>(getLoc(), RealAttribute::get(getContext(), std::abs(operand)));

	replaceAllUsesWith(newOp);
	erase();
}

mlir::Type AbsOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value AbsOp::operand()
{
	return Adaptor(*this).operand();
}

//===----------------------------------------------------------------------===//
// Modelica::SignOp
//===----------------------------------------------------------------------===//

mlir::Value SignOpAdaptor::operand()
{
	return getValues()[0];
}

llvm::ArrayRef<llvm::StringRef> SignOp::getAttributeNames()
{
	return {};
}

void SignOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand)
{
	state.addTypes(resultType);
	state.addOperands(operand);
}

mlir::ParseResult SignOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void SignOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << operand() << " : " << operand().getType() << " -> " << resultType();
}

mlir::ValueRange SignOp::getArgs()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

unsigned int SignOp::getArgExpectedRank(unsigned int argIndex)
{
	return 0;
}

mlir::ValueRange SignOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
{
	mlir::Type newResultType = resultType().cast<ArrayType>().slice(indexes.size());

	if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newResultType = arrayType.getElementType();

	mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

	if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newOperand = builder.create<LoadOp>(getLoc(), newOperand);

	auto op = builder.create<SignOp>(getLoc(), newResultType, newOperand);
	return op->getResults();
}

mlir::ValueRange SignOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	// D[sign(x)] = 0

	auto derivedOp = builder.create<ConstantOp>(getLoc(), RealAttribute::get(getContext(), 0));
	return derivedOp->getResults();
}

void SignOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{

}

void SignOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

void SignOp::foldConstants(mlir::OpBuilder& builder)
{
	if (!isOperandFoldable(operand()))
		return;

	ConstantOp operandOp = mlir::cast<ConstantOp>(operand().getDefiningOp());

	double operand = getAttributeValue(operandOp.value());

	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	mlir::Value newOp = builder.create<ConstantOp>(
		getLoc(), RealAttribute::get(getContext(), (operand > 0.0) - (operand < 0.0)));

	replaceAllUsesWith(newOp);
	erase();
}

mlir::Type SignOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value SignOp::operand()
{
	return Adaptor(*this).operand();
}

//===----------------------------------------------------------------------===//
// Modelica::SqrtOp
//===----------------------------------------------------------------------===//

mlir::Value SqrtOpAdaptor::operand()
{
	return getValues()[0];
}

llvm::ArrayRef<llvm::StringRef> SqrtOp::getAttributeNames()
{
	return {};
}

void SqrtOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand)
{
	state.addTypes(resultType);
	state.addOperands(operand);
}

mlir::ParseResult SqrtOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void SqrtOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << operand() << " : " << operand().getType() << " -> " << resultType();
}

mlir::ValueRange SqrtOp::getArgs()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

unsigned int SqrtOp::getArgExpectedRank(unsigned int argIndex)
{
	return 0;
}

mlir::ValueRange SqrtOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
{
	mlir::Type newResultType = resultType().cast<ArrayType>().slice(indexes.size());

	if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newResultType = arrayType.getElementType();

	mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

	if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newOperand = builder.create<LoadOp>(getLoc(), newOperand);

	auto op = builder.create<SqrtOp>(getLoc(), newResultType, newOperand);
	return op->getResults();
}

mlir::ValueRange SqrtOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	// D[sqrt(x)] = x' / sqrt(x) / 2

	mlir::Location loc = getLoc();
	mlir::Value derivedOperand = derivatives.lookup(operand());
	mlir::Type type = convertToRealType(resultType());

	mlir::Value sqrt = builder.create<SqrtOp>(loc, type, operand());
	mlir::Value numerator = builder.create<DivElementWiseOp>(loc, type, derivedOperand, sqrt);
	mlir::Value two = builder.create<ConstantOp>(loc, RealAttribute::get(getContext(), 2));
	auto derivedOp = builder.create<DivElementWiseOp>(loc, type, numerator, two);

	return derivedOp->getResults();
}

void SqrtOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(operand());
}

void SqrtOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

void SqrtOp::foldConstants(mlir::OpBuilder& builder)
{
	if (!isOperandFoldable(operand()))
		return;

	ConstantOp operandOp = mlir::cast<ConstantOp>(operand().getDefiningOp());

	double operand = getAttributeValue(operandOp.value());

	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	mlir::Value newOp = builder.create<ConstantOp>(getLoc(), RealAttribute::get(getContext(), sqrt(operand)));

	replaceAllUsesWith(newOp);
	erase();
}

mlir::Type SqrtOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value SqrtOp::operand()
{
	return Adaptor(*this).operand();
}

//===----------------------------------------------------------------------===//
// Modelica::SinOp
//===----------------------------------------------------------------------===//

mlir::Value SinOpAdaptor::operand()
{
	return getValues()[0];
}

llvm::ArrayRef<llvm::StringRef> SinOp::getAttributeNames()
{
	return {};
}

void SinOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand)
{
	state.addTypes(resultType);
	state.addOperands(operand);
}

mlir::ParseResult SinOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void SinOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << operand() << " : " << operand().getType() << " -> " << resultType();
}

mlir::ValueRange SinOp::getArgs()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

unsigned int SinOp::getArgExpectedRank(unsigned int argIndex)
{
	return 0;
}

mlir::ValueRange SinOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
{
	mlir::Type newResultType = resultType().cast<ArrayType>().slice(indexes.size());

	if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newResultType = arrayType.getElementType();

	mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

	if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newOperand = builder.create<LoadOp>(getLoc(), newOperand);

	auto op = builder.create<SinOp>(getLoc(), newResultType, newOperand);
	return op->getResults();
}

mlir::ValueRange SinOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	// D[sin(x)] = x' * cos(x)

	mlir::Location loc = getLoc();
	mlir::Value derivedOperand = derivatives.lookup(operand());
	mlir::Type type = convertToRealType(resultType());

	mlir::Value cos = builder.create<CosOp>(loc, type, operand());
	auto derivedOp = builder.create<MulElementWiseOp>(loc, type, cos, derivedOperand);

	return derivedOp->getResults();
}

void SinOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(operand());
}

void SinOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

void SinOp::foldConstants(mlir::OpBuilder& builder)
{
	if (!isOperandFoldable(operand()))
		return;

	ConstantOp operandOp = mlir::cast<ConstantOp>(operand().getDefiningOp());

	double operand = getAttributeValue(operandOp.value());

	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	mlir::Value newOp = builder.create<ConstantOp>(getLoc(), RealAttribute::get(getContext(), sin(operand)));

	replaceAllUsesWith(newOp);
	erase();
}

mlir::Type SinOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value SinOp::operand()
{
	return Adaptor(*this).operand();
}

//===----------------------------------------------------------------------===//
// Modelica::CosOp
//===----------------------------------------------------------------------===//

mlir::Value CosOpAdaptor::operand()
{
	return getValues()[0];
}

llvm::ArrayRef<llvm::StringRef> CosOp::getAttributeNames()
{
	return {};
}

void CosOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand)
{
	state.addTypes(resultType);
	state.addOperands(operand);
}

mlir::ParseResult CosOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void CosOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << operand() << " : " << operand().getType() << " -> " << resultType();
}

mlir::ValueRange CosOp::getArgs()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

unsigned int CosOp::getArgExpectedRank(unsigned int argIndex)
{
	return 0;
}

mlir::ValueRange CosOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
{
	mlir::Type newResultType = resultType().cast<ArrayType>().slice(indexes.size());

	if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newResultType = arrayType.getElementType();

	mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

	if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newOperand = builder.create<LoadOp>(getLoc(), newOperand);

	auto op = builder.create<CosOp>(getLoc(), newResultType, newOperand);
	return op->getResults();
}

mlir::ValueRange CosOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	// D[cos(x)] = -x' * sin(x)

	mlir::Location loc = getLoc();
	mlir::Value derivedOperand = derivatives.lookup(operand());
	mlir::Type type = convertToRealType(resultType());

	mlir::Value sin = builder.create<SinOp>(loc, type, operand());
	mlir::Value negatedSin = builder.create<NegateOp>(loc, type, sin);
	auto derivedOp = builder.create<MulElementWiseOp>(loc, type, negatedSin, derivedOperand);

	return derivedOp->getResults();
}

void CosOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(operand());
}

void CosOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

void CosOp::foldConstants(mlir::OpBuilder& builder)
{
	if (!isOperandFoldable(operand()))
		return;

	ConstantOp operandOp = mlir::cast<ConstantOp>(operand().getDefiningOp());

	double operand = getAttributeValue(operandOp.value());

	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	mlir::Value newOp = builder.create<ConstantOp>(getLoc(), RealAttribute::get(getContext(), cos(operand)));

	replaceAllUsesWith(newOp);
	erase();
}

mlir::Type CosOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value CosOp::operand()
{
	return Adaptor(*this).operand();
}

//===----------------------------------------------------------------------===//
// Modelica::TanOp
//===----------------------------------------------------------------------===//

mlir::Value TanOpAdaptor::operand()
{
	return getValues()[0];
}

llvm::ArrayRef<llvm::StringRef> TanOp::getAttributeNames()
{
	return {};
}

void TanOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand)
{
	state.addTypes(resultType);
	state.addOperands(operand);
}

mlir::ParseResult TanOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void TanOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << operand() << " : " << operand().getType() << " -> " << resultType();
}

mlir::ValueRange TanOp::getArgs()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

unsigned int TanOp::getArgExpectedRank(unsigned int argIndex)
{
	return 0;
}

mlir::ValueRange TanOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
{
	mlir::Type newResultType = resultType().cast<ArrayType>().slice(indexes.size());

	if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newResultType = arrayType.getElementType();

	mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

	if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newOperand = builder.create<LoadOp>(getLoc(), newOperand);

	auto op = builder.create<TanOp>(getLoc(), newResultType, newOperand);
	return op->getResults();
}

mlir::ValueRange TanOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	// D[tan(x)] = x' / (cos(x))^2

	mlir::Location loc = getLoc();
	mlir::Value derivedOperand = derivatives.lookup(operand());
	mlir::Type type = convertToRealType(resultType());

	mlir::Value cos = builder.create<CosOp>(loc, type, operand());
	mlir::Value denominator = builder.create<MulElementWiseOp>(loc, type, cos, cos);
	auto derivedOp = builder.create<DivElementWiseOp>(loc, type, derivedOperand, denominator);

	return derivedOp->getResults();
}

void TanOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(operand());
}

void TanOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

void TanOp::foldConstants(mlir::OpBuilder& builder)
{
	if (!isOperandFoldable(operand()))
		return;

	ConstantOp operandOp = mlir::cast<ConstantOp>(operand().getDefiningOp());

	double operand = getAttributeValue(operandOp.value());

	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	mlir::Value newOp = builder.create<ConstantOp>(getLoc(), RealAttribute::get(getContext(), tan(operand)));

	replaceAllUsesWith(newOp);
	erase();
}

mlir::Type TanOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value TanOp::operand()
{
	return Adaptor(*this).operand();
}

//===----------------------------------------------------------------------===//
// Modelica::AsinOp
//===----------------------------------------------------------------------===//

mlir::Value AsinOpAdaptor::operand()
{
	return getValues()[0];
}

llvm::ArrayRef<llvm::StringRef> AsinOp::getAttributeNames()
{
	return {};
}

void AsinOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand)
{
	state.addTypes(resultType);
	state.addOperands(operand);
}

mlir::ParseResult AsinOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void AsinOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << operand() << " : " << operand().getType() << " -> " << resultType();
}

mlir::ValueRange AsinOp::getArgs()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

unsigned int AsinOp::getArgExpectedRank(unsigned int argIndex)
{
	return 0;
}

mlir::ValueRange AsinOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
{
	mlir::Type newResultType = resultType().cast<ArrayType>().slice(indexes.size());

	if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newResultType = arrayType.getElementType();

	mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

	if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newOperand = builder.create<LoadOp>(getLoc(), newOperand);

	auto op = builder.create<AsinOp>(getLoc(), newResultType, newOperand);
	return op->getResults();
}

mlir::ValueRange AsinOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	// D[arcsin(x)] = x' / sqrt(1 - x^2)

	mlir::Location loc = getLoc();
	mlir::Value derivedOperand = derivatives.lookup(operand());
	mlir::Type type = convertToRealType(resultType());

	mlir::Value one = builder.create<ConstantOp>(loc, RealAttribute::get(getContext(), 1));
	mlir::Value argSquared = builder.create<MulElementWiseOp>(loc, type, operand(), operand());
	mlir::Value sub = builder.create<SubElementWiseOp>(loc, type, one, argSquared);
	mlir::Value denominator = builder.create<SqrtOp>(loc, type, sub);
	auto derivedOp = builder.create<DivElementWiseOp>(loc, type, derivedOperand, denominator);

	return derivedOp->getResults();
}

void AsinOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(operand());
}

void AsinOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

void AsinOp::foldConstants(mlir::OpBuilder& builder)
{
	if (!isOperandFoldable(operand()))
		return;

	ConstantOp operandOp = mlir::cast<ConstantOp>(operand().getDefiningOp());

	double operand = getAttributeValue(operandOp.value());

	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	mlir::Value newOp = builder.create<ConstantOp>(getLoc(), RealAttribute::get(getContext(), asin(operand)));

	replaceAllUsesWith(newOp);
	erase();
}

mlir::Type AsinOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value AsinOp::operand()
{
	return Adaptor(*this).operand();
}

//===----------------------------------------------------------------------===//
// Modelica::AcosOp
//===----------------------------------------------------------------------===//

mlir::Value AcosOpAdaptor::operand()
{
	return getValues()[0];
}

llvm::ArrayRef<llvm::StringRef> AcosOp::getAttributeNames()
{
	return {};
}

void AcosOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand)
{
	state.addTypes(resultType);
	state.addOperands(operand);
}

mlir::ParseResult AcosOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void AcosOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << operand() << " : " << operand().getType() << " -> " << resultType();
}

mlir::ValueRange AcosOp::getArgs()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

unsigned int AcosOp::getArgExpectedRank(unsigned int argIndex)
{
	return 0;
}

mlir::ValueRange AcosOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
{
	mlir::Type newResultType = resultType().cast<ArrayType>().slice(indexes.size());

	if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newResultType = arrayType.getElementType();

	mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

	if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newOperand = builder.create<LoadOp>(getLoc(), newOperand);

	auto op = builder.create<AcosOp>(getLoc(), newResultType, newOperand);
	return op->getResults();
}

mlir::ValueRange AcosOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	// D[acos(x)] = -x' / sqrt(1 - x^2)

	mlir::Location loc = getLoc();
	mlir::Value derivedOperand = derivatives.lookup(operand());
	mlir::Type type = convertToRealType(resultType());

	mlir::Value one = builder.create<ConstantOp>(loc, RealAttribute::get(getContext(), 1));
	mlir::Value argSquared = builder.create<MulElementWiseOp>(loc, type, operand(), operand());
	mlir::Value sub = builder.create<SubElementWiseOp>(loc, type, one, argSquared);
	mlir::Value denominator = builder.create<SqrtOp>(loc, type, sub);
	mlir::Value div = builder.create<DivElementWiseOp>(loc, type, derivedOperand, denominator);
	auto derivedOp = builder.create<NegateOp>(loc, type, div);

	return derivedOp->getResults();
}

void AcosOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(operand());
}

void AcosOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

void AcosOp::foldConstants(mlir::OpBuilder& builder)
{
	if (!isOperandFoldable(operand()))
		return;

	ConstantOp operandOp = mlir::cast<ConstantOp>(operand().getDefiningOp());

	double operand = getAttributeValue(operandOp.value());

	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	mlir::Value newOp = builder.create<ConstantOp>(getLoc(), RealAttribute::get(getContext(), acos(operand)));

	replaceAllUsesWith(newOp);
	erase();
}

mlir::Type AcosOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value AcosOp::operand()
{
	return Adaptor(*this).operand();
}

//===----------------------------------------------------------------------===//
// Modelica::AtanOp
//===----------------------------------------------------------------------===//

mlir::Value AtanOpAdaptor::operand()
{
	return getValues()[0];
}

llvm::ArrayRef<llvm::StringRef> AtanOp::getAttributeNames()
{
	return {};
}

void AtanOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand)
{
	state.addTypes(resultType);
	state.addOperands(operand);
}

mlir::ParseResult AtanOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void AtanOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << operand() << " : " << operand().getType() << " -> " << resultType();
}

mlir::ValueRange AtanOp::getArgs()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

unsigned int AtanOp::getArgExpectedRank(unsigned int argIndex)
{
	return 0;
}

mlir::ValueRange AtanOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
{
	mlir::Type newResultType = resultType().cast<ArrayType>().slice(indexes.size());

	if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newResultType = arrayType.getElementType();

	mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

	if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newOperand = builder.create<LoadOp>(getLoc(), newOperand);

	auto op = builder.create<AtanOp>(getLoc(), newResultType, newOperand);
	return op->getResults();
}

mlir::ValueRange AtanOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	// D[atan(x)] = x' / (1 + x^2)

	mlir::Location loc = getLoc();
	mlir::Value derivedOperand = derivatives.lookup(operand());
	mlir::Type type = convertToRealType(resultType());

	mlir::Value one = builder.create<ConstantOp>(loc, RealAttribute::get(getContext(), 1));
	mlir::Value argSquared = builder.create<MulElementWiseOp>(loc, type, operand(), operand());
	mlir::Value denominator = builder.create<AddElementWiseOp>(loc, type, one, argSquared);
	auto derivedOp = builder.create<DivElementWiseOp>(loc, type, derivedOperand, denominator);

	return derivedOp->getResults();
}

void AtanOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(operand());
}

void AtanOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

void AtanOp::foldConstants(mlir::OpBuilder& builder)
{
	if (!isOperandFoldable(operand()))
		return;

	ConstantOp operandOp = mlir::cast<ConstantOp>(operand().getDefiningOp());

	double operand = getAttributeValue(operandOp.value());

	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	mlir::Value newOp = builder.create<ConstantOp>(getLoc(), RealAttribute::get(getContext(), atan(operand)));

	replaceAllUsesWith(newOp);
	erase();
}

mlir::Type AtanOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value AtanOp::operand()
{
	return Adaptor(*this).operand();
}

//===----------------------------------------------------------------------===//
// Modelica::Atan2Op
//===----------------------------------------------------------------------===//

mlir::Value Atan2OpAdaptor::y()
{
	return getValues()[0];
}

mlir::Value Atan2OpAdaptor::x()
{
	return getValues()[1];
}

llvm::ArrayRef<llvm::StringRef> Atan2Op::getAttributeNames()
{
	return {};
}

void Atan2Op::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value y, mlir::Value x)
{
	state.addTypes(resultType);
	state.addOperands(y);
	state.addOperands(x);
}

mlir::ParseResult Atan2Op::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void Atan2Op::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName()
					<< " " << y() << ", " << x()
					<< " : (" << y().getType() << ", " << x().getType()
					<< ") -> " << resultType();
}

mlir::ValueRange Atan2Op::getArgs()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

unsigned int Atan2Op::getArgExpectedRank(unsigned int argIndex)
{
	return 0;
}

mlir::ValueRange Atan2Op::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
{
	mlir::Type newResultType = resultType().cast<ArrayType>().slice(indexes.size());

	if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newResultType = arrayType.getElementType();

	mlir::Value newY = builder.create<SubscriptionOp>(getLoc(), y(), indexes);

	if (auto arrayType = newY.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newY = builder.create<LoadOp>(getLoc(), newY);

	mlir::Value newX = builder.create<SubscriptionOp>(getLoc(), x(), indexes);

	if (auto arrayType = newX.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newX = builder.create<LoadOp>(getLoc(), newX);

	auto op = builder.create<Atan2Op>(getLoc(), newResultType, newY, newX);
	return op->getResults();
}

mlir::ValueRange Atan2Op::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	// D[atan2(y, x)] = (y' * x - y * x') / (y^2 + x^2)

	mlir::Location loc = getLoc();
	mlir::Value derivedY = derivatives.lookup(y());
	mlir::Value derivedX = derivatives.lookup(x());
	mlir::Type type = convertToRealType(resultType());

	mlir::Value firstMul = builder.create<MulElementWiseOp>(loc, type, derivedY, x());
	mlir::Value secondMul = builder.create<MulElementWiseOp>(loc, type, y(), derivedX);
	mlir::Value numerator = builder.create<SubElementWiseOp>(loc, type, firstMul, secondMul);

	mlir::Value firstSquared = builder.create<MulElementWiseOp>(loc, type, y(), y());
	mlir::Value secondSquared = builder.create<MulElementWiseOp>(loc, type, x(), x());
	mlir::Value denominator = builder.create<AddElementWiseOp>(loc, type, firstSquared, secondSquared);
	auto derivedOp = builder.create<DivElementWiseOp>(loc, type, numerator, denominator);

	return derivedOp->getResults();
}

void Atan2Op::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(y());
	toBeDerived.push_back(x());
}

void Atan2Op::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

void Atan2Op::foldConstants(mlir::OpBuilder& builder)
{
	if (!isOperandFoldable(y()) || !isOperandFoldable(x()))
		return;

	ConstantOp yOp = mlir::cast<ConstantOp>(y().getDefiningOp());
	ConstantOp xOp = mlir::cast<ConstantOp>(x().getDefiningOp());

	double y = getAttributeValue(yOp.value());
	double x = getAttributeValue(xOp.value());

	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	mlir::Value newOp = builder.create<ConstantOp>(getLoc(), RealAttribute::get(getContext(), atan2(y, x)));

	replaceAllUsesWith(newOp);
	erase();
}

mlir::Type Atan2Op::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value Atan2Op::y()
{
	return Adaptor(*this).y();
}

mlir::Value Atan2Op::x()
{
	return Adaptor(*this).x();
}

//===----------------------------------------------------------------------===//
// Modelica::SinhOp
//===----------------------------------------------------------------------===//

mlir::Value SinhOpAdaptor::operand()
{
	return getValues()[0];
}

llvm::ArrayRef<llvm::StringRef> SinhOp::getAttributeNames()
{
	return {};
}

void SinhOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand)
{
	state.addTypes(resultType);
	state.addOperands(operand);
}

mlir::ParseResult SinhOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void SinhOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << operand() << " : " << operand().getType() << " -> " << resultType();
}

mlir::ValueRange SinhOp::getArgs()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

unsigned int SinhOp::getArgExpectedRank(unsigned int argIndex)
{
	return 0;
}

mlir::ValueRange SinhOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
{
	mlir::Type newResultType = resultType().cast<ArrayType>().slice(indexes.size());

	if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newResultType = arrayType.getElementType();

	mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

	if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newOperand = builder.create<LoadOp>(getLoc(), newOperand);

	auto op = builder.create<SinhOp>(getLoc(), newResultType, newOperand);
	return op->getResults();
}

mlir::ValueRange SinhOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	// D[sinh(x)] = x' * cosh(x)

	mlir::Location loc = getLoc();
	mlir::Value derivedOperand = derivatives.lookup(operand());
	mlir::Type type = convertToRealType(resultType());

	mlir::Value cosh = builder.create<CoshOp>(loc, type, operand());
	auto derivedOp = builder.create<MulElementWiseOp>(loc, type, cosh, derivedOperand);

	return derivedOp->getResults();
}

void SinhOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(operand());
}

void SinhOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

void SinhOp::foldConstants(mlir::OpBuilder& builder)
{
	if (!isOperandFoldable(operand()))
		return;

	ConstantOp operandOp = mlir::cast<ConstantOp>(operand().getDefiningOp());

	double operand = getAttributeValue(operandOp.value());

	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	mlir::Value newOp = builder.create<ConstantOp>(getLoc(), RealAttribute::get(getContext(), sinh(operand)));

	replaceAllUsesWith(newOp);
	erase();
}

mlir::Type SinhOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value SinhOp::operand()
{
	return Adaptor(*this).operand();
}

//===----------------------------------------------------------------------===//
// Modelica::CoshOp
//===----------------------------------------------------------------------===//

mlir::Value CoshOpAdaptor::operand()
{
	return getValues()[0];
}

llvm::ArrayRef<llvm::StringRef> CoshOp::getAttributeNames()
{
	return {};
}

void CoshOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand)
{
	state.addTypes(resultType);
	state.addOperands(operand);
}

mlir::ParseResult CoshOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void CoshOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << operand() << " : " << operand().getType() << " -> " << resultType();
}

mlir::ValueRange CoshOp::getArgs()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

unsigned int CoshOp::getArgExpectedRank(unsigned int argIndex)
{
	return 0;
}

mlir::ValueRange CoshOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
{
	mlir::Type newResultType = resultType().cast<ArrayType>().slice(indexes.size());

	if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newResultType = arrayType.getElementType();

	mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

	if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newOperand = builder.create<LoadOp>(getLoc(), newOperand);

	auto op = builder.create<CoshOp>(getLoc(), newResultType, newOperand);
	return op->getResults();
}

mlir::ValueRange CoshOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	// D[cosh(x)] = x' * sinh(x)

	mlir::Location loc = getLoc();
	mlir::Value derivedOperand = derivatives.lookup(operand());
	mlir::Type type = convertToRealType(resultType());

	mlir::Value sinh = builder.create<SinhOp>(loc, type, operand());
	auto derivedOp = builder.create<MulElementWiseOp>(loc, type, sinh, derivedOperand);

	return derivedOp->getResults();
}

void CoshOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(operand());
}

void CoshOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

void CoshOp::foldConstants(mlir::OpBuilder& builder)
{
	if (!isOperandFoldable(operand()))
		return;

	ConstantOp operandOp = mlir::cast<ConstantOp>(operand().getDefiningOp());

	double operand = getAttributeValue(operandOp.value());

	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	mlir::Value newOp = builder.create<ConstantOp>(getLoc(), RealAttribute::get(getContext(), cosh(operand)));

	replaceAllUsesWith(newOp);
	erase();
}

mlir::Type CoshOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value CoshOp::operand()
{
	return Adaptor(*this).operand();
}

//===----------------------------------------------------------------------===//
// Modelica::TanhOp
//===----------------------------------------------------------------------===//

mlir::Value TanhOpAdaptor::operand()
{
	return getValues()[0];
}

llvm::ArrayRef<llvm::StringRef> TanhOp::getAttributeNames()
{
	return {};
}

void TanhOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand)
{
	state.addTypes(resultType);
	state.addOperands(operand);
}

mlir::ParseResult TanhOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void TanhOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << operand() << " : " << operand().getType() << " -> " << resultType();
}

mlir::ValueRange TanhOp::getArgs()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

unsigned int TanhOp::getArgExpectedRank(unsigned int argIndex)
{
	return 0;
}

mlir::ValueRange TanhOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
{
	mlir::Type newResultType = resultType().cast<ArrayType>().slice(indexes.size());

	if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newResultType = arrayType.getElementType();

	mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

	if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newOperand = builder.create<LoadOp>(getLoc(), newOperand);

	auto op = builder.create<TanhOp>(getLoc(), newResultType, newOperand);
	return op->getResults();
}

mlir::ValueRange TanhOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	// D[tanh(x)] = x' / (cosh(x))^2

	mlir::Location loc = getLoc();
	mlir::Value derivedOperand = derivatives.lookup(operand());
	mlir::Type type = convertToRealType(resultType());

	mlir::Value cosh = builder.create<CoshOp>(loc, type, operand());
	mlir::Value pow = builder.create<MulElementWiseOp>(loc, type, cosh, cosh);
	auto derivedOp = builder.create<DivElementWiseOp>(loc, type, derivedOperand, pow);

	return derivedOp->getResults();
}

void TanhOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(operand());
}

void TanhOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

void TanhOp::foldConstants(mlir::OpBuilder& builder)
{
	if (!isOperandFoldable(operand()))
		return;

	ConstantOp operandOp = mlir::cast<ConstantOp>(operand().getDefiningOp());

	double operand = getAttributeValue(operandOp.value());

	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	mlir::Value newOp = builder.create<ConstantOp>(getLoc(), RealAttribute::get(getContext(), tanh(operand)));

	replaceAllUsesWith(newOp);
	erase();
}

mlir::Type TanhOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value TanhOp::operand()
{
	return Adaptor(*this).operand();
}

//===----------------------------------------------------------------------===//
// Modelica::ExpOp
//===----------------------------------------------------------------------===//

mlir::Value ExpOpAdaptor::exponent()
{
	return getValues()[0];
}

llvm::ArrayRef<llvm::StringRef> ExpOp::getAttributeNames()
{
	return {};
}

void ExpOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value exponent)
{
	state.addTypes(resultType);
	state.addOperands(exponent);
}

mlir::ParseResult ExpOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::OpAsmParser::OperandType exponent;
	mlir::Type exponentType;
	mlir::Type resultType;

	if (parser.parseOperand(exponent) ||
			parser.parseColon() ||
			parser.parseType(exponentType) ||
			parser.resolveOperand(exponent, exponentType, result.operands) ||
			parser.parseArrow() ||
			parser.parseType(resultType))
		return mlir::failure();

	result.addTypes(resultType);
	return mlir::success();
}

void ExpOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << exponent() << " : " << exponent().getType() << " -> " << resultType();
}

mlir::ValueRange ExpOp::getArgs()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

unsigned int ExpOp::getArgExpectedRank(unsigned int argIndex)
{
	return 0;
}

mlir::ValueRange ExpOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
{
	mlir::Type newResultType = resultType().cast<ArrayType>().slice(indexes.size());

	if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newResultType = arrayType.getElementType();

	mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), exponent(), indexes);

	if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newOperand = builder.create<LoadOp>(getLoc(), newOperand);

	auto op = builder.create<ExpOp>(getLoc(), newResultType, newOperand);
	return op->getResults();
}

mlir::ValueRange ExpOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	// D[e^x] = x' * e^x

	mlir::Location loc = getLoc();
	mlir::Value derivedExponent = derivatives.lookup(exponent());
	mlir::Type type = convertToRealType(resultType());

	mlir::Value pow = builder.create<ExpOp>(loc, type, exponent());
	auto derivedOp = builder.create<MulElementWiseOp>(loc, type, pow, derivedExponent);

	return derivedOp->getResults();
}

void ExpOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(exponent());
}

void ExpOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

void ExpOp::foldConstants(mlir::OpBuilder& builder)
{
	if (!isOperandFoldable(exponent()))
		return;

	ConstantOp exponentOp = mlir::cast<ConstantOp>(exponent().getDefiningOp());

	double exponent = getAttributeValue(exponentOp.value());

	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	mlir::Value newOp = builder.create<ConstantOp>(getLoc(), RealAttribute::get(getContext(), exp(exponent)));

	replaceAllUsesWith(newOp);
	erase();
}

mlir::Type ExpOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value ExpOp::exponent()
{
	return Adaptor(*this).exponent();
}

//===----------------------------------------------------------------------===//
// Modelica::LogOp
//===----------------------------------------------------------------------===//

mlir::Value LogOpAdaptor::operand()
{
	return getValues()[0];
}

llvm::ArrayRef<llvm::StringRef> LogOp::getAttributeNames()
{
	return {};
}

void LogOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand)
{
	state.addTypes(resultType);
	state.addOperands(operand);
}

mlir::ParseResult LogOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void LogOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << operand() << " : " << operand().getType() << " -> " << resultType();
}

mlir::ValueRange LogOp::getArgs()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

unsigned int LogOp::getArgExpectedRank(unsigned int argIndex)
{
	return 0;
}

mlir::ValueRange LogOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
{
	mlir::Type newResultType = resultType().cast<ArrayType>().slice(indexes.size());

	if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newResultType = arrayType.getElementType();

	mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

	if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newOperand = builder.create<LoadOp>(getLoc(), newOperand);

	auto op = builder.create<LogOp>(getLoc(), newResultType, newOperand);
	return op->getResults();
}

mlir::ValueRange LogOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	// D[ln(x)] = x' / x

	mlir::Value derivedOperand = derivatives.lookup(operand());

	auto derivedOp = builder.create<DivElementWiseOp>(
			getLoc(), convertToRealType(resultType()), derivedOperand, operand());

	return derivedOp->getResults();
}

void LogOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(operand());
}

void LogOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

void LogOp::foldConstants(mlir::OpBuilder& builder)
{
	if (!isOperandFoldable(operand()))
		return;

	ConstantOp operandOp = mlir::cast<ConstantOp>(operand().getDefiningOp());

	double operand = getAttributeValue(operandOp.value());

	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	mlir::Value newOp = builder.create<ConstantOp>(getLoc(), RealAttribute::get(getContext(), log(operand)));

	replaceAllUsesWith(newOp);
	erase();
}

mlir::Type LogOp::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value LogOp::operand()
{
	return Adaptor(*this).operand();
}

//===----------------------------------------------------------------------===//
// Modelica::Log10Op
//===----------------------------------------------------------------------===//

mlir::Value Log10OpAdaptor::operand()
{
	return getValues()[0];
}

llvm::ArrayRef<llvm::StringRef> Log10Op::getAttributeNames()
{
	return {};
}

void Log10Op::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand)
{
	state.addTypes(resultType);
	state.addOperands(operand);
}

mlir::ParseResult Log10Op::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
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

void Log10Op::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << operand() << " : " << operand().getType() << " -> " << resultType();
}

mlir::ValueRange Log10Op::getArgs()
{
	return mlir::ValueRange(getOperation()->getOperands());
}

unsigned int Log10Op::getArgExpectedRank(unsigned int argIndex)
{
	return 0;
}

mlir::ValueRange Log10Op::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
{
	mlir::Type newResultType = resultType().cast<ArrayType>().slice(indexes.size());

	if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newResultType = arrayType.getElementType();

	mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

	if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
		newOperand = builder.create<LoadOp>(getLoc(), newOperand);

	auto op = builder.create<Log10Op>(getLoc(), newResultType, newOperand);
	return op->getResults();
}

mlir::ValueRange Log10Op::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
{
	// D[log10(x)] = x' / (x * ln(10))

	mlir::Location loc = getLoc();
	mlir::Value derivedOperand = derivatives.lookup(operand());
	mlir::Type type = convertToRealType(resultType());

	mlir::Value ten = builder.create<ConstantOp>(loc, RealAttribute::get(getContext(), 10));
	mlir::Value log = builder.create<LogOp>(loc, RealType::get(getContext()), ten);
	mlir::Value mul = builder.create<MulElementWiseOp>(loc, type, operand(), log);
	auto derivedOp = builder.create<DivElementWiseOp>(loc, resultType(), derivedOperand, mul);

	return derivedOp->getResults();
}

void Log10Op::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
{
	toBeDerived.push_back(operand());
}

void Log10Op::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
{

}

void Log10Op::foldConstants(mlir::OpBuilder& builder)
{
	if (!isOperandFoldable(operand()))
		return;

	ConstantOp operandOp = mlir::cast<ConstantOp>(operand().getDefiningOp());

	double operand = getAttributeValue(operandOp.value());

	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(*this);

	mlir::Value newOp = builder.create<ConstantOp>(getLoc(), RealAttribute::get(getContext(), log10(operand)));

	replaceAllUsesWith(newOp);
	erase();
}

mlir::Type Log10Op::resultType()
{
	return getOperation()->getResultTypes()[0];
}

mlir::Value Log10Op::operand()
{
	return Adaptor(*this).operand();
}

//===----------------------------------------------------------------------===//
// Modelica::NDimsOp
//===----------------------------------------------------------------------===//

mlir::Value NDimsOpAdaptor::memory()
{
	return getValues()[0];
}

llvm::ArrayRef<llvm::StringRef> NDimsOp::getAttributeNames()
{
	return {};
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

llvm::ArrayRef<llvm::StringRef> SizeOp::getAttributeNames()
{
	return {};
}

void SizeOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value memory, mlir::Value index)
{
	if (auto arrayType = resultType.dyn_cast<ArrayType>())
		resultType = arrayType.toAllocationScope(BufferAllocationScope::heap);

	state.addTypes(resultType);
	state.addOperands(memory);

	if (index != nullptr)
		state.addOperands(index);
}

void SizeOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << memory();

	if (hasIndex())
		printer << "[" << index() << "]";

	printer << " : " << resultType();
}

mlir::LogicalResult SizeOp::verify()
{
	if (!memory().getType().isa<ArrayType>())
		return emitOpError("requires the operand to be an array");

	return mlir::success();
}

void SizeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Read::get(), memory(), mlir::SideEffects::DefaultResource::get());
  populateAllocationEffects(effects, getResult());

  if (resultType().isa<ArrayType>())
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
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

llvm::ArrayRef<llvm::StringRef> IdentityOp::getAttributeNames()
{
	return {};
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

  if (!resultType().isa<ArrayType>())
    return emitOpError("requires the result to be an array");

	return mlir::success();
}

void IdentityOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
  populateAllocationEffects(effects, getResult());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
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

llvm::ArrayRef<llvm::StringRef> DiagonalOp::getAttributeNames()
{
	return {};
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
	if (!values().getType().isa<ArrayType>())
		return emitOpError("requires the values to be an array");

	if (auto arrayType = values().getType().cast<ArrayType>(); arrayType.getRank() != 1)
		return emitOpError("requires the values array to have rank 1");

  if (!resultType().isa<ArrayType>())
    return emitOpError("requires the result to be an array");

	return mlir::success();
}

void DiagonalOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
  effects.emplace_back(mlir::MemoryEffects::Read::get(), values(), mlir::SideEffects::DefaultResource::get());
  populateAllocationEffects(effects, getResult());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
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

llvm::ArrayRef<llvm::StringRef> ZerosOp::getAttributeNames()
{
	return {};
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
	if (!resultType().isa<ArrayType>())
		return emitOpError("requires the result to be an array");

	if (auto arrayType = resultType().cast<ArrayType>(); arrayType.getRank() != sizes().size())
		return emitOpError("requires the rank of the result array to match the sizes amount");

	return mlir::success();
}

void ZerosOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
  populateAllocationEffects(effects, getResult());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
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

llvm::ArrayRef<llvm::StringRef> OnesOp::getAttributeNames()
{
	return {};
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
	if (!resultType().isa<ArrayType>())
		return emitOpError("requires the result to be an array");

	if (auto arrayType = resultType().cast<ArrayType>(); arrayType.getRank() != sizes().size())
		return emitOpError("requires the rank of the result array to match the sizes amount");

	return mlir::success();
}

void OnesOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
  populateAllocationEffects(effects, getResult());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
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

llvm::ArrayRef<llvm::StringRef> LinspaceOp::getAttributeNames()
{
	return {};
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
	if (!resultType().isa<ArrayType>())
		return emitOpError("requires the result to be an array");

	if (auto arrayType = resultType().cast<ArrayType>(); arrayType.getRank() != 1)
		return emitOpError("requires the result array to have rank 1");

	return mlir::success();
}

void LinspaceOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
  populateAllocationEffects(effects, getResult());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
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

llvm::ArrayRef<llvm::StringRef> FillOp::getAttributeNames()
{
	return {};
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

llvm::ArrayRef<llvm::StringRef> MinOp::getAttributeNames()
{
	return {};
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
		if (auto arrayType = values()[0].getType().dyn_cast<ArrayType>();
				arrayType && isNumeric(arrayType.getElementType()))
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

llvm::ArrayRef<llvm::StringRef> MaxOp::getAttributeNames()
{
	return {};
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
		if (auto arrayType = values()[0].getType().dyn_cast<ArrayType>();
				arrayType && isNumeric(arrayType.getElementType()))
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

llvm::ArrayRef<llvm::StringRef> SumOp::getAttributeNames()
{
	return {};
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
	if (!array().getType().isa<ArrayType>())
		return emitOpError("requires the operand to be an array");

	if (!isNumeric(array().getType().cast<ArrayType>().getElementType()))
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

llvm::ArrayRef<llvm::StringRef> ProductOp::getAttributeNames()
{
	return {};
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
	if (!array().getType().isa<ArrayType>())
		return emitOpError("requires the operand to be an array");

	if (!isNumeric(array().getType().cast<ArrayType>().getElementType()))
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

llvm::ArrayRef<llvm::StringRef> TransposeOp::getAttributeNames()
{
	return {};
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
	if (!matrix().getType().isa<ArrayType>())
		return emitOpError("requires the source to be an array");

	auto sourceType = matrix().getType().cast<ArrayType>();

	if (sourceType.getRank() != 2)
		return emitOpError("requires the source to have rank 2");

	if (!resultType().isa<ArrayType>())
		return emitOpError("requires the result to be an array");

	auto destinationType = resultType().cast<ArrayType>();

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
  effects.emplace_back(mlir::MemoryEffects::Read::get(), matrix(), mlir::SideEffects::DefaultResource::get());
  populateAllocationEffects(effects, getResult());
  assert(getResult().getType().isa<ArrayType>());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
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

llvm::ArrayRef<llvm::StringRef> SymmetricOp::getAttributeNames()
{
	return {};
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
	if (!matrix().getType().isa<ArrayType>())
		return emitOpError("requires the source to be an array");

	auto sourceType = matrix().getType().cast<ArrayType>();

	if (sourceType.getRank() != 2)
		return emitOpError("requires the source to have rank 2");

	if (!resultType().isa<ArrayType>())
		return emitOpError("requires the result to be an array");

	auto destinationType = resultType().cast<ArrayType>();

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
  effects.emplace_back(mlir::MemoryEffects::Read::get(), matrix(), mlir::SideEffects::DefaultResource::get());
  populateAllocationEffects(effects, getResult());
  assert(getResult().getType().isa<ArrayType>());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
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

llvm::ArrayRef<llvm::StringRef> DerOp::getAttributeNames()
{
	return {};
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
// Modelica::DerSeedOp
//===----------------------------------------------------------------------===//

mlir::Value DerSeedOpAdaptor::member()
{
	return getValues()[0];
}

unsigned int DerSeedOpAdaptor::value()
{
	return getAttrs().getAs<mlir::IntegerAttr>("value").getInt();
}

llvm::ArrayRef<llvm::StringRef> DerSeedOp::getAttributeNames()
{
	static llvm::StringRef attrNames[] = {llvm::StringRef("value")};
	return llvm::makeArrayRef(attrNames);
}

void DerSeedOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value member, unsigned int value)
{
	state.addOperands(member);
	state.addAttribute("value", builder.getI32IntegerAttr(value));
}

void DerSeedOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << member() << ", " << value() << " : "
					<< member().getType();
}

mlir::Value DerSeedOp::member()
{
	return Adaptor(*this).member();
}

unsigned int DerSeedOp::value()
{
	return Adaptor(*this).value();
}

//===----------------------------------------------------------------------===//
// Modelica::PrintOp
//===----------------------------------------------------------------------===//

mlir::Value PrintOpAdaptor::value()
{
	return getValues()[0];
}

llvm::ArrayRef<llvm::StringRef> PrintOp::getAttributeNames()
{
	return {};
}

void PrintOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value value)
{
	state.addOperands(value);
}

mlir::ParseResult PrintOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
	mlir::OpAsmParser::OperandType operand;
	mlir::Type operandType;

	if (parser.parseOperand(operand) ||
			parser.parseColon() ||
			parser.parseType(operandType) ||
			parser.resolveOperand(operand, operandType, result.operands))
		return mlir::failure();

	return mlir::success();
}

void PrintOp::print(mlir::OpAsmPrinter& printer)
{
	printer << getOperationName() << " " << value() << ", " << value() << " : " << value().getType();
}

mlir::Value PrintOp::value()
{
	return Adaptor(*this).value();
}
