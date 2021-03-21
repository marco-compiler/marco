#include <mlir/Conversion/Passes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <modelica/mlirlowerer/Attribute.h>
#include <modelica/mlirlowerer/Ops.h>

using namespace modelica;

static bool isNumeric(mlir::Type type)
{
	return type.isa<mlir::IndexType, BooleanType, IntegerType, RealType>();
}

static bool isNumeric(mlir::Value value)
{
	return isNumeric(value.getType());
}

//===----------------------------------------------------------------------===//
// Modelica::ConstantOp
//===----------------------------------------------------------------------===//

llvm::StringRef ConstantOp::getOperationName()
{
	return "modelica.constant";
}

void ConstantOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Attribute attribute)
{
	state.addAttribute("value", attribute);
	state.addTypes(attribute.getType());
}

void ConstantOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.constant " << value();
}

mlir::OpFoldResult ConstantOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
{
	assert(operands.empty() && "constant has no operands");
	return value();
}

mlir::Attribute ConstantOp::value()
{
	return getOperation()->getAttr("value");
}

mlir::Type ConstantOp::getType()
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

mlir::LogicalResult CastOp::verify()
{
	mlir::Type inputType = value().getType();

	if (!inputType.isa<mlir::IndexType, BooleanType, IntegerType, RealType>())
		return emitOpError("requires the value to be an index, boolean, integer or real");

	if (!resultType().isa<mlir::IndexType, BooleanType, IntegerType, RealType>())
		return emitOpError("requires the result type to be index, boolean, integer or real");

	return mlir::success();
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

mlir::ValueRange CastCommonOpAdaptor::operands()
{
	return getValues();
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

		if (resultBaseType.isa<mlir::IndexType>() || baseType.isa<RealType>())
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
			types.emplace_back(PointerType::get(pointerType.getContext(), pointerType.getAllocationScope(), resultBaseType, shape));
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

llvm::StringRef AssignmentOp::getOperationName()
{
	return "modelica.assignment";
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

llvm::StringRef CallOp::getOperationName()
{
	return "modelica.call";
}

void CallOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::StringRef callee, mlir::TypeRange results, mlir::ValueRange args, unsigned int movedResults)
{
	state.addAttribute("callee", builder.getSymbolRefAttr(callee));
	state.addOperands(args);
	state.addTypes(results);
	state.addAttribute("movedResults", builder.getUI32IntegerAttr(movedResults));
}

void CallOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.call @" << callee() << "(" << args() << ")";
	auto resultTypes = getResultTypes();

	if (resultTypes.size() != 0)
		printer << " : " << getResultTypes();
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

	// The result arrays, which will be allocated by the callee on the heap,
	// must be seen as if they were allocated by the function call. This way,
	// the deallocation pass can free them.

	mlir::TypeRange types = getOperation()->getResultTypes();

	for (size_t i = 0, e = types.size(); i < e; ++i)
		if (auto pointerType = types[i].dyn_cast<PointerType>(); pointerType && pointerType.getAllocationScope() == heap)
			effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(i), mlir::SideEffects::DefaultResource::get());
}

mlir::CallInterfaceCallable CallOp::getCallableForCallee() {
	return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
}

mlir::Operation::operand_range CallOp::getArgOperands()
{
	return getOperands();
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
	mlir::IntegerAttr attr = getOperation()->getAttrOfType<mlir::IntegerAttr>("movedResults");
	return attr.getUInt();
}

//===----------------------------------------------------------------------===//
// Modelica::AllocaOp
//===----------------------------------------------------------------------===//

mlir::ValueRange AllocaOpAdaptor::dynamicDimensions()
{
	return getValues();
}

llvm::StringRef AllocaOp::getOperationName()
{
	return "modelica.alloca";
}

void AllocaOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type elementType, llvm::ArrayRef<long> shape, mlir::ValueRange dimensions)
{
	state.addTypes(PointerType::get(state.getContext(), BufferAllocationScope::stack, elementType, shape));
	state.addOperands(dimensions);
}

void AllocaOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.alloca ";
	printer.printOperands(getOperands());
	printer << ": ";
	printer.printType(getOperation()->getResultTypes()[0]);
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

PointerType AllocaOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<PointerType>();
}

mlir::ValueRange AllocaOp::dynamicDimensions()
{
	return Adaptor(*this).dynamicDimensions();
}

//===----------------------------------------------------------------------===//
// Modelica::AllocOp
//===----------------------------------------------------------------------===//

mlir::ValueRange AllocOpAdaptor::dynamicDimensions()
{
	return getValues();
}

llvm::StringRef AllocOp::getOperationName()
{
	return "modelica.alloc";
}

void AllocOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type elementType, llvm::ArrayRef<long> shape, mlir::ValueRange dimensions)
{
	state.addTypes(PointerType::get(state.getContext(), BufferAllocationScope::heap, elementType, shape));
	state.addOperands(dimensions);
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

PointerType AllocOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<PointerType>();
}

mlir::ValueRange AllocOp::dynamicDimensions()
{
	return Adaptor(*this).dynamicDimensions();
}

//===----------------------------------------------------------------------===//
// Modelica::FreeOp
//===----------------------------------------------------------------------===//

mlir::Value FreeOpAdaptor::memory()
{
	return getValues()[0];
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

mlir::LogicalResult FreeOp::verify()
{
	if (auto pointerType = memory().getType().dyn_cast<PointerType>(); pointerType)
	{
		if (pointerType.getAllocationScope() == heap)
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

llvm::StringRef PtrCastOp::getOperationName()
{
	return "modelica.ptr_cast";
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
	if (!memory().getType().isa<PointerType>())
		return emitOpError("requires operand to be a pointer");

	auto pointerType = memory().getType().cast<PointerType>();

	if (!resultType().isa<PointerType>())
		return emitOpError("requires the result type to be a pointer");

	if (pointerType.getElementType() != resultType().cast<PointerType>().getElementType())
		return emitOpError("requires the result pointer type to have the same element type of the operand");

	if (pointerType.getRank() != resultType().cast<PointerType>().getRank())
		return emitOpError("requires the result pointer type to have the same rank as the operand");

	return mlir::success();
}

mlir::Value PtrCastOp::memory()
{
	return Adaptor(*this).memory();
}

PointerType PtrCastOp::resultType()
{
	return getOperation()->getResultTypes()[0].cast<PointerType>();
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

mlir::LogicalResult DimOp::verify()
{
	if (!memory().getType().isa<PointerType>())
		return emitOpError("requires the operand to be a pointer to an array");

	return mlir::success();
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

llvm::StringRef SubscriptionOp::getOperationName()
{
	return "modelica.subscription";
}

void SubscriptionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value source, mlir::ValueRange indexes)
{
	state.addOperands(source);
	state.addOperands(indexes);

	auto sourcePointerType = source.getType().cast<PointerType>();
	auto shape = sourcePointerType.getShape();

	assert(indexes.size() <= shape.size() && "Too many subscription indexes");
	llvm::SmallVector<long, 3> resultShape;

	for (size_t i = indexes.size(), e = shape.size(); i < e; ++i)
		resultShape.push_back(shape[i]);

	mlir::Type resultType = PointerType::get(builder.getContext(), sourcePointerType.getAllocationScope(), sourcePointerType.getElementType(), resultShape);
	state.addTypes(resultType);
}

void SubscriptionOp::print(mlir::OpAsmPrinter& printer)
{
 	printer << "modelica.subscript " << source() << "[" << indexes() << "] : " << resultType();
}

mlir::Value SubscriptionOp::getViewSource()
{
	return source();
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
	printer.printType(getOperation()->getResultTypes()[0]);
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

llvm::StringRef ArrayCloneOp::getOperationName()
{
	return "modelica.array_clone";
}

void ArrayCloneOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value source, bool heap)
{
	state.addOperands(source);
	auto sourceType = source.getType().cast<PointerType>();
	BufferAllocationScope allocationScope = heap ? BufferAllocationScope::heap : BufferAllocationScope::stack;
	state.addTypes(PointerType::get(builder.getContext(), allocationScope, sourceType.getElementType(), sourceType.getShape()));
}

void ArrayCloneOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.array_clone " << source() << " : " << getPointerType();
}

void ArrayCloneOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	effects.emplace_back(mlir::MemoryEffects::Read::get(), source(), mlir::SideEffects::DefaultResource::get());
	effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
}

PointerType ArrayCloneOp::getPointerType()
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

mlir::Value ForOpAdaptor::breakCondition()
{
	return getValues()[0];
}

mlir::Value ForOpAdaptor::returnCondition()
{
	return getValues()[1];
}

mlir::ValueRange ForOpAdaptor::args()
{
	return mlir::ValueRange(std::next(getValues().begin(), 2), getValues().end());
}

llvm::StringRef ForOp::getOperationName()
{
	return "modelica.for";
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

mlir::LogicalResult ForOp::verify()
{
	if (auto breakPtr = breakCondition().getType().dyn_cast<PointerType>();
			!breakPtr || !breakPtr.getElementType().isa<BooleanType>() || breakPtr.getRank() != 0)
		return emitOpError("requires the break condition to be a pointer to a single boolean value");

	if (auto returnPtr = breakCondition().getType().dyn_cast<PointerType>();
			!returnPtr || !returnPtr.getElementType().isa<BooleanType>() || returnPtr.getRank() != 0)
		return emitOpError("requires the return condition to be a pointer to a single boolean value");

	return mlir::success();
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

mlir::Value WhileOpAdaptor::breakCondition()
{
	return getValues()[0];
}

mlir::Value WhileOpAdaptor::returnCondition()
{
	return getValues()[1];
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

mlir::LogicalResult WhileOp::verify()
{
	if (auto breakPtr = breakCondition().getType().dyn_cast<PointerType>();
			!breakPtr || !breakPtr.getElementType().isa<BooleanType>() || breakPtr.getRank() != 0)
		return emitOpError("requires the break condition to be a pointer to a single boolean value");

	if (auto returnPtr = breakCondition().getType().dyn_cast<PointerType>();
			!returnPtr || !returnPtr.getElementType().isa<BooleanType>() || returnPtr.getRank() != 0)
		return emitOpError("requires the return condition to be a pointer to a single boolean value");

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

mlir::Value ConditionOpAdaptor::condition()
{
	return getValues()[0];
}

mlir::ValueRange ConditionOpAdaptor::args()
{
	return mlir::ValueRange(std::next(getValues().begin()), getValues().end());
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

mlir::ValueRange YieldOpAdaptor::args()
{
	return getValues();
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
// Modelica::NotOp
//===----------------------------------------------------------------------===//

mlir::Value NotOpAdaptor::operand()
{
	return getValues()[0];
}

llvm::StringRef NotOp::getOperationName()
{
	return "modelica.not";
}

void NotOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value operand)
{
	state.addOperands(operand);
	state.addTypes(operand.getType());
}

void NotOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.not " << getOperand() << " : " << getOperation()->getResultTypes();
}

mlir::LogicalResult NotOp::verify()
{
	if (!operand().getType().isa<BooleanType>())
		if (auto pointerType = operand().getType().dyn_cast<PointerType>(); !pointerType || !pointerType.getElementType().isa<BooleanType>())
			return emitOpError("requires the operand to be a boolean or an array of booleans");

	return mlir::success();
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

llvm::StringRef AndOp::getOperationName()
{
	return "modelica.and";
}

void AndOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

void AndOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.and " << lhs() << ", " << rhs() << " : " << resultType();
}

mlir::LogicalResult AndOp::verify()
{
	if (!lhs().getType().isa<BooleanType>() || !rhs().getType().isa<BooleanType>())
		return emitOpError("requires the operands to be booleans");

	return mlir::success();
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

llvm::StringRef OrOp::getOperationName()
{
	return "modelica.or";
}

void OrOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

void OrOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.or " << lhs() << ", " << rhs() << " : " << resultType();
}

mlir::LogicalResult OrOp::verify()
{
	if (!lhs().getType().isa<BooleanType>() || !rhs().getType().isa<BooleanType>())
		return emitOpError("requires the operands to be booleans");

	return mlir::success();
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

llvm::StringRef EqOp::getOperationName()
{
	return "modelica.eq";
}

void EqOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

void EqOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.eq " << getOperands() << " : " << getOperation()->getResultTypes()[0];
}

mlir::LogicalResult EqOp::verify()
{
	if (!isNumeric(lhs()) || !isNumeric(rhs()))
		return emitOpError("Comparison operation are only defined for scalar operands of simple types");

	return mlir::success();
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

void NotEqOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.neq " << getOperands() << " : " << getOperation()->getResultTypes()[0];
}

mlir::LogicalResult NotEqOp::verify()
{
	if (!isNumeric(lhs()) || !isNumeric(rhs()))
		return emitOpError("Comparison operation are only defined for scalar operands of simple types");

	return mlir::success();
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

void GtOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.gt " << getOperands() << " : " << getOperation()->getResultTypes()[0];
}

mlir::LogicalResult GtOp::verify()
{
	if (!isNumeric(lhs()) || !isNumeric(rhs()))
		return emitOpError("Comparison operation are only defined for scalar operands of simple types");

	return mlir::success();
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

void GteOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.gte " << getOperands() << " : " << getOperation()->getResultTypes()[0];
}

mlir::LogicalResult GteOp::verify()
{
	if (!isNumeric(lhs()) || !isNumeric(rhs()))
		return emitOpError("Comparison operation are only defined for scalar operands of simple types");

	return mlir::success();
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

void LtOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.lt " << getOperands() << " : " << getOperation()->getResultTypes()[0];
}

mlir::LogicalResult LtOp::verify()
{
	if (!isNumeric(lhs()) || !isNumeric(rhs()))
		return emitOpError("Comparison operation are only defined for scalar operands of simple types");

	return mlir::success();
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

void LteOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.lte " << getOperands() << " : " << getOperation()->getResultTypes()[0];
}

mlir::LogicalResult LteOp::verify()
{
	if (!isNumeric(lhs()) || !isNumeric(rhs()))
		return emitOpError("Comparison operation are only defined for scalar operands of simple types");

	return mlir::success();
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

llvm::StringRef NegateOp::getOperationName()
{
	return "modelica.neg";
}

void NegateOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value value)
{
	state.addTypes(resultType);
	state.addOperands(value);
}

void NegateOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.neg " << operand() << " : " << resultType();
}

void NegateOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (resultType().isa<PointerType>())
	{
		effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());
		effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());
		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
	}
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

void AddOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (resultType().isa<PointerType>())
	{
		effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());
		effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());
		effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());
		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
	}
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

llvm::StringRef SubOp::getOperationName()
{
	return "modelica.sub";
}

void SubOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

void SubOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.sub " << lhs() << ", " << rhs() << " : " << resultType();
}

void SubOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (resultType().isa<PointerType>())
	{
		effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());
		effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());
		effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());
		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
	}
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

llvm::StringRef MulOp::getOperationName()
{
	return "modelica.mul";
}

void MulOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

void MulOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.mul " << lhs() << ", " << rhs() << " : " << resultType();
}

void MulOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (lhs().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());

	if (rhs().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());

	if (resultType().isa<PointerType>())
	{
		effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());
		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
	}
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

llvm::StringRef DivOp::getOperationName()
{
	return "modelica.div";
}

void DivOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
{
	state.addTypes(resultType);
	state.addOperands({ lhs, rhs });
}

void DivOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.div " << lhs() << ", " << rhs() << " : " << resultType();
}

void DivOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (lhs().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());

	if (rhs().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());

	if (resultType().isa<PointerType>())
	{
		effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());
		effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
	}
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

mlir::Value PowOpAdaptor::base()
{
	return getValues()[0];
}

mlir::Value PowOpAdaptor::exponent()
{
	return getValues()[1];
}

llvm::StringRef PowOp::getOperationName()
{
	return "modelica.pow";
}

void PowOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value base, mlir::Value exponent)
{
	state.addTypes(resultType);
	state.addOperands({ base, exponent });
}

void PowOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.pow " << base() << ", " << exponent() << " : " << resultType();
}

void PowOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
{
	if (base().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), base(), mlir::SideEffects::DefaultResource::get());

	if (exponent().getType().isa<PointerType>())
		effects.emplace_back(mlir::MemoryEffects::Read::get(), exponent(), mlir::SideEffects::DefaultResource::get());

	if (resultType().isa<PointerType>())
	{
		effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());
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

llvm::StringRef NDimsOp::getOperationName()
{
	return "modelica.ndims";
}

void NDimsOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value memory)
{
	state.addTypes(resultType);
	state.addOperands(memory);
}

void NDimsOp::print(mlir::OpAsmPrinter& printer)
{
	printer << "modelica.ndims " << memory() << " : " << resultType();
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

llvm::StringRef SizeOp::getOperationName()
{
	return "modelica.size";
}

void SizeOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value memory, mlir::Value index)
{
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
