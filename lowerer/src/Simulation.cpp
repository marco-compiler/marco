#include "modelica/lowerer/Simulation.hpp"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IRBuilder.h"
#include "modelica/lowerer/SimLowerer.hpp"

using namespace std;
using namespace modelica;
using namespace llvm;

static Value* getArrayElementPtr(
		IRBuilder<>& bld, Value* arrayPtr, size_t index)
{
	auto ptrType = dyn_cast<PointerType>(arrayPtr->getType());
	auto arrayType = dyn_cast<ArrayType>(ptrType->getContainedType(0));
	assert(index <= arrayType->getNumElements());	// NOLINT
	auto baseType = arrayType->getContainedType(0);
	auto ptrToBaseType = baseType->getPointerTo(0);

	auto intType = Type::getInt32Ty(bld.getContext());
	auto i = ConstantInt::get(intType, index);
	auto element = bld.CreateGEP(arrayPtr, i);

	return bld.CreateBitCast(element, ptrToBaseType);
}

static void storeToArrayElement(
		IRBuilder<>& bld, Value* value, Value* arrayPtr, size_t index)
{
	auto ptrToElem = getArrayElementPtr(bld, arrayPtr, index);
	bld.CreateStore(value, ptrToElem);
}

template<typename T>
void storeConstantToArrayElement(
		IRBuilder<>& bld, T value, Value* arrayPtr, size_t index)
{
	auto ptrToElem = getArrayElementPtr(bld, arrayPtr, index);
	makeConstantStore<T>(bld, value, ptrToElem);
}

static Value* loadToArrayElement(
		IRBuilder<>& bld, Value* arrayPtr, size_t index)
{
	auto ptrToElem = getArrayElementPtr(bld, arrayPtr, index);
	return bld.CreateLoad(ptrToElem);
}

template<typename T>
static Expected<AllocaInst*> lowerConstant(
		IRBuilder<> builder, const SimConst<T>& constant, const SimType& type)
{
	if (constant.size() != type.flatSize())
		return make_error<TypeConstantSizeMissMatch>(constant, type);

	auto llvmType = typeToLLVMType(builder.getContext(), type);
	auto alloca = builder.CreateAlloca(llvmType);

	for (size_t i = 0; i < constant.size(); i++)
		storeConstantToArrayElement<T>(builder, constant.get(i), alloca, i);
	return alloca;
}

static Expected<AllocaInst*> lowerConstant(
		IRBuilder<>& builder, const SimExp& exp)
{
	if (exp.isConstant<int>())
		return lowerConstant<int>(
				builder, exp.getConstant<int>(), exp.getSimType());
	if (exp.isConstant<float>())
		return lowerConstant<float>(
				builder, exp.getConstant<float>(), exp.getSimType());
	if (exp.isConstant<bool>())
		return lowerConstant<bool>(
				builder, exp.getConstant<bool>(), exp.getSimType());

	assert(false && "unreachable");	// NOLINT
	return nullptr;
}

template<int argumentsSize>
static AllocaInst* elementWiseOperation(
		IRBuilder<>& builder,
		ArrayRef<Value*>& args,
		std::function<Value*(IRBuilder<>&, ArrayRef<Value*>)> operation)
{
	auto ptrType = dyn_cast<PointerType>(args[0]->getType());
	auto arrayType = dyn_cast<ArrayType>(ptrType->getContainedType(0));

	auto alloca = builder.CreateAlloca(arrayType);

	for (size_t a = 0; a < arrayType->getNumElements(); a++)
	{
		SmallVector<Value*, argumentsSize> arguments;

		for (int i = 0; i < argumentsSize; i++)
			arguments.push_back(loadToArrayElement(builder, args[i], a));

		auto outVal = operation(builder, arguments);

		storeToArrayElement(builder, outVal, alloca, a);
	}
	return alloca;
}

static Value* createSingleBynaryOP(
		IRBuilder<>& builder, ArrayRef<Value*> args, SimExpKind kind)
{
	switch (kind)
	{
		case SimExpKind::add:
			return builder.CreateAdd(args[0], args[1]);
		case SimExpKind::sub:
			return builder.CreateSub(args[0], args[1]);
		case SimExpKind::mult:
			return builder.CreateMul(args[0], args[1]);
		case SimExpKind::divide:
		{
			if (!args[0]->getType()->isFloatTy())
				return builder.CreateSDiv(args[0], args[1]);
			return builder.CreateFDiv(args[0], args[1]);
		}
		case SimExpKind::equal:
			return builder.CreateICmpEQ(args[0], args[1]);
		case SimExpKind::different:
			return builder.CreateICmpNE(args[0], args[1]);
		case SimExpKind::greaterEqual:
			return builder.CreateICmpSGE(args[0], args[1]);
		case SimExpKind::greaterThan:
			return builder.CreateICmpSGT(args[0], args[1]);
		case SimExpKind::lessEqual:
			return builder.CreateICmpSLE(args[0], args[1]);
		case SimExpKind::less:
			return builder.CreateICmpSLT(args[0], args[1]);
		case SimExpKind::module:
		{
			// TODO
			assert(false && "module unsupported");	// NOLINT
			return nullptr;
		}
		case SimExpKind::elevation:
		{
			// TODO
			assert(false && "powerof unsupported");	// NOLINT
			return nullptr;
		}
		default:
			assert(false && "unreachable");	// NOLINT
			return nullptr;
	}
	assert(false && "unreachable");	// NOLINT
	return nullptr;
}

static Value* createSingleUnaryOp(
		IRBuilder<>& builder, ArrayRef<Value*> args, SimExpKind kind)
{
	switch (kind)
	{
		case SimExpKind::negate:
			return builder.CreateNeg(args[0]);
		default:
			assert(false && "unreachable");	// NOLINT
			return nullptr;
	}
	assert(false && "unreachable");	// NOLINT
	return nullptr;
}

static Expected<AllocaInst*> lowerBinOp(
		IRBuilder<>& builder, const SimExp& exp, ArrayRef<Value*> subExp)
{
	assert(exp.isBinary());			 // NOLINT
	assert(subExp.size() == 2);	// NOLINT
	const auto binaryOp = [type = exp.getKind()](auto& builder, auto args) {
		return createSingleBynaryOP(builder, args, type);
	};

	return elementWiseOperation<2>(builder, subExp, binaryOp);
}

static Expected<AllocaInst*> lowerUnOp(
		IRBuilder<>& builder, const SimExp& exp, ArrayRef<Value*> subExp)
{
	assert(exp.isUnary());			 // NOLINT
	assert(subExp.size() == 1);	// NOLINT

	const auto unaryOp = [type = exp.getKind()](auto& builder, auto args) {
		return createSingleUnaryOp(builder, args, type);
	};

	return elementWiseOperation<1>(builder, subExp, unaryOp);
}

static Expected<AllocaInst*> lowerTernOp(
		IRBuilder<>& builder, const SimExp& exp, ArrayRef<Value*> subExp)
{
	assert(exp.isTernary());		 // NOLINT
	assert(subExp.size() == 3);	// NOLINT

	switch (exp.getKind())
	{
		case SimExpKind::zero:
		{
			SimType type(BultinSimTypes::INT);
			IntSimConst constant(0);
			return lowerConstant(builder, constant, type);
		}
		case SimExpKind::conditional:
		{
			assert(subExp.size() == 3);	// NOLINT
			// TODO
			assert(false && "conditional unsupported");	// NOLINT
			return nullptr;
		}
		default:
			assert(false && "unreachable");	// NOLINT
			return nullptr;
	}
	assert(false && "unreachable");	// NOLINT
	return nullptr;
}

static Expected<AllocaInst*> lowerOperation(
		IRBuilder<>& builder, const SimExp& exp, ArrayRef<Value*> subExp)
{
	assert(exp.isOperation());	// NOLINT
	if (exp.isUnary())
		return lowerUnOp(builder, exp, subExp);

	if (exp.isBinary())
		return lowerBinOp(builder, exp, subExp);

	return lowerTernOp(builder, exp, subExp);
}

static Expected<Value*> lowerExp(IRBuilder<>& builder, const SimExp& exp)
{
	SmallVector<Value*, 3> values;

	if (!exp.areSubExpressionCompatibles())
		return make_error<TypeMissMatch>(exp);

	if (exp.isConstant())
		return lowerConstant(builder, exp);

	if (exp.isReference())
		return lowerReference(builder, exp.getReference());

	if (exp.isUnary() || exp.isBinary() || exp.isTernary())
	{
		auto subexp = lowerExp(builder, exp.getLeftHand());
		if (!subexp)
			return subexp;
		values.push_back(move(*subexp));
	}
	if (exp.isBinary() || exp.isTernary())
	{
		auto subexp = lowerExp(builder, exp.getRightHand());
		if (!subexp)
			return subexp;
		values.push_back(move(*subexp));
	}
	if (exp.isTernary())
	{
		auto subexp = lowerExp(builder, exp.getCondition());
		if (!subexp)
			return subexp;
		values.push_back(move(*subexp));
	}

	return lowerOperation(builder, exp, values);
}

static Expected<Function*> populateMain(
		Module& m,
		StringRef entryPointName,
		Function* init,
		Function* update,
		Function* printValues,
		unsigned simulationStop)
{
	assert(init != nullptr);				 // NOLINT
	assert(update != nullptr);			 // NOLINT
	assert(printValues != nullptr);	// NOLINT

	auto expectedMain = makePrivateFunction(entryPointName, m);
	if (!expectedMain)
		return expectedMain;
	auto main = expectedMain.get();
	main->setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);

	IRBuilder<> builder(&main->getEntryBlock());
	// call init
	builder.CreateCall(init);

	const auto forBody = [update, printValues](IRBuilder<>& builder) {
		builder.CreateCall(update);
		builder.CreateCall(printValues);
	};

	// creates a for with simulationStop iterations what invokes
	// update and print values each time
	auto loopExit =
			createForCycle(*main, main->getEntryBlock(), simulationStop, forBody);

	// returns right after the loop
	builder.SetInsertPoint(loopExit);
	builder.CreateRet(nullptr);
	return main;
}

static void insertGlobalString(Module& m, StringRef name, StringRef content)
{
	auto str = ConstantDataArray::getString(m.getContext(), content);
	auto type = ArrayType::get(
			IntegerType::getInt8Ty(m.getContext()), content.size() + 1);
	auto global = m.getOrInsertGlobal(name, type);
	dyn_cast<GlobalVariable>(global)->setInitializer(str);
}

static Error insertGlobal(
		Module& module,
		StringRef name,
		const SimExp& exp,
		GlobalValue::LinkageTypes linkage)
{
	const auto& type = exp.getSimType();
	const string oldName = name.str() + "_old";

	if (auto e = simExpToGlobalVar(module, name, type, linkage); e)
		return e;
	if (auto e = simExpToGlobalVar(module, oldName, type, linkage); e)
		return e;
	insertGlobalString(module, name.str() + "_str", name);
	return Error::success();
}

static Error createAllGlobals(
		Module& m, StringMap<SimExp> vars, GlobalValue::LinkageTypes linkType)
{
	for (const auto& pair : vars)
		if (auto e = insertGlobal(m, pair.first(), pair.second, linkType); e)
			return e;

	return Error::success();
}

static Expected<Function*> initializeGlobals(Module& m, StringMap<SimExp> vars)
{
	auto initFunctionExpected = makePrivateFunction("init", m);
	if (!initFunctionExpected)
		return initFunctionExpected;
	auto initFunction = initFunctionExpected.get();
	IRBuilder builder(&initFunction->getEntryBlock());

	for (const auto& pair : vars)
	{
		auto val = lowerExp(builder, pair.second);
		if (!val)
			return val.takeError();
		auto loaded = builder.CreateLoad(*val);
		builder.CreateStore(loaded, m.getGlobalVariable(pair.first(), true));
		builder.CreateStore(
				loaded, m.getGlobalVariable(pair.first().str() + "_old", true));
	}
	builder.CreateRet(nullptr);
	return initFunction;
}

static Expected<Function*> createUpdates(Module& m, StringMap<SimExp> upds)
{
	auto updateFunctionExpected = makePrivateFunction("update", m);
	if (!updateFunctionExpected)
		return updateFunctionExpected;
	auto updateFunction = updateFunctionExpected.get();
	IRBuilder bld(&updateFunction->getEntryBlock());

	for (const auto& pair : upds)
	{
		auto val = lowerExp(bld, pair.second);
		if (!val)
			return val.takeError();
		auto loaded = bld.CreateLoad(*val);
		bld.CreateStore(loaded, m.getGlobalVariable(pair.first(), true));
	}
	for (const auto& pair : upds)
	{
		auto globalVal = m.getGlobalVariable(pair.first(), true);
		auto val = bld.CreateLoad(globalVal);

		bld.CreateStore(
				val, m.getGlobalVariable(pair.first().str() + "_old", true));
	}

	bld.CreateRet(nullptr);
	return updateFunction;
}

static Expected<Function*> populatePrint(Module& m, StringMap<SimExp> vars)
{
	auto printFunctionExpected = makePrivateFunction("print", m);
	if (!printFunctionExpected)
		return printFunctionExpected;
	auto printFunction = printFunctionExpected.get();
	IRBuilder bld(&printFunction->getEntryBlock());

	auto charType = IntegerType::getInt8Ty(m.getContext());
	auto floatType = IntegerType::getFloatTy(m.getContext());
	auto charPtrType = PointerType::get(charType, 0);
	SmallVector<Type*, 2> args({ charPtrType, floatType });
	auto printType =
			FunctionType::get(Type::getVoidTy(m.getContext()), args, false);
	auto externalPrint =
			dyn_cast<Function>(m.getOrInsertFunction("modelicaPrint", printType));

	for (const auto& pair : vars)
	{
		auto strName = m.getGlobalVariable(pair.first().str() + "_str");
		auto charStar = bld.CreateBitCast(strName, charPtrType);
		auto variable = m.getGlobalVariable(pair.first(), true);

		for (size_t a = 0; a < pair.second.getSimType().flatSize(); a++)
		{
			auto value = loadToArrayElement(bld, variable, a);
			auto casted = bld.CreateCast(Instruction::SIToFP, value, floatType);
			SmallVector<Value*, 2> args({ charStar, casted });
			bld.CreateCall(externalPrint, args);
		}
	}
	bld.CreateRet(nullptr);
	return printFunction;
}

Error Simulation::lower()
{
	if (auto e = createAllGlobals(module, variables, getVarLinkage()); e)
		return e;

	auto initFunction = initializeGlobals(module, variables);
	if (!initFunction)
		return initFunction.takeError();

	auto updateFunction = createUpdates(module, updates);
	if (!updateFunction)
		return updateFunction.takeError();

	auto printFunction = populatePrint(module, variables);
	if (!printFunction)
		return printFunction.takeError();

	auto e = populateMain(
			module,
			entryPointName,
			*initFunction,
			*updateFunction,
			*printFunction,
			stopTime);
	if (!e)
		return e.takeError();

	return Error::success();
}

void Simulation::dump(raw_ostream& OS) const
{
	auto const dumpEquation = [&OS](const auto& couple) {
		OS << couple.first().data();
		OS << " = ";
		couple.second.dump(OS);
		OS << "\n";
	};

	OS << "Init:\n";
	for_each(begin(variables), end(variables), dumpEquation);

	OS << "Update:\n";
	for_each(begin(updates), end(updates), dumpEquation);
}

void Simulation::dumpBC(raw_ostream& OS) const
{
	WriteBitcodeToFile(module, OS);
}
