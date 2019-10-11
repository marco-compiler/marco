#include "modelica/lowerer/Simulation.hpp"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IRBuilder.h"
#include "modelica/lowerer/SimLowerer.hpp"

using namespace std;
using namespace modelica;
using namespace llvm;

static Value* getArrayElementPtr(
		IRBuilder<>& bld, Value* arrayPtr, Value* index)
{
	auto intType = Type::getInt32Ty(bld.getContext());

	auto zero = ConstantInt::get(intType, 0);
	SmallVector<Value*, 2> args = { zero, index };
	return bld.CreateGEP(arrayPtr, args);
}

static Value* getArrayElementPtr(
		IRBuilder<>& bld, Value* arrayPtr, size_t index)
{
	auto ptrType = dyn_cast<PointerType>(arrayPtr->getType());
	auto arrayType = dyn_cast<ArrayType>(ptrType->getContainedType(0));
	assert(index <= arrayType->getNumElements());	 // NOLINT

	auto intType = Type::getInt32Ty(bld.getContext());

	auto zero = ConstantInt::get(intType, 0);
	auto i = ConstantInt::get(intType, index);
	return getArrayElementPtr(bld, arrayPtr, i);
}

static void storeToArrayElement(
		IRBuilder<>& bld, Value* value, Value* arrayPtr, size_t index)
{
	auto ptrToElem = getArrayElementPtr(bld, arrayPtr, index);
	bld.CreateStore(value, ptrToElem);
}

static void storeToArrayElement(
		IRBuilder<>& bld, Value* value, Value* arrayPtr, Value* index)
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

static Value* loadToArrayElement(
		IRBuilder<>& bld, Value* arrayPtr, Value* index)
{
	auto ptrToElem = getArrayElementPtr(bld, arrayPtr, index);
	return bld.CreateLoad(ptrToElem);
}

static AllocaInst* allocaSimType(IRBuilder<>& bld, const SimType& type)
{
	auto llvmType = typeToLLVMType(bld.getContext(), type);
	return bld.CreateAlloca(llvmType);
}

template<typename T>
static Expected<AllocaInst*> lowerConstant(
		IRBuilder<> builder, const SimConst<T>& constant, const SimType& type)
{
	if (constant.size() != type.flatSize())
		return make_error<TypeConstantSizeMissMatch>(constant, type);

	auto alloca = allocaSimType(builder, type);

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

	assert(false && "unreachable");	 // NOLINT
	return nullptr;
}

static Type* typeFromOperation(
		LLVMContext& context, SimExpKind op, Type* leftHand)
{
	assert(leftHand->isArrayTy());	// NOLINT
	auto t = dyn_cast<ArrayType>(leftHand);
	auto boolType = IntegerType::getInt1Ty(context);
	auto boolVector = ArrayType::get(boolType, t->getNumElements());
	switch (op)
	{
		case (SimExpKind::negate):
		case (SimExpKind::add):
		case (SimExpKind::sub):
		case (SimExpKind::mult):
		case (SimExpKind::divide):
		case (SimExpKind::elevation):
		case (SimExpKind::module):
		case (SimExpKind::conditional):
			return leftHand;
		case (SimExpKind::greaterThan):
		case (SimExpKind::greaterEqual):
		case (SimExpKind::equal):
		case (SimExpKind::different):
		case (SimExpKind::less):
		case (SimExpKind::lessEqual):
			return boolVector;
		default:
			assert(false && "unreachable");	 // NOLINT
			return nullptr;
	}
}

template<int argumentsSize>
static AllocaInst* elementWiseOperation(
		IRBuilder<>& builder,
		Function* fun,
		ArrayRef<Value*>& args,
		SimExpKind opKind,
		std::function<Value*(IRBuilder<>&, ArrayRef<Value*>)> operation)
{
	auto ptrType = dyn_cast<PointerType>(args[0]->getType());
	auto arrayType = dyn_cast<ArrayType>(ptrType->getContainedType(0));

	auto alloca = builder.CreateAlloca(
			typeFromOperation(builder.getContext(), opKind, arrayType));

	const auto performOp = [alloca, &operation, &args](
														 IRBuilder<>& builder, Value* index) {
		SmallVector<Value*, argumentsSize> arguments;

		for (int i = 0; i < argumentsSize; i++)
			arguments.push_back(loadToArrayElement(builder, args[i], index));
		auto outVal = operation(builder, arguments);

		storeToArrayElement(builder, outVal, alloca, index);
	};

	createForCycle(fun, builder, arrayType->getNumElements(), performOp);

	return alloca;
}

static Value* createSingleBynaryOP(
		IRBuilder<>& builder, ArrayRef<Value*> args, SimExpKind kind)
{
	bool leftHandIsFloat = args[0]->getType()->isFloatTy();
	switch (kind)
	{
		case SimExpKind::add:
			if (leftHandIsFloat)
				return builder.CreateFAdd(args[0], args[1]);
			return builder.CreateAdd(args[0], args[1]);
		case SimExpKind::sub:
			if (leftHandIsFloat)
				return builder.CreateFSub(args[0], args[1]);
			return builder.CreateSub(args[0], args[1]);
		case SimExpKind::mult:
			if (leftHandIsFloat)
				return builder.CreateFMul(args[0], args[1]);
			return builder.CreateMul(args[0], args[1]);
		case SimExpKind::divide:
			if (leftHandIsFloat)
				return builder.CreateFDiv(args[0], args[1]);
			return builder.CreateSDiv(args[0], args[1]);
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
		case SimExpKind::module: {
			// TODO
			assert(false && "module unsupported");	// NOLINT
			return nullptr;
		}
		case SimExpKind::elevation: {
			// TODO
			assert(false && "powerof unsupported");	 // NOLINT
			return nullptr;
		}
		default:
			assert(false && "unreachable");	 // NOLINT
			return nullptr;
	}
	assert(false && "unreachable");	 // NOLINT
	return nullptr;
}

static Value* createSingleUnaryOp(
		IRBuilder<>& builder, ArrayRef<Value*> args, SimExpKind kind)
{
	auto intType = IntegerType::getInt32Ty(builder.getContext());
	auto boolType = IntegerType::getInt1Ty(builder.getContext());
	auto zero = ConstantInt::get(boolType, 0);
	switch (kind)
	{
		case SimExpKind::negate:
			if (args[0]->getType()->isFloatTy())
				return builder.CreateFNeg(args[0]);

			if (args[0]->getType() == intType)
				return builder.CreateNeg(args[0]);

			if (args[0]->getType() == boolType)
				return builder.CreateICmpEQ(args[0], zero);

			assert(false && "unreachable");	 // NOLINT
		default:
			assert(false && "unreachable");	 // NOLINT
			return nullptr;
	}
	assert(false && "unreachable");	 // NOLINT
	return nullptr;
}

static Expected<AllocaInst*> lowerBinOp(
		IRBuilder<>& builder,
		Function* fun,
		const SimExp& exp,
		ArrayRef<Value*> subExp)
{
	assert(exp.isBinary());			 // NOLINT
	assert(subExp.size() == 2);	 // NOLINT
	const auto binaryOp = [type = exp.getKind()](auto& builder, auto args) {
		return createSingleBynaryOP(builder, args, type);
	};

	return elementWiseOperation<2>(builder, fun, subExp, exp.getKind(), binaryOp);
}

static Expected<AllocaInst*> lowerUnOp(
		IRBuilder<>& builder,
		Function* fun,
		const SimExp& exp,
		ArrayRef<Value*> subExp)
{
	assert(exp.isUnary());			 // NOLINT
	assert(subExp.size() == 1);	 // NOLINT

	const auto unaryOp = [type = exp.getKind()](auto& builder, auto args) {
		return createSingleUnaryOp(builder, args, type);
	};

	return elementWiseOperation<1>(builder, fun, subExp, exp.getKind(), unaryOp);
}

static Expected<Value*> lowerExp(
		IRBuilder<>& builder, Function* fun, const SimExp& exp);

static Expected<Value*> lowerTernary(
		IRBuilder<>& builder, Function* fun, const SimExp& exp)
{
	assert(exp.isTernary());	// NOLINT
	const auto leftHandLowerer = [&exp, fun](IRBuilder<>& builder) {
		return lowerExp(builder, fun, exp.getLeftHand());
	};
	const auto rightHandLowerer = [&exp, fun](IRBuilder<>& builder) {
		return lowerExp(builder, fun, exp.getRightHand());
	};
	const auto conditionLowerer =
			[&exp, fun](IRBuilder<>& builder) -> Expected<Value*> {
		auto& condition = exp.getCondition();
		assert(condition.getSimType() == SimType(BultinSimTypes::BOOL));	// NOLINT
		auto ptrToCond = lowerExp(builder, fun, condition);
		if (!ptrToCond)
			return ptrToCond;

		size_t zero = 0;
		return loadToArrayElement(builder, *ptrToCond, zero);
	};

	auto type = exp.getLeftHand().getSimType();
	auto llvmType = typeToLLVMType(builder.getContext(), type)->getPointerTo(0);

	return createTernaryOp(
			fun,
			builder,
			llvmType,
			conditionLowerer,
			leftHandLowerer,
			rightHandLowerer);
}

static Expected<Value*> lowerOperation(
		IRBuilder<>& builder, Function* fun, const SimExp& exp)
{
	assert(exp.isOperation());	// NOLINT

	if (SimExpKind::zero == exp.getKind())
	{
		SimType type(BultinSimTypes::INT);
		IntSimConst constant(0);
		return lowerConstant(builder, constant, type);
	}

	if (exp.isTernary())
		return lowerTernary(builder, fun, exp);

	if (exp.isUnary())
	{
		SmallVector<Value*, 2> values;
		auto subexp = lowerExp(builder, fun, exp.getLeftHand());
		if (!subexp)
			return subexp.takeError();
		values.push_back(move(*subexp));

		return lowerUnOp(builder, fun, exp, values);
	}

	if (exp.isBinary())
	{
		SmallVector<Value*, 2> values;
		auto leftHand = lowerExp(builder, fun, exp.getLeftHand());
		if (!leftHand)
			return leftHand.takeError();
		values.push_back(move(*leftHand));

		auto rightHand = lowerExp(builder, fun, exp.getRightHand());
		if (!rightHand)
			return rightHand.takeError();
		values.push_back(move(*rightHand));

		return lowerBinOp(builder, fun, exp, values);
	}
	assert(false && "Unreachable");
	return nullptr;
}

static Expected<Value*> uncastedLowerExp(
		IRBuilder<>& builder, Function* fun, const SimExp& exp)
{
	if (!exp.areSubExpressionCompatibles())
		return make_error<TypeMissMatch>(exp);

	if (exp.isConstant())
		return lowerConstant(builder, exp);

	if (exp.isReference())
		return lowerReference(builder, exp.getReference());

	return lowerOperation(builder, fun, exp);
}

static Value* castSingleElem(IRBuilder<>& builder, Value* val, Type* type)
{
	auto floatType = Type::getFloatTy(builder.getContext());
	auto intType = Type::getInt32Ty(builder.getContext());
	auto boolType = Type::getInt1Ty(builder.getContext());

	auto constantZero = ConstantInt::get(intType, 0);

	if (type == floatType)
		return builder.CreateSIToFP(val, floatType);

	if (type == intType)
	{
		if (val->getType() == floatType)
			return builder.CreateFPToSI(val, intType);

		return builder.CreateIntCast(val, intType, true);
	}

	if (val->getType() == floatType)
		return builder.CreateFPToSI(val, boolType);

	return builder.CreateTrunc(val, boolType);
}

static Expected<Value*> castReturnValue(
		IRBuilder<>& builder, Value* val, const SimType& type)
{
	auto ptrArrayType = dyn_cast<PointerType>(val->getType());
	auto arrayType = dyn_cast<ArrayType>(ptrArrayType->getContainedType(0));

	assert(arrayType->getNumElements() == type.flatSize());	 // NOLINT

	auto destType = typeToLLVMType(builder.getContext(), type);
	auto singleDestType = destType->getContainedType(0);

	if (destType == arrayType)
		return val;

	auto alloca = allocaSimType(builder, type);

	for (size_t a = 0; a < arrayType->getNumElements(); a++)
	{
		auto loadedElem = loadToArrayElement(builder, val, a);

		Value* casted = castSingleElem(builder, loadedElem, singleDestType);
		storeToArrayElement(builder, casted, alloca, a);
	}

	return alloca;
}

static Expected<Value*> lowerExp(
		IRBuilder<>& builder, Function* fun, const SimExp& exp)
{
	auto retVal = uncastedLowerExp(builder, fun, exp);
	if (!retVal)
		return retVal;

	return castReturnValue(builder, *retVal, exp.getSimType());
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
	assert(printValues != nullptr);	 // NOLINT

	auto expectedMain = makePrivateFunction(entryPointName, m);
	if (!expectedMain)
		return expectedMain;
	auto main = expectedMain.get();
	main->setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);

	IRBuilder<> builder(&main->getEntryBlock());
	// call init
	builder.CreateCall(init);

	const auto forBody = [update, printValues](IRBuilder<>& builder, auto) {
		builder.CreateCall(update);
		builder.CreateCall(printValues);
	};

	// creates a for with simulationStop iterations what invokes
	// update and print values each time
	auto loopExit = createForCycle(main, builder, simulationStop, forBody);

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
		auto val = lowerExp(builder, initFunction, pair.second);
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
		auto expFun = makePrivateFunction("update" + pair.first().str(), m);
		if (!expFun)
			return expFun;

		auto fun = expFun.get();
		bld.SetInsertPoint(&fun->getEntryBlock());

		auto val = lowerExp(bld, fun, pair.second);
		if (!val)
			return val.takeError();
		auto loaded = bld.CreateLoad(*val);
		bld.CreateStore(loaded, m.getGlobalVariable(pair.first(), true));
		bld.CreateRet(nullptr);

		bld.SetInsertPoint(&updateFunction->getEntryBlock());
		bld.CreateCall(fun);
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

static void createPrintOfVar(
		Module& m,
		IRBuilder<>& builder,
		GlobalValue* varString,
		GlobalValue* ptrToVar)
{
	size_t index = 0;
	auto ptrToFirstElem = getArrayElementPtr(builder, ptrToVar, index);
	auto ptrToStrName = getArrayElementPtr(builder, varString, index);

	auto ptrType = ptrToVar->getType();
	auto arrayType = dyn_cast<ArrayType>(ptrType->getContainedType(0));
	auto ptrToBaseType = ptrToFirstElem->getType();
	auto baseType = ptrToBaseType->getContainedType(0);

	auto charPtrType = ptrToStrName->getType();
	auto intType = IntegerType::getInt32Ty(builder.getContext());

	SmallVector<Type*, 3> args({ charPtrType, ptrToBaseType, intType });
	auto printType =
			FunctionType::get(Type::getVoidTy(m.getContext()), args, false);

	const auto selectPrintName = [intType](Type* t) {
		if (t->isFloatTy())
			return "modelicaPrintFVector";
		if (t == intType)
			return "modelicaPrintIVector";

		return "modelicaPrintBVector";
	};

	auto callee = m.getOrInsertFunction(selectPrintName(baseType), printType);
	auto externalPrint = dyn_cast<Function>(callee.getCallee());

	auto numElements = ConstantInt::get(intType, arrayType->getNumElements());
	SmallVector<Value*, 3> argsVal({ ptrToStrName, ptrToFirstElem, numElements });
	builder.CreateCall(externalPrint, argsVal);
}

static Expected<Function*> populatePrint(Module& m, StringMap<SimExp> vars)
{
	auto printFunctionExpected = makePrivateFunction("print", m);
	if (!printFunctionExpected)
		return printFunctionExpected;
	auto printFunction = printFunctionExpected.get();
	IRBuilder bld(&printFunction->getEntryBlock());

	for (const auto& pair : vars)
	{
		auto varString = m.getGlobalVariable(pair.first().str() + "_str");
		auto ptrToVar = m.getGlobalVariable(pair.first());
		createPrintOfVar(m, bld, varString, ptrToVar);
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

void Simulation::dumpHeader(raw_ostream& OS) const
{
	OS << "#pragma once\n\n";

	for (const auto& var : variables)
	{
		const auto& type = var.second.getSimType();

		OS << "extern ";
		type.dumpCSyntax(var.first(), OS);

		OS << ";\n";
	}

	OS << "extern \"C\"{";
	OS << "void " << entryPointName << "();";
	OS << "}";
}
