#include "modelica/lowerer/Simulation.hpp"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IRBuilder.h"
#include "modelica/lowerer/SimLowerer.hpp"

using namespace std;
using namespace modelica;
using namespace llvm;

template<typename T>
static Expected<AllocaInst*> lowerConstant(
		IRBuilder<> builder, const SimConst<T>& constant, const SimType& type)
{
	if (constant.size() != type.flatSize())
		return make_error<TypeConstantSizeMissMatch>(constant, type);

	auto llvmType = typeToLLVMType(builder.getContext(), type);
	auto alloca = builder.CreateAlloca(llvmType);
	auto intType = Type::getInt32Ty(builder.getContext());

	for (size_t i = 0; i < constant.size(); i++)
	{
		auto index = ConstantInt::get(intType, i);
		auto element = builder.CreateGEP(alloca, index);

		auto ptrToElem = builder.CreateBitCast(
				element, llvmType->getContainedType(0)->getPointerTo(0));

		makeConstantStore<T>(
				builder, constant.get(i), llvmType->getContainedType(0), ptrToElem);
	}
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
		array<Value*, argumentsSize> args,
		std::function<Value*(IRBuilder<>&, array<Value*, argumentsSize>&)>
				operation)
{
	auto ptrType = dyn_cast<PointerType>(args.at(0)->getType());
	auto arrayType = dyn_cast<ArrayType>(ptrType->getContainedType(0));
	auto fondamentalType = arrayType->getContainedType(0);
	auto ptrToFoundamental = fondamentalType->getPointerTo(0);

	auto alloca = builder.CreateAlloca(arrayType);
	auto intType = Type::getInt32Ty(builder.getContext());

	for (size_t a = 0; a < arrayType->getNumElements(); a++)
	{
		auto index = ConstantInt::get(intType, a);

		array<Value*, argumentsSize> arguments;
		for (int i = 0; i < argumentsSize; i++)
		{
			auto ptr = builder.CreateGEP(args.at(i), index);
			auto ptrToSingle = builder.CreateBitCast(ptr, ptrToFoundamental);
			arguments.at(i) = builder.CreateLoad(ptrToSingle);
		}

		auto outVal = operation(builder, arguments);
		auto outPtr = builder.CreateGEP(alloca, index);
		auto outPtrSingle = builder.CreateBitCast(outPtr, ptrToFoundamental);
		builder.CreateStore(outVal, outPtrSingle);
	}
	return alloca;
}

static Expected<AllocaInst*> lowerOperation(
		IRBuilder<>& builder, const SimExp& exp, SmallVector<Value*, 3>& subExp)
{
	const auto unaryOp = [type = exp.getKind()](
													 IRBuilder<>& builder,
													 array<Value*, 1> args) -> Value* {
		switch (type)
		{
			case SimExpKind::negate:
				return builder.CreateNeg(args[0]);
			default:
				assert(false && "unreachable");	// NOLINT
				return nullptr;
		}
	};
	const auto binaryOp = [type = exp.getKind()](
														IRBuilder<>& builder,
														array<Value*, 2> args) -> Value* {
		switch (type)
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
	};

	if (exp.isUnary())
	{
		assert(subExp.size() == 1);	// NOLINT
		array<Value*, 1> args = { subExp[0] };
		return elementWiseOperation<1>(builder, args, unaryOp);
	}

	if (exp.isBinary())
	{
		assert(subExp.size() == 2);	// NOLINT
		array<Value*, 2> args = { subExp[0], subExp[1] };
		return elementWiseOperation<2>(builder, args, binaryOp);
	}

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

	auto intType = IntegerType::getInt32Ty(m.getContext());
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
			auto index = ConstantInt::get(intType, a);
			auto ptr = bld.CreateGEP(variable, index);
			auto foundamentalVarType =
					ptr->getType()->getContainedType(0)->getContainedType(0);
			auto singleElementPtr =
					bld.CreateBitCast(ptr, foundamentalVarType->getPointerTo(0));
			auto value = bld.CreateLoad(singleElementPtr);
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
