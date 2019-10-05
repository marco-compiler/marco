#include "modelica/lowerer/Simulation.hpp"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IRBuilder.h"

using namespace std;
using namespace modelica;
using namespace llvm;

constexpr auto internalLinkage = GlobalValue::LinkageTypes::InternalLinkage;

static FunctionType* getVoidType(LLVMContext& context)
{
	return FunctionType::get(Type::getVoidTy(context), false);
}

static Function* makePrivateFunction(StringRef name, Module& m)
{
	auto function = m.getOrInsertFunction(name, getVoidType(m.getContext()));
	auto f = dyn_cast<Function>(function);
	BasicBlock::Create(function->getContext(), "entry", f);
	f->setLinkage(internalLinkage);
	return f;
}

static Type* builtInToLLVMType(LLVMContext& context, BultinSimTypes type)
{
	switch (type)
	{
		case BultinSimTypes::INT:
			return Type::getInt32Ty(context);
		case BultinSimTypes::BOOL:
			return Type::getInt1Ty(context);
		case BultinSimTypes::FLOAT:
			return Type::getFloatTy(context);
	}

	assert(false);	// NOLINT
	return nullptr;
}

static Type* typeToLLVMType(LLVMContext& context, const SimType& type)
{
	auto baseType = builtInToLLVMType(context, type.getBuiltin());

	if (type.getDimensionsCount() == 0)
		return baseType;

	return ArrayType::get(baseType, type.flatSize());
}

static void createGlobal(
		Module& module,
		StringRef name,
		const SimExp& initValue,
		GlobalValue::LinkageTypes linkage)
{
	auto type = typeToLLVMType(module.getContext(), initValue.getSimType());
	auto varDecl = module.getOrInsertGlobal(name, type);

	auto global = dyn_cast<GlobalVariable>(varDecl);
	global->setLinkage(linkage);
	auto constant = ConstantInt::get(type, 0);
	global->setInitializer(constant);
}

template<typename T>
static Value* makeStore(
		IRBuilder<>& builder, T value, Type* llvmType, Value* location)
{
	if constexpr (is_same<T, int>::value)
		return builder.CreateStore(ConstantInt::get(llvmType, value), location);
	else if constexpr (is_same<T, bool>::value)
		return builder.CreateStore(ConstantInt::get(llvmType, value), location);
	else if constexpr (is_same<T, float>::value)
		return builder.CreateStore(ConstantFP::get(llvmType, value), location);

	assert(false);	// NOLINT
	return nullptr;
}

template<typename T>
static Value* lowerConstant(
		IRBuilder<> builder, const SimConst<T>& constant, const SimType& type)
{
	auto llvmType = typeToLLVMType(builder.getContext(), type);
	auto alloca = builder.CreateAlloca(llvmType);
	if (type.getDimensionsCount() == 0)
	{
		makeStore<T>(builder, constant.get(0), llvmType, alloca);
		return builder.CreateLoad(alloca);
	}

	auto intType = Type::getInt32Ty(builder.getContext());
	for (size_t i = 0; i < constant.size(); i++)
	{
		auto index = ConstantInt::get(intType, i);
		auto element = builder.CreateGEP(alloca, index);

		makeStore<T>(builder, constant.get(i), llvmType, element);
	}
	return alloca;
}

static Value* lowerConstant(IRBuilder<>& builder, const SimExp& exp)
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

	return nullptr;
}

static Value* lowerReference(IRBuilder<>& builder, const SimExp& exp)
{
	auto module = builder.GetInsertBlock()->getModule();
	auto global = module->getGlobalVariable(exp.getReference() + "_old", true);
	return builder.CreateLoad(global);
}

static Value* lowerExpression(
		IRBuilder<>& builder, const SimExp& exp, SmallVector<Value*, 0>& subExp)
{
	switch (exp.getKind())
	{
		case SimExpKind::zero:
		{
			assert(subExp.size() == 0);	// NOLINT
			auto intType = Type::getInt32Ty(builder.getContext());
			auto loc = builder.CreateAlloca(intType);
			makeStore<int>(builder, 0, intType, loc);
			return builder.CreateLoad(intType, loc);
		}
		case SimExpKind::negate:
		{
			assert(subExp.size() == 1);	// NOLINT
			return builder.CreateNeg(subExp[0]);
		}
		case SimExpKind::add:
		{
			assert(subExp.size() == 2);	// NOLINT
			return builder.CreateAdd(subExp[0], subExp[1]);
		}
		case SimExpKind::sub:
		{
			assert(subExp.size() == 2);	// NOLINT
			return builder.CreateSub(subExp[0], subExp[1]);
		}
		case SimExpKind::mult:
		{
			assert(subExp.size() == 2);	// NOLINT
			return builder.CreateMul(subExp[0], subExp[1]);
		}
		case SimExpKind::divide:
		{
			assert(subExp.size() == 2);	// NOLINT
			if (exp.getLeftHand().getSimType().getBuiltin() != BultinSimTypes::FLOAT)
				return builder.CreateSDiv(subExp[0], subExp[1]);
			return builder.CreateFDiv(subExp[0], subExp[1]);
		}
		case SimExpKind::equal:
		{
			assert(subExp.size() == 2);	// NOLINT
			return builder.CreateICmpEQ(subExp[0], subExp[1]);
		}
		case SimExpKind::different:
		{
			assert(subExp.size() == 2);	// NOLINT
			return builder.CreateICmpNE(subExp[0], subExp[1]);
		}
		case SimExpKind::greaterEqual:
		{
			assert(subExp.size() == 2);	// NOLINT
			return builder.CreateICmpSGE(subExp[0], subExp[1]);
		}
		case SimExpKind::greaterThan:
		{
			assert(subExp.size() == 2);	// NOLINT
			return builder.CreateICmpSGT(subExp[0], subExp[1]);
		}
		case SimExpKind::lessEqual:
		{
			assert(subExp.size() == 2);	// NOLINT
			return builder.CreateICmpSLE(subExp[0], subExp[1]);
		}
		case SimExpKind::less:
		{
			assert(subExp.size() == 2);	// NOLINT
			return builder.CreateICmpSLT(subExp[0], subExp[1]);
		}
		case SimExpKind::module:
		{
			assert(subExp.size() == 2);	// NOLINT
			// TODO
			assert(false && "module unsupported");	// NOLINT
			return nullptr;
		}
		case SimExpKind::elevation:
		{
			assert(subExp.size() == 2);	// NOLINT
			// TODO
			assert(false && "powerof unsupported");	// NOLINT
			return nullptr;
		}
		case SimExpKind::conditional:
		{
			assert(subExp.size() == 3);	// NOLINT
			// TODO
			assert(false && "conditional unsupported");	// NOLINT
			return nullptr;
		}
	}
	assert(false && "unreachable");	// NOLINT
	return nullptr;
}

static Value* lowerExp(IRBuilder<>& builder, const SimExp& exp)
{
	SmallVector<Value*, 0> values;

	if (exp.isConstant())
		return lowerConstant(builder, exp);

	if (exp.isReference())
		return lowerReference(builder, exp);

	if (exp.isUnary() || exp.isBinary() || exp.isTernary())
		values.push_back(lowerExp(builder, exp.getLeftHand()));
	if (exp.isBinary() || exp.isTernary())
		values.push_back(lowerExp(builder, exp.getRightHand()));
	if (exp.isTernary())
		values.push_back(lowerExp(builder, exp.getCondition()));

	return lowerExpression(builder, exp, values);
}

static Function* populateMain(
		Module& m,
		StringRef entryPointName,
		Function* init,
		Function* update,
		Function* printValues,
		unsigned simulationStop)
{
	auto main = makePrivateFunction(entryPointName, m);
	main->setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);

	// Creates the 3 basic blocks
	auto condition = BasicBlock::Create(main->getContext(), "condition", main);
	auto loopBody = BasicBlock::Create(main->getContext(), "loopBody", main);
	auto exit = BasicBlock::Create(main->getContext(), "exit", main);

	IRBuilder builder(&main->getEntryBlock());
	auto unsignedInt = Type::getInt32Ty(builder.getContext());

	// call init
	builder.CreateCall(init);

	// allocate and initialize counter
	auto iterationCounter = builder.CreateAlloca(unsignedInt);
	makeStore<int>(builder, simulationStop, unsignedInt, iterationCounter);

	// jump to condition bb
	builder.CreateBr(condition);

	// load counter
	builder.SetInsertPoint(condition);
	auto value = builder.CreateLoad(unsignedInt, iterationCounter);
	auto iterCmp =
			builder.CreateICmpEQ(value, Constant::getNullValue(unsignedInt));

	// brach if equal to zero
	builder.CreateCondBr(iterCmp, exit, loopBody);

	// invoke update
	builder.SetInsertPoint(loopBody);
	builder.CreateCall(update);
	builder.CreateCall(printValues);

	// load, reduce and store the counter
	value = builder.CreateLoad(unsignedInt, iterationCounter);
	auto reducedCounter =
			builder.CreateSub(value, ConstantInt::get(unsignedInt, 1));
	builder.CreateStore(reducedCounter, iterationCounter);
	builder.CreateBr(condition);

	builder.SetInsertPoint(exit);
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

static void insertGlobal(
		Module& module,
		StringRef name,
		const SimExp& exp,
		GlobalValue::LinkageTypes linkage)
{
	createGlobal(module, name, exp, linkage);
	createGlobal(module, name.str() + "_old", exp, linkage);

	insertGlobalString(module, name.str() + "_str", name);
}

static void createAllGlobals(
		Module& m, StringMap<SimExp> vars, GlobalValue::LinkageTypes linkType)
{
	for_each(begin(vars), end(vars), [&m, linkType](const auto& pair) {
		insertGlobal(m, pair.first(), pair.second, linkType);
	});
}

static Function* initializeGlobals(Module& m, StringMap<SimExp> vars)
{
	auto initFunction = makePrivateFunction("init", m);
	IRBuilder builder(&initFunction->getEntryBlock());
	for_each(begin(vars), end(vars), [&builder, &m](const auto& pair) {
		auto val = lowerExp(builder, pair.second);
		builder.CreateStore(val, m.getGlobalVariable(pair.first(), true));
		builder.CreateStore(
				val, m.getGlobalVariable(pair.first().str() + "_old", true));
	});
	builder.CreateRet(nullptr);
	return initFunction;
}

static Function* createUpdates(Module& m, StringMap<SimExp> upds)
{
	auto updateFunction = makePrivateFunction("update", m);
	IRBuilder bld(&updateFunction->getEntryBlock());

	for_each(begin(upds), end(upds), [&bld, &m](const auto& pair) {
		auto val = lowerExp(bld, pair.second);
		bld.CreateStore(val, m.getGlobalVariable(pair.first(), true));
	});
	for_each(begin(upds), end(upds), [&bld, &m](const auto& pair) {
		auto globalVal = m.getGlobalVariable(pair.first(), true);
		auto val = bld.CreateLoad(globalVal);

		bld.CreateStore(
				val, m.getGlobalVariable(pair.first().str() + "_old", true));
	});

	bld.CreateRet(nullptr);
	return updateFunction;
}

static Function* populatePrint(Module& m, StringMap<SimExp> vars)
{
	auto printFunction = makePrivateFunction("print", m);
	IRBuilder bld(&printFunction->getEntryBlock());

	auto charType = IntegerType::getInt8Ty(m.getContext());
	auto floatType = IntegerType::getFloatTy(m.getContext());
	auto charPtrType = PointerType::get(charType, 0);
	SmallVector<Type*, 2> args({ charPtrType, floatType });
	auto printType =
			FunctionType::get(Type::getVoidTy(m.getContext()), args, false);
	auto externalPrint =
			dyn_cast<Function>(m.getOrInsertFunction("modelicaPrint", printType));

	for_each(
			begin(vars),
			end(vars),
			[&bld, &m, externalPrint, floatType, charPtrType](const auto& pair) {
				auto strName = m.getGlobalVariable(pair.first().str() + "_str");
				auto charStar = bld.CreateBitCast(strName, charPtrType);

				auto value = bld.CreateLoad(m.getGlobalVariable(pair.first(), true));
				auto casted = bld.CreateCast(Instruction::SIToFP, value, floatType);
				SmallVector<Value*, 2> args({ charStar, casted });
				bld.CreateCall(externalPrint, args);
			});
	bld.CreateRet(nullptr);
	return printFunction;
}

void Simulation::lower()
{
	createAllGlobals(module, variables, getVarLinkage());
	auto initFunction = initializeGlobals(module, variables);
	auto updateFunction = createUpdates(module, updates);
	auto printFunction = populatePrint(module, variables);
	populateMain(
			module,
			entryPointName,
			initFunction,
			updateFunction,
			printFunction,
			stopTime);
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
