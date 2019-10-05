#include "modelica/lowerer/Simulation.hpp"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IRBuilder.h"
#include "modelica/lowerer/SimLowerer.hpp"

using namespace std;
using namespace modelica;
using namespace llvm;

template<typename T>
static Value* lowerConstant(
		IRBuilder<> builder, const SimConst<T>& constant, const SimType& type)
{
	auto llvmType = typeToLLVMType(builder.getContext(), type);
	auto alloca = builder.CreateAlloca(llvmType);
	if (type.getDimensionsCount() == 0)
	{
		makeConstantStore<T>(builder, constant.get(0), llvmType, alloca);
		return builder.CreateLoad(alloca);
	}

	auto intType = Type::getInt32Ty(builder.getContext());
	for (size_t i = 0; i < constant.size(); i++)
	{
		auto index = ConstantInt::get(intType, i);
		auto element = builder.CreateGEP(alloca, index);

		makeConstantStore<T>(builder, constant.get(i), llvmType, element);
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
			makeConstantStore<int>(builder, 0, intType, loc);
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
		return lowerReference(builder, exp.getReference());

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
	makeConstantStore<int>(
			builder, simulationStop, unsignedInt, iterationCounter);

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

static Function* initializeGlobals(Module& m, StringMap<SimExp> vars)
{
	auto initFunction = makePrivateFunction("init", m);
	IRBuilder builder(&initFunction->getEntryBlock());

	for (const auto& pair : vars)
	{
		auto val = lowerExp(builder, pair.second);
		builder.CreateStore(val, m.getGlobalVariable(pair.first(), true));
		builder.CreateStore(
				val, m.getGlobalVariable(pair.first().str() + "_old", true));
	}
	builder.CreateRet(nullptr);
	return initFunction;
}

static Function* createUpdates(Module& m, StringMap<SimExp> upds)
{
	auto updateFunction = makePrivateFunction("update", m);
	IRBuilder bld(&updateFunction->getEntryBlock());

	for (const auto& pair : upds)
	{
		auto val = lowerExp(bld, pair.second);
		bld.CreateStore(val, m.getGlobalVariable(pair.first(), true));
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

	for (const auto& pair : vars)
	{
		auto strName = m.getGlobalVariable(pair.first().str() + "_str");
		auto charStar = bld.CreateBitCast(strName, charPtrType);

		auto value = bld.CreateLoad(m.getGlobalVariable(pair.first(), true));
		auto casted = bld.CreateCast(Instruction::SIToFP, value, floatType);
		SmallVector<Value*, 2> args({ charStar, casted });
		bld.CreateCall(externalPrint, args);
	}
	bld.CreateRet(nullptr);
	return printFunction;
}

Error Simulation::lower()
{
	if (auto e = createAllGlobals(module, variables, getVarLinkage()); e)
		return e;
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
