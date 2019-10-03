#include "modelica/lowerer/Simulation.hpp"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IRBuilder.h"

using namespace std;
using namespace modelica;
using namespace llvm;

constexpr auto internalLinkage = GlobalValue::LinkageTypes::InternalLinkage;

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

static FunctionType* getVoidType(LLVMContext& context)
{
	return FunctionType::get(Type::getVoidTy(context), false);
}

template<typename T>
static Value* makeStore(
		IRBuilder<>& builder, T value, Type* llvmType, Value* location)
{
	if constexpr (std::is_same<T, int>::value)
		return builder.CreateStore(ConstantInt::get(llvmType, value), location);
	else if constexpr (std::is_same<T, bool>::value)
		return builder.CreateStore(ConstantInt::get(llvmType, value), location);
	else if constexpr (std::is_same<T, float>::value)
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
			assert(false && "powerof unsupported");	// NOLINT
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

static void populateMain(
		Function* main,
		Function* init,
		Function* update,
		Function* printValues,
		unsigned simulationStop)
{
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
}

static void insertGlobal(
		Module& module,
		StringRef name,
		const SimExp& exp,
		GlobalValue::LinkageTypes linkage)
{
	createGlobal(module, name, exp, linkage);
	createGlobal(module, name.str() + "_old", exp, linkage);

	std::string varName = name.str() + "_str";
	auto str = ConstantDataArray::getString(module.getContext(), name);
	auto type = ArrayType::get(
			IntegerType::getInt8Ty(module.getContext()), name.size() + 1);
	auto global = module.getOrInsertGlobal(varName, type);
	dyn_cast<GlobalVariable>(global)->setInitializer(str);
}

void Simulation::lower()
{
	auto initFunction = makePrivateFunction("init");
	auto updateFunction = makePrivateFunction("update");
	auto mainFunction = makePrivateFunction(entryPointName);
	auto printFunction = makePrivateFunction("print");
	mainFunction->setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);

	auto& module = this->module;
	std::for_each(
			variables.begin(),
			variables.end(),
			[&module, linkage = getVarLinkage()](const auto& pair) {
				insertGlobal(module, pair.first(), pair.second, linkage);
			});

	IRBuilder builder(&initFunction->getEntryBlock());
	std::for_each(
			variables.begin(),
			variables.end(),
			[&builder, &module](const auto& pair) {
				auto val = lowerExp(builder, pair.second);
				builder.CreateStore(val, module.getGlobalVariable(pair.first(), true));
				builder.CreateStore(
						val, module.getGlobalVariable(pair.first().str() + "_old", true));
			});
	builder.CreateRet(nullptr);
	builder.SetInsertPoint(&updateFunction->getEntryBlock());

	std::for_each(
			updates.begin(), updates.end(), [&builder, &module](const auto& pair) {
				auto val = lowerExp(builder, pair.second);
				builder.CreateStore(val, module.getGlobalVariable(pair.first(), true));
			});
	std::for_each(
			updates.begin(), updates.end(), [&builder, &module](const auto& pair) {
				auto globalVal = module.getGlobalVariable(pair.first(), true);
				auto val = builder.CreateLoad(globalVal);

				builder.CreateStore(
						val, module.getGlobalVariable(pair.first().str() + "_old", true));
			});

	builder.CreateRet(nullptr);

	populateMain(
			mainFunction, initFunction, updateFunction, printFunction, stopTime);

	builder.SetInsertPoint(&printFunction->getEntryBlock());

	auto charType = IntegerType::getInt8Ty(module.getContext());
	auto floatType = IntegerType::getFloatTy(module.getContext());
	auto charPtrType = PointerType::get(charType, 0);
	SmallVector<Type*, 2> args({ charPtrType, floatType });
	auto printType =
			FunctionType::get(Type::getVoidTy(module.getContext()), args, false);
	auto externalPrint = dyn_cast<Function>(
			module.getOrInsertFunction("modelicaPrint", printType));

	std::for_each(
			variables.begin(),
			variables.end(),
			[&builder, &module, externalPrint, floatType, charPtrType](
					const auto& pair) {
				auto strName = module.getGlobalVariable(pair.first().str() + "_str");
				auto charStar = builder.CreateBitCast(strName, charPtrType);

				auto value =
						builder.CreateLoad(module.getGlobalVariable(pair.first(), true));
				auto casted = builder.CreateCast(Instruction::SIToFP, value, floatType);
				SmallVector<Value*, 2> args({ charStar, casted });
				builder.CreateCall(externalPrint, args);
			});
	builder.CreateRet(nullptr);
}

Function* Simulation::makePrivateFunction(StringRef name)
{
	auto function = module.getOrInsertFunction(name, getVoidType(context));
	auto f = dyn_cast<Function>(function);
	BasicBlock::Create(function->getContext(), "entry", f);
	f->setLinkage(internalLinkage);
	return f;
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
	std::for_each(variables.begin(), variables.end(), dumpEquation);

	OS << "Update:\n";
	std::for_each(updates.begin(), updates.end(), dumpEquation);
}

void Simulation::dumpBC(raw_ostream& OS) const
{
	WriteBitcodeToFile(module, OS);
}
