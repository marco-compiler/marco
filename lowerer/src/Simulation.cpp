#include "modelica/lowerer/Simulation.hpp"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IRBuilder.h"

using namespace std;
using namespace modelica;

constexpr auto internalLinkage =
		llvm::GlobalValue::LinkageTypes::InternalLinkage;

static llvm::Type* builtInToLLVMType(
		llvm::LLVMContext& context, BultinTypes type)
{
	switch (type)
	{
		case BultinTypes::INT:
			return llvm::Type::getInt32Ty(context);
		case BultinTypes::BOOL:
			return llvm::Type::getInt1Ty(context);
		case BultinTypes::FLOAT:
			return llvm::Type::getFloatTy(context);
	}

	assert(false);	// NOLINT
	return nullptr;
}

static llvm::Type* typeToLLVMType(llvm::LLVMContext& context, const Type& type)
{
	auto baseType = builtInToLLVMType(context, type.getBuiltin());

	if (type.getDimensionsCount() == 0)
		return baseType;

	return llvm::ArrayType::get(baseType, type.flatSize());
}

static void createGlobal(
		llvm::Module& module, llvm::StringRef name, const Expression& initValue)
{
	auto type = typeToLLVMType(module.getContext(), initValue.getType());
	auto varDecl = module.getOrInsertGlobal(name, type);

	auto global = llvm::dyn_cast<llvm::GlobalVariable>(varDecl);
	global->setLinkage(internalLinkage);
	auto constant = llvm::ConstantInt::get(type, 0);
	global->setInitializer(constant);
}

static llvm::FunctionType* getVoidType(llvm::LLVMContext& context)
{
	return llvm::FunctionType::get(llvm::Type::getVoidTy(context), false);
}

template<typename T>
static llvm::Value* makeStore(
		llvm::IRBuilder<>& builder,
		T value,
		llvm::Type* llvmType,
		llvm::Value* location)
{
	if constexpr (std::is_same<T, int>::value)
		return builder.CreateStore(
				llvm::ConstantInt::get(llvmType, value), location);
	else if constexpr (std::is_same<T, bool>::value)
		return builder.CreateStore(
				llvm::ConstantInt::get(llvmType, value), location);
	else if constexpr (std::is_same<T, float>::value)
		return builder.CreateStore(
				llvm::ConstantFP::get(llvmType, value), location);

	assert(false);	// NOLINT
	return nullptr;
}

template<typename T>
static llvm::Value* lowerConstant(
		llvm::IRBuilder<> builder, const Constant<T>& constant, const Type& type)
{
	auto llvmType = typeToLLVMType(builder.getContext(), type);
	auto alloca = builder.CreateAlloca(llvmType);
	if (type.getDimensionsCount() == 0)
	{
		makeStore<T>(builder, constant.get(0), llvmType, alloca);
		return builder.CreateLoad(alloca);
	}

	auto intType = llvm::Type::getInt32Ty(builder.getContext());
	for (size_t i = 0; i < constant.size(); i++)
	{
		auto index = llvm::ConstantInt::get(intType, i);
		auto element = builder.CreateGEP(alloca, index);

		makeStore<T>(builder, constant.get(i), llvmType, element);
	}
	return alloca;
}

static llvm::Value* lowerConstant(
		llvm::IRBuilder<>& builder, const Expression& exp)
{
	if (exp.isConstant<int>())
		return lowerConstant<int>(builder, exp.getConstant<int>(), exp.getType());
	if (exp.isConstant<float>())
		return lowerConstant<float>(
				builder, exp.getConstant<float>(), exp.getType());
	if (exp.isConstant<bool>())
		return lowerConstant<bool>(builder, exp.getConstant<bool>(), exp.getType());

	return nullptr;
}

static llvm::Value* lowerReference(
		llvm::IRBuilder<>& builder, const Expression& exp)
{
	auto module = builder.GetInsertBlock()->getModule();
	auto global = module->getGlobalVariable(exp.getReference() + "_old", true);
	return builder.CreateLoad(global);
}

static llvm::Value* lowerExpression(
		llvm::IRBuilder<>& builder,
		const Expression& exp,
		llvm::SmallVector<llvm::Value*, 0>& subExp)
{
	switch (exp.getKind())
	{
		case ExpressionKind::zero:
		{
			assert(subExp.size() == 0);	// NOLINT
			auto intType = llvm::Type::getInt32Ty(builder.getContext());
			auto loc = builder.CreateAlloca(intType);
			makeStore<int>(builder, 0, intType, loc);
			return builder.CreateLoad(intType, loc);
		}
		case ExpressionKind::negate:
		{
			assert(subExp.size() == 1);	// NOLINT
			return builder.CreateNeg(subExp[0]);
		}
		case ExpressionKind::add:
		{
			assert(subExp.size() == 2);	// NOLINT
			return builder.CreateAdd(subExp[0], subExp[1]);
		}
		case ExpressionKind::sub:
		{
			assert(subExp.size() == 2);	// NOLINT
			return builder.CreateSub(subExp[0], subExp[1]);
		}
		case ExpressionKind::mult:
		{
			assert(subExp.size() == 2);	// NOLINT
			return builder.CreateMul(subExp[0], subExp[1]);
		}
		case ExpressionKind::divide:
		{
			assert(subExp.size() == 2);	// NOLINT
			if (exp.getLeftHand().getType().getBuiltin() != BultinTypes::FLOAT)
				return builder.CreateSDiv(subExp[0], subExp[1]);
			return builder.CreateFDiv(subExp[0], subExp[1]);
		}
		case ExpressionKind::equal:
		{
			assert(subExp.size() == 2);	// NOLINT
			return builder.CreateICmpEQ(subExp[0], subExp[1]);
		}
		case ExpressionKind::different:
		{
			assert(subExp.size() == 2);	// NOLINT
			return builder.CreateICmpNE(subExp[0], subExp[1]);
		}
		case ExpressionKind::greaterEqual:
		{
			assert(subExp.size() == 2);	// NOLINT
			return builder.CreateICmpSGE(subExp[0], subExp[1]);
		}
		case ExpressionKind::greaterThan:
		{
			assert(subExp.size() == 2);	// NOLINT
			return builder.CreateICmpSGT(subExp[0], subExp[1]);
		}
		case ExpressionKind::lessEqual:
		{
			assert(subExp.size() == 2);	// NOLINT
			return builder.CreateICmpSLE(subExp[0], subExp[1]);
		}
		case ExpressionKind::less:
		{
			assert(subExp.size() == 2);	// NOLINT
			return builder.CreateICmpSLT(subExp[0], subExp[1]);
		}
		case ExpressionKind::module:
		{
			assert(subExp.size() == 2);	// NOLINT
			// TODO
			assert(false && "module unsupported");	// NOLINT
			return nullptr;
		}
		case ExpressionKind::elevation:
		{
			assert(subExp.size() == 2);	// NOLINT
			// TODO
			assert(false && "powerof unsupported");	// NOLINT
			return nullptr;
		}
		case ExpressionKind::conditional:
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

static llvm::Value* lowerExp(llvm::IRBuilder<>& builder, const Expression& exp)
{
	llvm::SmallVector<llvm::Value*, 0> values;

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
		llvm::Function* main,
		llvm::Function* init,
		llvm::Function* update,
		llvm::Function* printValues,
		unsigned simulationStop)
{
	// Creates the 3 basic blocks
	auto condition =
			llvm::BasicBlock::Create(main->getContext(), "condition", main);
	auto loopBody =
			llvm::BasicBlock::Create(main->getContext(), "loopBody", main);
	auto exit = llvm::BasicBlock::Create(main->getContext(), "exit", main);

	llvm::IRBuilder builder(&main->getEntryBlock());
	auto unsignedInt = llvm::Type::getInt32Ty(builder.getContext());

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
			builder.CreateICmpEQ(value, llvm::Constant::getNullValue(unsignedInt));

	// brach if equal to zero
	builder.CreateCondBr(iterCmp, exit, loopBody);

	// invoke update
	builder.SetInsertPoint(loopBody);
	builder.CreateCall(update);
	builder.CreateCall(printValues);

	// load, reduce and store the counter
	value = builder.CreateLoad(unsignedInt, iterationCounter);
	auto reducedCounter =
			builder.CreateSub(value, llvm::ConstantInt::get(unsignedInt, 1));
	builder.CreateStore(reducedCounter, iterationCounter);
	builder.CreateBr(condition);

	builder.SetInsertPoint(exit);
	builder.CreateRet(nullptr);
}

static void insertGlobal(
		llvm::Module& module, llvm::StringRef name, const Expression& exp)
{
	createGlobal(module, name, exp);
	createGlobal(module, name.str() + "_old", exp);

	std::string varName = name.str() + "_str";
	auto str = llvm::ConstantDataArray::getString(module.getContext(), name);
	auto type = llvm::ArrayType::get(
			llvm::IntegerType::getInt8Ty(module.getContext()), name.size() + 1);
	auto global = module.getOrInsertGlobal(varName, type);
	llvm::dyn_cast<llvm::GlobalVariable>(global)->setInitializer(str);
}

void Simulation::lower()
{
	auto initFunction = makePrivateFunction("init");
	auto updateFunction = makePrivateFunction("update");
	auto mainFunction = makePrivateFunction("main");
	auto printFunction = makePrivateFunction("print");
	mainFunction->setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);

	auto& module = this->module;
	std::for_each(
			variables.begin(), variables.end(), [&module](const auto& pair) {
				insertGlobal(module, pair.first(), pair.second);
			});

	llvm::IRBuilder builder(&initFunction->getEntryBlock());
	std::for_each(
			variables.begin(),
			variables.end(),
			[&builder, &module](const auto& pair) {
				auto val = lowerExp(builder, pair.second);
				builder.CreateStore(val, module.getGlobalVariable(pair.first(), true));
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

	auto charType = llvm::IntegerType::getInt8Ty(module.getContext());
	auto floatType = llvm::IntegerType::getFloatTy(module.getContext());
	auto charPtrType = llvm::PointerType::get(charType, 0);
	llvm::SmallVector<llvm::Type*, 2> args({ charPtrType, floatType });
	auto printType = llvm::FunctionType::get(
			llvm::Type::getVoidTy(module.getContext()), args, false);
	auto externalPrint = llvm::dyn_cast<llvm::Function>(
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
				auto casted =
						builder.CreateCast(llvm::Instruction::SIToFP, value, floatType);
				llvm::SmallVector<llvm::Value*, 2> args({ charStar, casted });
				builder.CreateCall(externalPrint, args);
			});
	builder.CreateRet(nullptr);
}

llvm::Function* Simulation::makePrivateFunction(llvm::StringRef name)
{
	auto function = module.getOrInsertFunction(name, getVoidType(context));
	auto f = llvm::dyn_cast<llvm::Function>(function);
	llvm::BasicBlock::Create(function->getContext(), "entry", f);
	f->setLinkage(internalLinkage);
	return f;
}

void Simulation::dump(llvm::raw_ostream& OS) const
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

void Simulation::dumpBC(llvm::raw_ostream& OS) const
{
	llvm::WriteBitcodeToFile(module, OS);
}
