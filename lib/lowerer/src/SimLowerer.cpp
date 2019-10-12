#include "modelica/lowerer/SimLowerer.hpp"

#include "llvm/IR/IRBuilder.h"
#include "modelica/lowerer/SimConst.hpp"
#include "modelica/lowerer/Simulation.hpp"

using namespace llvm;
using namespace std;

namespace modelica
{
	constexpr auto internalLinkage = GlobalValue::LinkageTypes::InternalLinkage;

	static FunctionType* getVoidType(LLVMContext& context)
	{
		return FunctionType::get(Type::getVoidTy(context), false);
	}

	Expected<Function*> makePrivateFunction(StringRef name, Module& m)
	{
		if (m.getFunction(name) != nullptr)
			return make_error<FunctionAlreadyExists>(name);

		auto function = m.getOrInsertFunction(name, getVoidType(m.getContext()));
		auto f = dyn_cast<llvm::Function>(function.getCallee());
		BasicBlock::Create(m.getContext(), "entry", f);
		f->setLinkage(internalLinkage);
		return f;
	}

	Type* builtInToLLVMType(LLVMContext& context, BultinSimTypes type)
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

		assert(false && "Unreachable");	 // NOLINT
		return nullptr;
	}

	ArrayType* typeToLLVMType(LLVMContext& context, const SimType& type)
	{
		auto baseType = builtInToLLVMType(context, type.getBuiltin());
		return ArrayType::get(baseType, type.flatSize());
	}

	Error simExpToGlobalVar(
			Module& module,
			StringRef name,
			const SimType& simType,
			GlobalValue::LinkageTypes linkage)
	{
		auto type = typeToLLVMType(module.getContext(), simType);
		auto varDecl = module.getOrInsertGlobal(name, type);
		if (varDecl == nullptr)
			return make_error<GlobalVariableCreationFailure>(name.str());

		auto global = dyn_cast<GlobalVariable>(varDecl);
		global->setLinkage(linkage);

		global->setInitializer(ConstantAggregateZero::get(type));
		return Error::success();
	}

	Expected<Value*> lowerReference(IRBuilder<>& builder, StringRef exp)
	{
		auto module = builder.GetInsertBlock()->getModule();
		auto global = module->getGlobalVariable(exp.str() + "_old", true);
		if (global == nullptr)
			return make_error<UnkownVariable>(exp.str());
		return global;
	}

	Expected<Value*> createTernaryOp(
			Function* function,
			IRBuilder<>& builder,
			Type* outType,
			std::function<Expected<Value*>(IRBuilder<>&)> condition,
			std::function<Expected<Value*>(IRBuilder<>&)> trueBlock,
			std::function<Expected<Value*>(IRBuilder<>&)> falseBlock)
	{
		auto& context = builder.getContext();
		auto conditionBlock = BasicBlock::Create(context, "if-condition", function);
		auto trueBranch = BasicBlock::Create(context, "if-trueBranch", function);
		auto falseBranch = BasicBlock::Create(context, "if-falseBranch", function);
		auto exit = BasicBlock::Create(context, "if-exit", function);

		builder.CreateBr(conditionBlock);
		builder.SetInsertPoint(conditionBlock);

		auto expConditionValue = condition(builder);
		if (!expConditionValue)
			return expConditionValue;
		auto conditionValue = *expConditionValue;
		assert(conditionValue->getType()->isIntegerTy());	 // NOLINT
		builder.CreateCondBr(conditionValue, trueBranch, falseBranch);

		builder.SetInsertPoint(trueBranch);
		auto truValue = trueBlock(builder);
		if (!truValue)
			return truValue;
		builder.CreateBr(exit);

		builder.SetInsertPoint(falseBranch);
		auto falseValue = falseBlock(builder);
		if (!falseValue)
			return falseValue;
		builder.CreateBr(exit);

		builder.SetInsertPoint(exit);

		auto phi = builder.CreatePHI(outType, 2);
		phi->addIncoming(*truValue, trueBranch);
		phi->addIncoming(*falseValue, falseBranch);
		return phi;
	}

	BasicBlock* createForCycle(
			Function* function,
			IRBuilder<>& builder,
			size_t iterationCount,
			std::function<void(IRBuilder<>&, Value*)> whileContent)
	{
		auto& context = builder.getContext();
		auto condition = BasicBlock::Create(context, "condition", function);
		auto loopBody = BasicBlock::Create(context, "loopBody", function);
		auto exit = BasicBlock::Create(context, "exit", function);

		auto unsignedInt = Type::getInt32Ty(context);

		// alocates iteration counter
		auto iterationCounter = builder.CreateAlloca(unsignedInt);
		makeConstantStore<int>(builder, 0, iterationCounter);

		// jump to condition bb
		builder.CreateBr(condition);

		// load counter
		builder.SetInsertPoint(condition);
		auto value = builder.CreateLoad(unsignedInt, iterationCounter);
		auto iterCmp = builder.CreateICmpEQ(
				value, ConstantInt::get(unsignedInt, iterationCount));

		// brach if equal to zero
		builder.CreateCondBr(iterCmp, exit, loopBody);

		builder.SetInsertPoint(loopBody);

		// populate body of the loop
		whileContent(builder, value);

		// load, reduce and store the counter
		value = builder.CreateLoad(unsignedInt, iterationCounter);
		auto reducedCounter =
				builder.CreateAdd(value, ConstantInt::get(unsignedInt, 1));
		builder.CreateStore(reducedCounter, iterationCounter);
		builder.CreateBr(condition);

		builder.SetInsertPoint(exit);

		return exit;
	}
}	 // namespace modelica
