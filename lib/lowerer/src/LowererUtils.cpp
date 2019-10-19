#include "LowererUtils.hpp"

#include "llvm/IR/IRBuilder.h"
#include "modelica/lowerer/Lowerer.hpp"
#include "modelica/model/ModConst.hpp"
#include "modelica/model/ModErrors.hpp"

using namespace llvm;
using namespace std;

namespace modelica
{
	constexpr auto internalLinkage = GlobalValue::LinkageTypes::InternalLinkage;

	Value* getArrayElementPtr(IRBuilder<>& bld, Value* arrayPtr, size_t index)
	{
		auto ptrType = dyn_cast<PointerType>(arrayPtr->getType());
		auto arrayType = dyn_cast<ArrayType>(ptrType->getContainedType(0));
		assert(index <= arrayType->getNumElements());	 // NOLINT

		auto intType = Type::getInt32Ty(bld.getContext());

		auto zero = ConstantInt::get(intType, 0);
		auto i = ConstantInt::get(intType, index);
		return getArrayElementPtr(bld, arrayPtr, i);
	}

	Value* getArrayElementPtr(IRBuilder<>& bld, Value* arrayPtr, Value* index)
	{
		auto intType = Type::getInt32Ty(bld.getContext());
		auto zero = ConstantInt::get(intType, 0);
		if (index->getType()->isIntegerTy())
		{
			SmallVector<Value*, 2> args = { zero, index };
			return bld.CreateGEP(arrayPtr, args);
		}

		auto ptrType = dyn_cast<PointerType>(index->getType());
		auto type = dyn_cast<ArrayType>(ptrType->getContainedType(0));
		Value* val = arrayPtr;
		for (size_t a = 0; a < type->getNumElements(); a++)
		{
			auto partialIndex = loadArrayElement(bld, index, a);
			SmallVector<Value*, 2> args = { zero, partialIndex };
			val = bld.CreateGEP(val, args);
		}
		return val;
	}

	void storeToArrayElement(
			IRBuilder<>& bld, Value* value, Value* arrayPtr, Value* index)
	{
		auto ptrToElem = getArrayElementPtr(bld, arrayPtr, index);
		bld.CreateStore(value, ptrToElem);
	}

	Value* loadArrayElement(IRBuilder<>& bld, Value* arrayPtr, size_t index)
	{
		auto ptrToElem = getArrayElementPtr(bld, arrayPtr, index);
		return bld.CreateLoad(ptrToElem);
	}

	Value* loadArrayElement(IRBuilder<>& bld, Value* arrayPtr, Value* index)
	{
		auto ptrToElem = getArrayElementPtr(bld, arrayPtr, index);
		return bld.CreateLoad(ptrToElem);
	}
	AllocaInst* allocaModType(IRBuilder<>& bld, const ModType& type)
	{
		auto llvmType = typeToLLVMType(bld.getContext(), type);
		return bld.CreateAlloca(llvmType);
	}

	void storeToArrayElement(
			IRBuilder<>& bld, Value* value, Value* arrayPtr, size_t index)
	{
		auto ptrToElem = getArrayElementPtr(bld, arrayPtr, index);
		bld.CreateStore(value, ptrToElem);
	}

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

	Type* builtInToLLVMType(LLVMContext& context, BultinModTypes type)
	{
		switch (type)
		{
			case BultinModTypes::INT:
				return Type::getInt32Ty(context);
			case BultinModTypes::BOOL:
				return Type::getInt1Ty(context);
			case BultinModTypes::FLOAT:
				return Type::getFloatTy(context);
		}

		assert(false && "Unreachable");	 // NOLINT
		return nullptr;
	}

	ArrayType* typeToLLVMType(LLVMContext& context, const ModType& type)
	{
		auto baseType = builtInToLLVMType(context, type.getBuiltin());
		Type* tp = baseType;
		for (auto dim = type.getDimensions().rbegin();
				 dim != type.getDimensions().rend();
				 dim++)
			tp = ArrayType::get(tp, *dim);
		return dyn_cast<ArrayType>(tp);
	}

	Error simExpToGlobalVar(
			Module& module,
			StringRef name,
			const ModType& simType,
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

	AllocaInst* getTypeDimensionsArray(IRBuilder<>& bld, const ModType& type)
	{
		auto allocaDim = type.getDimensionsCount() + 1;
		auto longType = IntegerType::getInt64Ty(bld.getContext());
		auto arrayType = ArrayType::get(longType, allocaDim);

		auto alloca = bld.CreateAlloca(arrayType);

		for (size_t a = 0; a < type.getDimensionsCount(); a++)
		{
			auto constant = ConstantInt::get(longType, type.getDimension(a));
			storeToArrayElement(bld, constant, alloca, a);
		}

		auto zero = ConstantInt::get(longType, 0);
		storeToArrayElement(bld, zero, alloca, allocaDim - 1);

		return alloca;
	}

	BasicBlock* createForCycle(
			Function* function,
			IRBuilder<>& builder,
			size_t iterationCountBegin,
			size_t iterationCountEnd,
			std::function<void(IRBuilder<>&, Value*)> whileContent)
	{
		auto& context = builder.getContext();
		auto condition = BasicBlock::Create(
				context, "condition " + to_string(iterationCountEnd), function);

		auto loopBody = BasicBlock::Create(
				context, "loopBody " + to_string(iterationCountEnd), function);
		auto exit = BasicBlock::Create(
				context, "exit " + to_string(iterationCountEnd), function);

		auto unsignedInt = Type::getInt32Ty(context);

		// alocates iteration counter
		auto iterationCounter = builder.CreateAlloca(unsignedInt);
		makeConstantStore<int>(builder, iterationCountBegin, iterationCounter);

		// jump to condition bb
		builder.CreateBr(condition);

		// load counter
		builder.SetInsertPoint(condition);
		auto value = builder.CreateLoad(unsignedInt, iterationCounter);
		auto iterCmp = builder.CreateICmpEQ(
				value, ConstantInt::get(unsignedInt, iterationCountEnd));

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

	static Value* valueArrayFromArrayOfValues(
			IRBuilder<>& bld, SmallVector<Value*, 3> vals)
	{
		if (vals.empty())
			return nullptr;
		assert(	 // NOLINT
				accumulate(
						begin(vals),
						end(vals),
						true,
						[type = vals[0]->getType()](auto old, auto next) {
							return old && next->getType() == type;
						}) &&
				"TYPE MISSMATCH IN VALS");

		auto type = ArrayType::get(vals[0]->getType(), vals.size());
		auto alloca = bld.CreateAlloca(type);

		size_t slot = 0;
		for (auto val : vals)
		{
			storeToArrayElement(bld, val, alloca, slot);
			slot++;
		}
		return alloca;
	}

	static BasicBlock* createdNestedForCycleImp(
			Function* function,
			IRBuilder<>& builder,
			ArrayRef<size_t> iterationsCountBegin,
			ArrayRef<size_t> iterationsCountEnd,
			std::function<void(IRBuilder<>&, Value*)> whileContent,
			SmallVector<Value*, 3>& indexes)
	{
		std::function<void(IRBuilder<>&, Value*)> loopBody;

		if (indexes.size() == iterationsCountBegin.size() - 1)
			loopBody = [&](auto& bld, auto value) {
				indexes.push_back(value);
				auto iters = valueArrayFromArrayOfValues(bld, indexes);
				whileContent(bld, iters);
			};
		else
			loopBody = [&](auto& bld, auto value) {
				indexes.push_back(value);
				createdNestedForCycleImp(
						function,
						bld,
						iterationsCountBegin,
						iterationsCountEnd,
						whileContent,
						indexes);
			};

		return createForCycle(
				function,
				builder,
				iterationsCountBegin[indexes.size()],
				iterationsCountEnd[indexes.size()],
				loopBody);
	}

	BasicBlock* createdNestedForCycle(
			Function* function,
			IRBuilder<>& builder,
			ArrayRef<size_t> iterationsCountBegin,
			ArrayRef<size_t> iterationsCountEnd,
			std::function<void(IRBuilder<>&, Value*)> whileContent)
	{
		SmallVector<Value*, 3> indexes;
		return createdNestedForCycleImp(
				function,
				builder,
				iterationsCountBegin,
				iterationsCountEnd,
				whileContent,
				indexes);
	}
}	 // namespace modelica
