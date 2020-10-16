#include "modelica/lowerer/LowererUtils.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/IRBuilder.h"
#include "modelica/lowerer/Lowerer.hpp"
#include "modelica/model/Assigment.hpp"
#include "modelica/model/ModConst.hpp"
#include "modelica/model/ModErrors.hpp"
#include "modelica/utils/Interval.hpp"

using namespace llvm;
using namespace std;
using namespace modelica;

constexpr auto internalLinkage = GlobalValue::LinkageTypes::InternalLinkage;

Value* LowererContext::getArrayElementPtr(Value* arrayPtr, size_t index)
{
	auto* ptrType = dyn_cast<PointerType>(arrayPtr->getType());
	auto* arrayType = dyn_cast<ArrayType>(ptrType->getContainedType(0));
	assert(index <= arrayType->getNumElements());	 // NOLINT

	auto* intType = Type::getInt32Ty(builder.getContext());

	auto* zero = ConstantInt::get(intType, 0);
	auto* i = ConstantInt::get(intType, index);
	return getArrayElementPtr(arrayPtr, i);
}

Value* LowererContext::getArrayElementPtr(Value* arrayPtr, Value* index)
{
	auto* intType = Type::getInt32Ty(builder.getContext());
	auto* zero = ConstantInt::get(intType, 0);
	if (index->getType()->isIntegerTy())
	{
		SmallVector<Value*, 2> args = { zero, index };
		return builder.CreateGEP(arrayPtr, args);
	}

	auto* ptrType = dyn_cast<PointerType>(index->getType());
	auto* type = dyn_cast<ArrayType>(ptrType->getContainedType(0));
	Value* val = arrayPtr;
	for (size_t a = 0; a < type->getNumElements(); a++)
	{
		auto* partialIndex = loadArrayElement(index, a);
		SmallVector<Value*, 2> args = { zero, partialIndex };
		val = builder.CreateGEP(val, args);
	}
	return val;
}

void LowererContext::storeToArrayElement(
		Value* value, Value* arrayPtr, Value* index)
{
	auto ptrToElem = getArrayElementPtr(arrayPtr, index);
	builder.CreateStore(value, ptrToElem);
}

Value* LowererContext::loadArrayElement(Value* arrayPtr, size_t index)
{
	auto ptrToElem = getArrayElementPtr(arrayPtr, index);
	return builder.CreateLoad(ptrToElem);
}

Value* LowererContext::loadArrayElement(Value* arrayPtr, Value* index)
{
	auto ptrToElem = getArrayElementPtr(arrayPtr, index);
	return builder.CreateLoad(ptrToElem);
}
AllocaInst* LowererContext::allocaModType(const ModType& type)
{
	auto llvmType = typeToLLVMType(getContext(), type, useDouble);
	return builder.CreateAlloca(llvmType);
}

void LowererContext::storeToArrayElement(
		Value* value, Value* arrayPtr, size_t index)
{
	auto ptrToElem = getArrayElementPtr(arrayPtr, index);
	builder.CreateStore(value, ptrToElem);
}

Type* modelica::builtInToLLVMType(
		LLVMContext& context, BultinModTypes type, bool useDouble)
{
	switch (type)
	{
		case BultinModTypes::INT:
			return Type::getInt32Ty(context);
		case BultinModTypes::BOOL:
			return Type::getInt1Ty(context);
		case BultinModTypes::FLOAT:
			return useDouble ? Type::getDoubleTy(context) : Type::getFloatTy(context);
	}

	assert(false && "Unreachable");	 // NOLINT
	return nullptr;
}

ArrayType* modelica::typeToLLVMType(
		LLVMContext& context, const ModType& type, bool useDouble)
{
	auto* baseType = builtInToLLVMType(context, type.getBuiltin(), useDouble);
	Type* tp = baseType;
	for (auto dim = type.getDimensions().rbegin();
			 dim != type.getDimensions().rend();
			 dim++)
		tp = ArrayType::get(tp, *dim);
	return dyn_cast<ArrayType>(tp);
}

Expected<Value*> LowererContext::lowerReference(StringRef exp)
{
	auto* module = builder.GetInsertBlock()->getModule();
	auto global = module->getGlobalVariable(exp.str(), true);
	if (global == nullptr)
		return make_error<UnkownVariable>(exp.str());
	return global;
}

Expected<Value*> LowererContext::createTernaryOp(
		Type* outType,
		std::function<Expected<Value*>()> condition,
		std::function<Expected<Value*>()> trueBlock,
		std::function<Expected<Value*>()> falseBlock)
{
	auto& context = builder.getContext();
	auto conditionBlock = BasicBlock::Create(context, "if-condition", function);
	auto trueBranch = BasicBlock::Create(context, "if-trueBranch", function);
	auto falseBranch = BasicBlock::Create(context, "if-falseBranch", function);
	auto exit = BasicBlock::Create(context, "if-exit", function);

	builder.CreateBr(conditionBlock);
	builder.SetInsertPoint(conditionBlock);

	auto expConditionValue = condition();
	if (!expConditionValue)
		return expConditionValue;
	auto conditionPtr = *expConditionValue;
	size_t zero = 0;
	auto conditionValue = loadArrayElement(conditionPtr, zero);
	assert(conditionValue->getType()->isIntegerTy());	 // NOLINT
	builder.CreateCondBr(conditionValue, trueBranch, falseBranch);

	builder.SetInsertPoint(trueBranch);
	auto truValue = trueBlock();
	if (!truValue)
		return truValue;
	builder.CreateBr(exit);

	builder.SetInsertPoint(falseBranch);
	auto falseValue = falseBlock();
	if (!falseValue)
		return falseValue;
	builder.CreateBr(exit);

	builder.SetInsertPoint(exit);

	auto* phi = builder.CreatePHI(outType, 2);
	phi->addIncoming(*truValue, trueBranch);
	phi->addIncoming(*falseValue, falseBranch);
	return phi;
}

AllocaInst* LowererContext::getTypeDimensionsArray(const ModType& type)
{
	auto allocaDim = type.getDimensionsCount() + 1;
	auto* longType = IntegerType::getInt64Ty(builder.getContext());
	auto* arrayType = ArrayType::get(longType, allocaDim);

	auto* alloca = builder.CreateAlloca(arrayType);

	for (size_t a = 0; a < type.getDimensionsCount(); a++)
	{
		auto* constant = ConstantInt::get(longType, type.getDimension(a));
		storeToArrayElement(constant, alloca, a);
	}

	auto* zero = ConstantInt::get(longType, 0);
	storeToArrayElement(zero, alloca, allocaDim - 1);

	return alloca;
}

BasicBlock* LowererContext::createForCycle(
		Interval var, std::function<void(Value*)> whileContent, bool inverseRange)
{
	const size_t start = inverseRange ? var.max() - 1 : var.min();
	const size_t end = inverseRange ? var.min() - 1 : var.max();
	const int increment = inverseRange ? -1 : 1;
	auto& context = builder.getContext();
	auto condition = BasicBlock::Create(context, "condition ", function);

	auto loopBody = BasicBlock::Create(context, "loopBody ", function);
	auto exit = BasicBlock::Create(context, "exit ", function);

	auto unsignedInt = Type::getInt32Ty(context);

	// alocates iteration counter
	auto iterationCounter = builder.CreateAlloca(unsignedInt);
	makeConstantStore<int>(start, iterationCounter);

	// jump to condition bb
	builder.CreateBr(condition);

	// load counter
	builder.SetInsertPoint(condition);
	auto value = builder.CreateLoad(unsignedInt, iterationCounter);
	auto iterCmp =
			builder.CreateICmpEQ(value, ConstantInt::get(unsignedInt, end));

	// brach if equal to zero
	builder.CreateCondBr(iterCmp, exit, loopBody);

	builder.SetInsertPoint(loopBody);

	// populate body of the loop
	whileContent(value);

	// load, reduce and store the counter
	value = builder.CreateLoad(unsignedInt, iterationCounter);
	auto reducedCounter =
			builder.CreateAdd(value, ConstantInt::get(unsignedInt, increment));
	builder.CreateStore(reducedCounter, iterationCounter);
	builder.CreateBr(condition);

	builder.SetInsertPoint(exit);

	return exit;
}

Value* LowererContext::valueArrayFromArrayOfValues(SmallVector<Value*, 3> vals)
{
	if (vals.empty())
		return nullptr;

	auto type = ArrayType::get(vals[0]->getType(), vals.size());
	auto alloca = builder.CreateAlloca(type);

	size_t slot = 0;
	for (auto val : vals)
	{
		storeToArrayElement(val, alloca, slot);
		slot++;
	}
	return alloca;
}

BasicBlock* LowererContext::createdNestedForCycleImp(
		const OrderedMultiDimInterval& iterationsCountBegin,
		std::function<void(Value*)> whileContent,
		SmallVector<Value*, 3>& indexes)
{
	return createForCycle(
			iterationsCountBegin[indexes.size()],
			[&](Value* value) {
				indexes.push_back(value);
				if (indexes.size() != iterationsCountBegin.dimensions())
					createdNestedForCycleImp(iterationsCountBegin, whileContent, indexes);
				else
					whileContent(valueArrayFromArrayOfValues(indexes));
			},
			iterationsCountBegin.isBackward());
}

BasicBlock* LowererContext::createdNestedForCycle(
		const OrderedMultiDimInterval& iterationsCountBegin,
		std::function<void(Value*)> whileContent)
{
	SmallVector<Value*, 3> indexes;
	return createdNestedForCycleImp(iterationsCountBegin, whileContent, indexes);
}

BasicBlock* LowererContext::createdNestedForCycle(
		ArrayRef<size_t> iterationsCountEnd, std::function<void(Value*)> body)
{
	SmallVector<Interval, 2> inducts;
	for (auto& v : iterationsCountEnd)
		inducts.emplace_back(0, v);

	OrderedMultiDimInterval interval(inducts);
	return createdNestedForCycle(interval, body);
}

BasicBlock* LowererContext::createForArrayElement(
		const ModType& type, std::function<void(Value*)> body)
{
	return createdNestedForCycle(type.getDimensions(), body);
}

BultinModTypes modelica::builtinTypeFromLLVMType(Type* tp)
{
	if (tp->isIntegerTy(32))
		return BultinModTypes::INT;
	if (tp->isIntegerTy(1))
		return BultinModTypes::BOOL;
	if (tp->isFloatTy() or tp->isDoubleTy())
		return BultinModTypes::FLOAT;
	assert(false && "unreachable");
	return BultinModTypes::INT;
}

ModType modelica::modTypeFromLLVMType(ArrayType* type)
{
	SmallVector<size_t, 3> dims;
	Type* t = type;
	while (isa<ArrayType>(t))
	{
		auto tp = dyn_cast<ArrayType>(t);
		dims.push_back(tp->getNumElements());
		t = tp->getContainedType(0);
	}
	return ModType(builtinTypeFromLLVMType(t), move(dims));
}
