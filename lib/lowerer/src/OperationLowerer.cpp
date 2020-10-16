#include "OperationLowerer.hpp"

#include "ExpLowerer.hpp"
#include "llvm/IR/Type.h"

using namespace modelica;
using namespace llvm;
using namespace std;

static bool isModelicaFloat(llvm::Type* t)
{
	return t->isFloatTy() or t->isDoubleTy();
}

template<>
Value* modelica::op<ModExpKind::negate>(IRBuilder<>& builder, Value* arg1)
{
	auto* boolType = IntegerType::getInt1Ty(builder.getContext());
	auto* zero = ConstantInt::get(boolType, 0);
	auto* type = arg1->getType();

	if (isModelicaFloat(type))
		return builder.CreateFNeg(arg1);

	if (type->isIntegerTy(32))
		return builder.CreateNeg(arg1);

	if (type->isIntegerTy(1))
		return builder.CreateICmpEQ(arg1, zero);
	assert(false && "unreachable");
	return nullptr;
}

template<>
Value* modelica::op<ModExpKind::add>(
		IRBuilder<>& builder, Value* arg1, Value* arg2)
{
	if (arg1->getType()->isIntegerTy())
		return builder.CreateAdd(arg1, arg2);

	if (isModelicaFloat(arg1->getType()))
		return builder.CreateFAdd(arg1, arg2);

	assert(false && "unreachable");
	return nullptr;
}

template<>
Value* modelica::op<ModExpKind::greaterThan>(
		IRBuilder<>& builder, Value* arg1, Value* arg2)
{
	if (arg1->getType()->isIntegerTy())
		return builder.CreateICmpSGT(arg1, arg2);

	if (isModelicaFloat(arg1->getType()))
		return builder.CreateFCmpOGT(arg1, arg2);

	assert(false && "unreachable");
	return nullptr;
}

template<>
Value* modelica::op<ModExpKind::lessEqual>(
		IRBuilder<>& builder, Value* arg1, Value* arg2)
{
	if (arg1->getType()->isIntegerTy())
		return builder.CreateICmpSLE(arg1, arg2);

	if (isModelicaFloat(arg1->getType()))
		return builder.CreateFCmpOLE(arg1, arg2);

	assert(false && "unreachable");
	return nullptr;
}

template<>
Value* modelica::op<ModExpKind::less>(
		IRBuilder<>& builder, Value* arg1, Value* arg2)
{
	if (arg1->getType()->isIntegerTy())
		return builder.CreateICmpSLT(arg1, arg2);

	if (isModelicaFloat(arg1->getType()))
		return builder.CreateFCmpOLT(arg1, arg2);

	assert(false && "unreachable");
	return nullptr;
}
template<>
Value* modelica::op<ModExpKind::greaterEqual>(
		IRBuilder<>& builder, Value* arg1, Value* arg2)
{
	if (arg1->getType()->isIntegerTy())
		return builder.CreateICmpSGE(arg1, arg2);

	if (isModelicaFloat(arg1->getType()))
		return builder.CreateFCmpOGE(arg1, arg2);

	assert(false && "unreachable");
	return nullptr;
}

template<>
Value* modelica::op<ModExpKind::equal>(
		IRBuilder<>& builder, Value* arg1, Value* arg2)
{
	if (arg1->getType()->isIntegerTy())
		return builder.CreateICmpEQ(arg1, arg2);

	if (isModelicaFloat(arg1->getType()))
		return builder.CreateFCmpOEQ(arg1, arg2);

	assert(false && "unreachable");
	return nullptr;
}
template<>
Value* modelica::op<ModExpKind::different>(
		IRBuilder<>& builder, Value* arg1, Value* arg2)
{
	if (arg1->getType()->isIntegerTy())
		return builder.CreateICmpNE(arg1, arg2);

	if (isModelicaFloat(arg1->getType()))
		return builder.CreateFCmpONE(arg1, arg2);

	assert(false && "unreachable");
	return nullptr;
}
template<>
Value* modelica::op<ModExpKind::sub>(
		IRBuilder<>& builder, Value* arg1, Value* arg2)
{
	if (arg1->getType()->isIntegerTy())
		return builder.CreateSub(arg1, arg2);

	if (isModelicaFloat(arg1->getType()))
		return builder.CreateFSub(arg1, arg2);

	assert(false && "unreachable");
	return nullptr;
}

template<>
Value* modelica::op<ModExpKind::mult>(
		IRBuilder<>& builder, Value* arg1, Value* arg2)
{
	if (arg1->getType()->isIntegerTy())
		return builder.CreateMul(arg1, arg2);

	if (isModelicaFloat(arg1->getType()))
		return builder.CreateFMul(arg1, arg2);

	assert(false && "unreachable");
	return nullptr;
}

template<>
Value* modelica::op<ModExpKind::divide>(
		IRBuilder<>& builder, Value* arg1, Value* arg2)
{
	if (arg1->getType()->isIntegerTy())
		return builder.CreateSDiv(arg1, arg2);

	if (isModelicaFloat(arg1->getType()))
		return builder.CreateFDiv(arg1, arg2);

	assert(false && "unreachable");
	return nullptr;
}

Expected<Value*> modelica::lowerAtOperation(
		LowererContext& info, const ModExp& exp)
{
	assert(exp.getKind() == ModExpKind::at);	// NOLINT
	auto leftHand = lowerExp(info, exp.getLeftHand());
	if (!leftHand)
		return leftHand.takeError();

	auto rightHand = lowerExp(info, exp.getRightHand());
	if (!rightHand)
		return rightHand.takeError();

	auto& builder = info.getBuilder();
	auto casted = builder.CreatePointerCast(
			*rightHand, Type::getInt32Ty(builder.getContext())->getPointerTo(0));
	auto index = builder.CreateLoad(casted);

	return info.getArrayElementPtr(*leftHand, index);
}

Expected<Value*> modelica::lowerNegate(LowererContext& info, const ModExp& arg1)
{
	auto lowered = lowerExp(info, arg1);
	if (!lowered)
		return lowered;

	auto& builder = info.getBuilder();
	auto exitVal = info.allocaModType(arg1.getModType());
	info.createForArrayElement(arg1.getModType(), [&](Value* iterationIndexes) {
		auto loaded = info.loadArrayElement(*lowered, iterationIndexes);
		auto calculated = op<ModExpKind::negate>(builder, loaded);
		info.storeToArrayElement(calculated, exitVal, iterationIndexes);
	});

	return exitVal;
}

Expected<Value*> modelica::lowerInduction(
		LowererContext& info, const ModExp& arg1)
{
	auto lowered = lowerExp(info, arg1);
	if (!lowered)
		return lowered;
	assert(
			info.getInductionVars() != nullptr &&
			"induction operation outside of for loop");

	constexpr size_t zero = 0;
	auto loaded = info.loadArrayElement(*lowered, zero);
	return info.getArrayElementPtr(info.getInductionVars(), loaded);
}
