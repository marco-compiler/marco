#include "OperationLowerer.hpp"

#include "ExpLowerer.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;

template<>
Value* modelica::op<ModExpKind::negate>(IRBuilder<>& builder, Value* arg1)
{
	auto boolType = IntegerType::getInt1Ty(builder.getContext());
	auto zero = ConstantInt::get(boolType, 0);
	auto type = arg1->getType();
	if (type->isFloatTy())
		return builder.CreateFNeg(arg1);

	if (type->isIntegerTy(32))
		return builder.CreateNeg(arg1);

	if (type->isIntegerTy(1))
		return builder.CreateICmpEQ(arg1, zero);
	assert(false && "unreachable");	 // NOLINT
	return nullptr;
}

template<>
Value* modelica::op<ModExpKind::add>(
		IRBuilder<>& builder, Value* arg1, Value* arg2)
{
	if (arg1->getType()->isIntegerTy())
		return builder.CreateAdd(arg1, arg2);

	if (arg1->getType()->isFloatTy())
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

	if (arg1->getType()->isFloatTy())
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

	if (arg1->getType()->isFloatTy())
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

	if (arg1->getType()->isFloatTy())
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

	if (arg1->getType()->isFloatTy())
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

	if (arg1->getType()->isFloatTy())
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

	if (arg1->getType()->isFloatTy())
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

	if (arg1->getType()->isFloatTy())
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

	if (arg1->getType()->isFloatTy())
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

	if (arg1->getType()->isFloatTy())
		return builder.CreateFDiv(arg1, arg2);

	assert(false && "unreachable");
	return nullptr;
}

Expected<Value*> modelica::lowerAtOperation(
		LoweringInfo& info, const ModExp& exp, bool loadOld)
{
	assert(exp.getKind() == ModExpKind::at);	// NOLINT
	auto leftHand = lowerExp(info, exp.getLeftHand(), loadOld);
	if (!leftHand)
		return leftHand.takeError();

	auto rightHand = lowerExp(info, exp.getRightHand(), loadOld);
	if (!rightHand)
		return rightHand.takeError();

	auto casted = info.builder.CreatePointerCast(
			*rightHand, Type::getInt32Ty(info.builder.getContext())->getPointerTo(0));
	auto index = info.builder.CreateLoad(casted);

	return getArrayElementPtr(info.builder, *leftHand, index);
}

Expected<Value*> modelica::lowerNegate(
		LoweringInfo& info, const ModExp& arg1, bool loadOld)
{
	auto lowered = lowerExp(info, arg1, loadOld);
	if (!lowered)
		return lowered;

	auto exitVal = allocaModType(info.builder, arg1.getModType());
	createForArrayElement(
			info.function,
			info.builder,
			arg1.getModType(),
			[&lowered, exitVal](auto& bld, Value* iterationIndexes) {
				auto loaded = loadArrayElement(bld, *lowered, iterationIndexes);
				auto calculated = op<ModExpKind::negate>(bld, loaded);
				storeToArrayElement(bld, calculated, exitVal, iterationIndexes);
			});

	return exitVal;
}

Expected<Value*> modelica::lowerInduction(
		LoweringInfo& info, const ModExp& arg1, bool loadOld)
{
	auto lowered = lowerExp(info, arg1, loadOld);
	if (!lowered)
		return lowered;
	assert(
			info.inductionsVars != nullptr &&
			"induction operation outside of for loop");

	constexpr size_t zero = 0;
	auto loaded = loadArrayElement(info.builder, *lowered, zero);
	return getArrayElementPtr(info.builder, info.inductionsVars, loaded);
}
