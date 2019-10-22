#include "ExpLowerer.hpp"

#include "CallLowerer.hpp"
#include "OperationLowerer.hpp"
#include "modelica/model/ModErrors.hpp"

using namespace std;
using namespace llvm;
using namespace modelica;

Type* flatPtrType(PointerType* t)
{
	size_t flatSize = 1;
	Type* arrayType = t->getContainedType(0);
	while (isa<ArrayType>(arrayType))
	{
		auto t = dyn_cast<ArrayType>(arrayType);
		flatSize *= t->getNumElements();
		arrayType = t->getContainedType(0);
	}

	return ArrayType::get(arrayType, flatSize)->getPointerTo(0);
}

template<ModExpKind kind>
Expected<Value*> lowerMemberWiseOp(LowererContext& info, const ModExp& exp)
{
	assert(exp.getKind() == kind);	// NOLINT
	auto left = lowerExp(info, exp.getLeftHand());
	if (!left)
		return left;

	auto right = lowerExp(info, exp.getRightHand());
	if (!right)
		return right;

	auto exitVal = info.allocaModType(exp.getModType());
	info.createForArrayElement(
			exp.getLeftHand().getModType(), [&](Value* iterationIndexes) {
				auto leftEl = info.loadArrayElement(*left, iterationIndexes);
				auto rightEl = info.loadArrayElement(*right, iterationIndexes);
				auto calculated = op<kind>(info.getBuilder(), leftEl, rightEl);
				info.storeToArrayElement(calculated, exitVal, iterationIndexes);
			});

	return exitVal;
}

template<typename T>
Expected<AllocaInst*> lowerConstantTyped(
		LowererContext& cont, const ModConst<T>& constant, const ModType& type)
{
	if (constant.size() != type.flatSize())
		return make_error<TypeConstantSizeMissMatch>(constant, type);

	auto& builder = cont.getBuilder();
	auto alloca = cont.allocaModType(type);
	auto castedAlloca =
			builder.CreatePointerCast(alloca, flatPtrType(alloca->getType()));

	for (size_t i = 0; i < constant.size(); i++)
		cont.storeConstantToArrayElement<T>(constant.get(i), castedAlloca, i);
	return alloca;
}

static Expected<Value*> lowerTernary(LowererContext& info, const ModExp& exp)
{
	assert(exp.isTernary());	// NOLINT

	assert(
			exp.getCondition().getModType() ==
			ModType(BultinModTypes::BOOL));	 // NOLINT
	auto type = exp.getLeftHand().getModType();
	auto llvmType = typeToLLVMType(info.getContext(), type)->getPointerTo();

	return info.createTernaryOp(
			llvmType,
			[&]() { return lowerExp(info, exp.getCondition()); },
			[&]() { return lowerExp(info, exp.getLeftHand()); },
			[&]() { return lowerExp(info, exp.getRightHand()); });
}

static Expected<Value*> lowerBinaryOp(LowererContext& info, const ModExp& exp)
{
	assert(exp.isBinary());											// NOLINT
	assert(exp.areSubExpressionCompatibles());	// NOLINT
	const auto& left = exp.getLeftHand();
	const auto& right = exp.getRightHand();
	if (exp.getKind() == ModExpKind::add)
		return lowerMemberWiseOp<ModExpKind::add>(info, exp);
	if (exp.getKind() == ModExpKind::sub)
		return lowerMemberWiseOp<ModExpKind::sub>(info, exp);
	if (exp.getKind() == ModExpKind::mult)
		return lowerMemberWiseOp<ModExpKind::mult>(info, exp);
	if (exp.getKind() == ModExpKind::divide)
		return lowerMemberWiseOp<ModExpKind::divide>(info, exp);
	if (exp.getKind() == ModExpKind::greaterEqual)
		return lowerMemberWiseOp<ModExpKind::greaterEqual>(info, exp);
	if (exp.getKind() == ModExpKind::greaterThan)
		return lowerMemberWiseOp<ModExpKind::greaterThan>(info, exp);
	if (exp.getKind() == ModExpKind::lessEqual)
		return lowerMemberWiseOp<ModExpKind::lessEqual>(info, exp);
	if (exp.getKind() == ModExpKind::less)
		return lowerMemberWiseOp<ModExpKind::less>(info, exp);
	if (exp.getKind() == ModExpKind::equal)
		return lowerMemberWiseOp<ModExpKind::equal>(info, exp);
	if (exp.getKind() == ModExpKind::different)
		return lowerMemberWiseOp<ModExpKind::different>(info, exp);
	if (exp.getKind() == ModExpKind::at)
		return lowerAtOperation(info, exp);
	assert(false && "unreachable");	 // NOLINT
	return nullptr;
}

static Expected<Value*> lowerUnaryOp(LowererContext& info, const ModExp& exp)
{
	assert(exp.isUnary());	// NOLINT

	if (exp.getKind() == ModExpKind::negate)
		return lowerNegate(info, exp.getLeftHand());
	if (exp.getKind() == ModExpKind::induction)
		return lowerInduction(info, exp.getLeftHand());

	assert(false && "unreachable");	 // NOLINT
	return nullptr;
}

static Expected<Value*> lowerOperation(LowererContext& info, const ModExp& exp)
{
	assert(exp.isOperation());	// NOLINT

	if (ModExpKind::zero == exp.getKind())
	{
		ModType type(BultinModTypes::INT);
		IntModConst constant(0);
		return lowerConstantTyped(info, constant, type);
	}

	if (exp.isTernary())
		return lowerTernary(info, exp);

	if (exp.isUnary())
		return lowerUnaryOp(info, exp);

	if (exp.isBinary())
		return lowerBinaryOp(info, exp);
	assert(false && "Unreachable");	 // NOLINT
	return nullptr;
}

static Expected<Value*> uncastedLowerExp(
		LowererContext& info, const ModExp& exp)
{
	if (!exp.areSubExpressionCompatibles())
		return make_error<TypeMissMatch>(exp);

	if (exp.isConstant())
		return lowerConstant(info, exp);

	if (exp.isReference())
		return info.lowerReference(exp.getReference());

	if (exp.isCall())
		return lowerCall(info, exp.getCall());

	auto opRes = lowerOperation(info, exp);
	if (!opRes)
		return opRes;

	Type* operationType =
			typeToLLVMType(info.getContext(), exp.getOperationReturnType());
	operationType = operationType->getPointerTo();
	return info.getBuilder().CreatePointerCast(*opRes, operationType);
}

static Value* castSingleElem(IRBuilder<>& builder, Value* val, Type* type)
{
	auto floatType = Type::getFloatTy(builder.getContext());
	auto intType = Type::getInt32Ty(builder.getContext());
	auto boolType = Type::getInt1Ty(builder.getContext());

	auto constantZero = ConstantInt::get(intType, 0);

	if (type == floatType)
		return builder.CreateSIToFP(val, floatType);

	if (type == intType)
	{
		if (val->getType() == floatType)
			return builder.CreateFPToSI(val, intType);

		return builder.CreateIntCast(val, intType, true);
	}

	if (val->getType() == floatType)
		return builder.CreateFPToSI(val, boolType);

	return builder.CreateTrunc(val, boolType);
}

static Expected<Value*> castReturnValue(
		LowererContext& info, Value* val, const ModType& type)
{
	auto ptrArrayType = dyn_cast<PointerType>(val->getType());
	auto arrayType = dyn_cast<ArrayType>(ptrArrayType->getContainedType(0));

	auto srcType = modTypeFromLLVMType(arrayType);
	assert(srcType.getDimensions() == type.getDimensions());	// NOLINT

	if (srcType.getBuiltin() == type.getBuiltin())
		return val;

	auto alloca = info.allocaModType(type);

	info.createForArrayElement(type, [&](Value* inductionsVar) {
		auto loadedElem = info.loadArrayElement(val, inductionsVar);
		auto singleDestType =
				builtInToLLVMType(info.getContext(), type.getBuiltin());
		Value* casted =
				castSingleElem(info.getBuilder(), loadedElem, singleDestType);
		info.storeToArrayElement(casted, alloca, inductionsVar);
	});

	return alloca;
}

namespace modelica
{
	Expected<AllocaInst*> lowerConstant(
			LowererContext& context, const ModExp& exp)
	{
		if (exp.isConstant<int>())
			return lowerConstantTyped<int>(
					context, exp.getConstant<int>(), exp.getModType());
		if (exp.isConstant<float>())
			return lowerConstantTyped<float>(
					context, exp.getConstant<float>(), exp.getModType());
		if (exp.isConstant<bool>())
			return lowerConstantTyped<bool>(
					context, exp.getConstant<bool>(), exp.getModType());

		assert(false && "unreachable");	 // NOLINT
		return nullptr;
	}

	Expected<Value*> lowerExp(LowererContext& info, const ModExp& exp)
	{
		auto retVal = uncastedLowerExp(info, exp);
		if (!retVal)
			return retVal;

		return castReturnValue(info, *retVal, exp.getModType());
	}
}	 // namespace modelica
