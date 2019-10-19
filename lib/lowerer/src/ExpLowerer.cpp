#include "ExpLowerer.hpp"

#include "CallLowerer.hpp"
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

template<typename T>
Expected<AllocaInst*> lowerConstantTyped(
		IRBuilder<>& builder, const ModConst<T>& constant, const ModType& type)
{
	if (constant.size() != type.flatSize())
		return make_error<TypeConstantSizeMissMatch>(constant, type);

	auto alloca = allocaModType(builder, type);
	auto castedAlloca =
			builder.CreatePointerCast(alloca, flatPtrType(alloca->getType()));

	for (size_t i = 0; i < constant.size(); i++)
		storeConstantToArrayElement<T>(builder, constant.get(i), castedAlloca, i);
	return alloca;
}

template<int argumentsSize>
static AllocaInst* elementWiseOperation(
		LoweringInfo& info,
		ArrayRef<Value*>& args,
		const ModType& operationOutType,
		std::function<Value*(IRBuilder<>&, ArrayRef<Value*>)> operation)
{
	auto alloca = allocaModType(info.builder, operationOutType);

	const auto forBody = [alloca, &operation, &args](
													 IRBuilder<>& bld, Value* index) {
		SmallVector<Value*, argumentsSize> arguments;

		for (auto arg : args)
			arguments.push_back(loadArrayElement(bld, arg, index));

		auto outVal = operation(bld, arguments);

		storeToArrayElement(bld, outVal, alloca, index);
	};

	SmallVector<size_t, 3> zeros;
	for (auto& v : operationOutType.getDimensions())
		zeros.push_back(0);
	createdNestedForCycle(
			info.function,
			info.builder,
			zeros,
			operationOutType.getDimensions(),
			forBody);

	return alloca;
}

static Value* createFloatSingleBynaryOp(
		LoweringInfo& info, ArrayRef<Value*> args, ModExpKind kind)
{
	assert(args[0]->getType()->isFloatTy());					// NOLINT
	assert(ModExp::Operation::arityOfOp(kind) == 2);	// NOLINT
	IRBuilder<>& builder = info.builder;
	switch (kind)
	{
		case ModExpKind::add:
			return builder.CreateFAdd(args[0], args[1]);
		case ModExpKind::sub:
			return builder.CreateFSub(args[0], args[1]);
		case ModExpKind::mult:
			return builder.CreateFMul(args[0], args[1]);
		case ModExpKind::divide:
			return builder.CreateFDiv(args[0], args[1]);
		case ModExpKind::equal:
			return builder.CreateFCmpOEQ(args[0], args[1]);
		case ModExpKind::different:
			return builder.CreateFCmpONE(args[0], args[1]);
		case ModExpKind::greaterEqual:
			return builder.CreateFCmpOGE(args[0], args[1]);
		case ModExpKind::greaterThan:
			return builder.CreateFCmpOGT(args[0], args[1]);
		case ModExpKind::lessEqual:
			return builder.CreateFCmpOLE(args[0], args[1]);
		case ModExpKind::less:
			return builder.CreateFCmpOLT(args[0], args[1]);
		case ModExpKind::module: {
			// TODO
			assert(false && "module unsupported");	// NOLINT
			return nullptr;
		}
		case ModExpKind::elevation: {
			// TODO
			assert(false && "powerof unsupported");	 // NOLINT
			return nullptr;
		}
		case ModExpKind::at:
		case ModExpKind::zero:
		case ModExpKind::negate:
		case ModExpKind::conditional:
		case ModExpKind::induction:
			assert(false && "unreachable");	 // NOLINT
			return nullptr;
	}
	assert(false && "unreachable");	 // NOLINT
	return nullptr;
}

static Value* createIntSingleBynaryOp(
		LoweringInfo& info, ArrayRef<Value*> args, ModExpKind kind)
{
	assert(args[0]->getType()->isIntegerTy());				// NOLINT
	assert(ModExp::Operation::arityOfOp(kind) == 2);	// NOLINT
	IRBuilder<>& builder = info.builder;
	switch (kind)
	{
		case ModExpKind::add:
			return builder.CreateAdd(args[0], args[1]);
		case ModExpKind::sub:
			return builder.CreateSub(args[0], args[1]);
		case ModExpKind::mult:
			return builder.CreateMul(args[0], args[1]);
		case ModExpKind::divide:
			return builder.CreateSDiv(args[0], args[1]);
		case ModExpKind::equal:
			return builder.CreateICmpEQ(args[0], args[1]);
		case ModExpKind::different:
			return builder.CreateICmpNE(args[0], args[1]);
		case ModExpKind::greaterEqual:
			return builder.CreateICmpSGE(args[0], args[1]);
		case ModExpKind::greaterThan:
			return builder.CreateICmpSGT(args[0], args[1]);
		case ModExpKind::lessEqual:
			return builder.CreateICmpSLE(args[0], args[1]);
		case ModExpKind::less:
			return builder.CreateICmpSLT(args[0], args[1]);
		case ModExpKind::module: {
			// TODO
			assert(false && "module unsupported");	// NOLINT
			return nullptr;
		}
		case ModExpKind::elevation: {
			// TODO
			assert(false && "powerof unsupported");	 // NOLINT
			return nullptr;
		}
		case ModExpKind::at:
		case ModExpKind::zero:
		case ModExpKind::induction:
		case ModExpKind::negate:
		case ModExpKind::conditional:
			assert(false && "unreachable");	 // NOLINT
			return nullptr;
	}
	assert(false && "unreachable");	 // NOLINT
	return nullptr;
}

static Value* createSingleBynaryOP(
		LoweringInfo& info, ArrayRef<Value*> args, ModExpKind kind)
{
	assert(ModExp::Operation::arityOfOp(kind) == 2);	// NOLINT
	auto type = args[0]->getType();
	if (type->isIntegerTy())
		return createIntSingleBynaryOp(info, args, kind);
	if (type->isFloatTy())
		return createFloatSingleBynaryOp(info, args, kind);

	assert(false && "unreachable");	 // NOLINT
	return nullptr;
}

static Value* createSingleUnaryOp(
		LoweringInfo& info, ArrayRef<Value*> args, ModExpKind kind)
{
	IRBuilder<>& builder = info.builder;
	auto intType = IntegerType::getInt32Ty(builder.getContext());
	auto boolType = IntegerType::getInt1Ty(builder.getContext());
	auto zero = ConstantInt::get(boolType, 0);
	switch (kind)
	{
		case ModExpKind::negate:
			if (args[0]->getType()->isFloatTy())
				return builder.CreateFNeg(args[0]);

			if (args[0]->getType() == intType)
				return builder.CreateNeg(args[0]);

			if (args[0]->getType() == boolType)
				return builder.CreateICmpEQ(args[0], zero);
			assert(false && "unreachable");	 // NOLINT
			return nullptr;

		case ModExpKind::induction:
			return loadArrayElement(builder, info.inductionsVars, args[0]);

			assert(false && "unreachable");	 // NOLINT
		default:
			assert(false && "unreachable");	 // NOLINT
			return nullptr;
	}
	assert(false && "unreachable");	 // NOLINT
	return nullptr;
}

template<size_t arity>
static Expected<AllocaInst*> lowerUnOrBinOp(
		LoweringInfo& info, const ModExp& exp, ArrayRef<Value*> subExp)
{
	static_assert(arity < 3 && arity > 0, "cannot lower op with this arity");
	assert(exp.getArity() == arity);	// NOLINT;
	assert(subExp.size() == arity);		// NOLINT
	const auto binaryOp = [type = exp.getKind(), &info](
														auto& builder, auto args) {
		LoweringInfo newInfo = {
			builder, info.module, info.function, info.inductionsVars
		};
		if constexpr (arity == 1)
			return createSingleUnaryOp(newInfo, args, type);
		else
			return createSingleBynaryOP(newInfo, args, type);
	};

	auto opType = exp.getOperationReturnType();
	return elementWiseOperation<arity>(info, subExp, opType, binaryOp);
}

static Expected<Value*> lowerTernary(LoweringInfo& info, const ModExp& exp)
{
	assert(exp.isTernary());	// NOLINT
	const auto leftHandLowerer = [&exp, &info](IRBuilder<>& builder) {
		LoweringInfo newInfo = {
			builder, info.module, info.function, info.inductionsVars
		};
		return lowerExp(newInfo, exp.getLeftHand());
	};
	const auto rightHandLowerer = [&exp, &info](IRBuilder<>& builder) {
		LoweringInfo newInfo = {
			builder, info.module, info.function, info.inductionsVars
		};
		return lowerExp(newInfo, exp.getRightHand());
	};
	const auto conditionLowerer =
			[&exp, &info](IRBuilder<>& builder) -> Expected<Value*> {
		auto& condition = exp.getCondition();
		assert(condition.getModType() == ModType(BultinModTypes::BOOL));	// NOLINT
		LoweringInfo newInfo = {
			builder, info.module, info.function, info.inductionsVars
		};
		auto ptrToCond = lowerExp(newInfo, condition);
		if (!ptrToCond)
			return ptrToCond;

		size_t zero = 0;
		return loadArrayElement(builder, *ptrToCond, zero);
	};

	auto type = exp.getLeftHand().getModType();
	auto llvmType =
			typeToLLVMType(info.builder.getContext(), type)->getPointerTo(0);

	return createTernaryOp(
			info.function,
			info.builder,
			llvmType,
			conditionLowerer,
			leftHandLowerer,
			rightHandLowerer);
}

static Expected<Value*> lowerOperation(LoweringInfo& info, const ModExp& exp)
{
	assert(exp.isOperation());	// NOLINT

	if (ModExpKind::zero == exp.getKind())
	{
		ModType type(BultinModTypes::INT);
		IntModConst constant(0);
		return lowerConstantTyped(info.builder, constant, type);
	}

	if (exp.isTernary())
		return lowerTernary(info, exp);

	if (exp.isUnary())
	{
		SmallVector<Value*, 1> values;
		auto subexp = lowerExp(info, exp.getLeftHand());
		if (!subexp)
			return subexp.takeError();
		values.push_back(move(*subexp));

		return lowerUnOrBinOp<1>(info, exp, values);
	}
	if (exp.getKind() == ModExpKind::at)
	{
		auto leftHand = lowerExp(info, exp.getLeftHand());
		if (!leftHand)
			return leftHand.takeError();

		auto rightHand = lowerExp(info, exp.getRightHand());
		if (!rightHand)
			return rightHand.takeError();

		auto casted = info.builder.CreatePointerCast(
				*rightHand,
				Type::getInt32Ty(info.builder.getContext())->getPointerTo(0));
		auto index = info.builder.CreateLoad(casted);

		return getArrayElementPtr(info.builder, *leftHand, index);
	}

	if (exp.isBinary())
	{
		SmallVector<Value*, 2> values;
		auto leftHand = lowerExp(info, exp.getLeftHand());
		if (!leftHand)
			return leftHand.takeError();
		values.push_back(move(*leftHand));

		auto rightHand = lowerExp(info, exp.getRightHand());
		if (!rightHand)
			return rightHand.takeError();
		values.push_back(move(*rightHand));

		return lowerUnOrBinOp<2>(info, exp, values);
	}
	assert(false && "Unreachable");	 // NOLINT
	return nullptr;
}

static Expected<Value*> uncastedLowerExp(LoweringInfo& info, const ModExp& exp)
{
	if (!exp.areSubExpressionCompatibles())
		return make_error<TypeMissMatch>(exp);

	if (exp.isConstant())
		return lowerConstant(info.builder, exp);

	if (exp.isReference())
		return lowerReference(info.builder, exp.getReference());

	if (exp.isCall())
		return lowerCall(info, exp.getCall());

	return lowerOperation(info, exp);
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

static bool castable(ArrayType* type, const ModType& dest)
{
	SmallVector<size_t, 3> dims;
	Type* t = type;
	while (isa<ArrayType>(t))
	{
		auto tp = dyn_cast<ArrayType>(t);
		dims.push_back(tp->getNumElements());
		t = tp->getContainedType(0);
	}

	return dest.getDimensions() == dims;
}

static Expected<Value*> castReturnValue(
		IRBuilder<>& builder, Value* val, const ModType& type)
{
	auto ptrArrayType = dyn_cast<PointerType>(val->getType());
	auto arrayType = dyn_cast<ArrayType>(ptrArrayType->getContainedType(0));

	assert(castable(arrayType, type));	// NOLINT

	auto destType = typeToLLVMType(builder.getContext(), type);
	auto singleDestType = destType->getContainedType(0);
	while (isa<ArrayType>(singleDestType))
		singleDestType = singleDestType->getContainedType(0);

	if (destType == arrayType)
		return val;

	auto alloca = allocaModType(builder, type);

	for (size_t a = 0; a < arrayType->getNumElements(); a++)
	{
		auto loadedElem = loadArrayElement(builder, val, a);

		Value* casted = castSingleElem(builder, loadedElem, singleDestType);
		storeToArrayElement(builder, casted, alloca, a);
	}

	return alloca;
}

namespace modelica
{
	Expected<AllocaInst*> lowerConstant(IRBuilder<>& builder, const ModExp& exp)
	{
		if (exp.isConstant<int>())
			return lowerConstantTyped<int>(
					builder, exp.getConstant<int>(), exp.getModType());
		if (exp.isConstant<float>())
			return lowerConstantTyped<float>(
					builder, exp.getConstant<float>(), exp.getModType());
		if (exp.isConstant<bool>())
			return lowerConstantTyped<bool>(
					builder, exp.getConstant<bool>(), exp.getModType());

		assert(false && "unreachable");	 // NOLINT
		return nullptr;
	}

	Expected<Value*> lowerExp(LoweringInfo& info, const ModExp& exp)
	{
		auto retVal = uncastedLowerExp(info, exp);
		if (!retVal)
			return retVal;

		return castReturnValue(info.builder, *retVal, exp.getModType());
	}
}	 // namespace modelica
