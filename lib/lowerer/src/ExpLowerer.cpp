#include "ExpLowerer.hpp"

#include "CallLowerer.hpp"
#include "modelica/model/ModErrors.hpp"

using namespace std;
using namespace llvm;
using namespace modelica;

template<typename T>
Expected<AllocaInst*> lowerConstantTyped(
		IRBuilder<> builder, const ModConst<T>& constant, const ModType& type)
{
	if (constant.size() != type.flatSize())
		return make_error<TypeConstantSizeMissMatch>(constant, type);

	auto alloca = allocaModType(builder, type);

	for (size_t i = 0; i < constant.size(); i++)
		storeConstantToArrayElement<T>(builder, constant.get(i), alloca, i);
	return alloca;
}

template<int argumentsSize>
static AllocaInst* elementWiseOperation(
		IRBuilder<>& builder,
		Function* fun,
		ArrayRef<Value*>& args,
		const ModType& operationOutType,
		std::function<Value*(IRBuilder<>&, ArrayRef<Value*>)> operation)
{
	auto alloca = allocaModType(builder, operationOutType);

	const auto forBody = [alloca, &operation, &args](
													 IRBuilder<>& bld, Value* index) {
		SmallVector<Value*, argumentsSize> arguments;

		for (auto arg : args)
			arguments.push_back(loadArrayElement(bld, arg, index));

		auto outVal = operation(bld, arguments);

		storeToArrayElement(bld, outVal, alloca, index);
	};

	createForCycle(fun, builder, operationOutType.flatSize(), forBody);

	return alloca;
}

static Value* createFloatSingleBynaryOp(
		IRBuilder<>& builder, ArrayRef<Value*> args, ModExpKind kind)
{
	assert(args[0]->getType()->isFloatTy());					// NOLINT
	assert(ModExp::Operation::arityOfOp(kind) == 2);	// NOLINT
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
		case ModExpKind::zero:
		case ModExpKind::negate:
		case ModExpKind::conditional:
			assert(false && "unreachable");	 // NOLINT
			return nullptr;
	}
	assert(false && "unreachable");	 // NOLINT
	return nullptr;
}

static Value* createIntSingleBynaryOp(
		IRBuilder<>& builder, ArrayRef<Value*> args, ModExpKind kind)
{
	assert(args[0]->getType()->isIntegerTy());				// NOLINT
	assert(ModExp::Operation::arityOfOp(kind) == 2);	// NOLINT
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
		case ModExpKind::zero:
		case ModExpKind::negate:
		case ModExpKind::conditional:
			assert(false && "unreachable");	 // NOLINT
			return nullptr;
	}
	assert(false && "unreachable");	 // NOLINT
	return nullptr;
}

static Value* createSingleBynaryOP(
		IRBuilder<>& builder, ArrayRef<Value*> args, ModExpKind kind)
{
	assert(ModExp::Operation::arityOfOp(kind) == 2);	// NOLINT
	auto type = args[0]->getType();
	if (type->isIntegerTy())
		return createIntSingleBynaryOp(builder, args, kind);
	if (type->isFloatTy())
		return createFloatSingleBynaryOp(builder, args, kind);

	assert(false && "unreachable");	 // NOLINT
	return nullptr;
}

static Value* createSingleUnaryOp(
		IRBuilder<>& builder, ArrayRef<Value*> args, ModExpKind kind)
{
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
		default:
			assert(false && "unreachable");	 // NOLINT
			return nullptr;
	}
	assert(false && "unreachable");	 // NOLINT
	return nullptr;
}

template<size_t arity>
static Expected<AllocaInst*> lowerUnOrBinOp(
		IRBuilder<>& builder,
		Function* fun,
		const ModExp& exp,
		ArrayRef<Value*> subExp)
{
	static_assert(arity < 3 && arity > 0, "cannot lower op with this arity");
	assert(exp.getArity() == arity);	// NOLINT;
	assert(subExp.size() == arity);		// NOLINT
	const auto binaryOp = [type = exp.getKind()](auto& builder, auto args) {
		if constexpr (arity == 1)
			return createSingleUnaryOp(builder, args, type);
		else
			return createSingleBynaryOP(builder, args, type);
	};

	auto opType = exp.getOperationReturnType();
	return elementWiseOperation<arity>(builder, fun, subExp, opType, binaryOp);
}

static Expected<Value*> lowerTernary(
		IRBuilder<>& builder, Module& module, Function* fun, const ModExp& exp)
{
	assert(exp.isTernary());	// NOLINT
	const auto leftHandLowerer = [&exp, fun, &module](IRBuilder<>& builder) {
		return lowerExp(builder, module, fun, exp.getLeftHand());
	};
	const auto rightHandLowerer = [&exp, fun, &module](IRBuilder<>& builder) {
		return lowerExp(builder, module, fun, exp.getRightHand());
	};
	const auto conditionLowerer =
			[&exp, fun, &module](IRBuilder<>& builder) -> Expected<Value*> {
		auto& condition = exp.getCondition();
		assert(condition.getModType() == ModType(BultinModTypes::BOOL));	// NOLINT
		auto ptrToCond = lowerExp(builder, module, fun, condition);
		if (!ptrToCond)
			return ptrToCond;

		size_t zero = 0;
		return loadArrayElement(builder, *ptrToCond, zero);
	};

	auto type = exp.getLeftHand().getModType();
	auto llvmType = typeToLLVMType(builder.getContext(), type)->getPointerTo(0);

	return createTernaryOp(
			fun,
			builder,
			llvmType,
			conditionLowerer,
			leftHandLowerer,
			rightHandLowerer);
}

static Expected<Value*> lowerOperation(
		IRBuilder<>& builder, Module& m, Function* fun, const ModExp& exp)
{
	assert(exp.isOperation());	// NOLINT

	if (ModExpKind::zero == exp.getKind())
	{
		ModType type(BultinModTypes::INT);
		IntModConst constant(0);
		return lowerConstantTyped(builder, constant, type);
	}

	if (exp.isTernary())
		return lowerTernary(builder, m, fun, exp);

	if (exp.isUnary())
	{
		SmallVector<Value*, 1> values;
		auto subexp = lowerExp(builder, m, fun, exp.getLeftHand());
		if (!subexp)
			return subexp.takeError();
		values.push_back(move(*subexp));

		return lowerUnOrBinOp<1>(builder, fun, exp, values);
	}

	if (exp.isBinary())
	{
		SmallVector<Value*, 2> values;
		auto leftHand = lowerExp(builder, m, fun, exp.getLeftHand());
		if (!leftHand)
			return leftHand.takeError();
		values.push_back(move(*leftHand));

		auto rightHand = lowerExp(builder, m, fun, exp.getRightHand());
		if (!rightHand)
			return rightHand.takeError();
		values.push_back(move(*rightHand));

		return lowerUnOrBinOp<2>(builder, fun, exp, values);
	}
	assert(false && "Unreachable");	 // NOLINT
	return nullptr;
}

static Expected<Value*> uncastedLowerExp(
		IRBuilder<>& builder, Module& mod, Function* fun, const ModExp& exp)
{
	if (!exp.areSubExpressionCompatibles())
		return make_error<TypeMissMatch>(exp);

	if (exp.isConstant())
		return lowerConstant(builder, exp);

	if (exp.isReference())
		return lowerReference(builder, exp.getReference());

	if (exp.isCall())
		return lowerCall(builder, mod, fun, exp.getCall());

	return lowerOperation(builder, mod, fun, exp);
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
		IRBuilder<>& builder, Value* val, const ModType& type)
{
	auto ptrArrayType = dyn_cast<PointerType>(val->getType());
	auto arrayType = dyn_cast<ArrayType>(ptrArrayType->getContainedType(0));

	assert(arrayType->getNumElements() == type.flatSize());	 // NOLINT

	auto destType = typeToLLVMType(builder.getContext(), type);
	auto singleDestType = destType->getContainedType(0);

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

	Expected<Value*> lowerExp(
			IRBuilder<>& builder, Module& module, Function* fun, const ModExp& exp)
	{
		auto retVal = uncastedLowerExp(builder, module, fun, exp);
		if (!retVal)
			return retVal;

		return castReturnValue(builder, *retVal, exp.getModType());
	}
}	 // namespace modelica
