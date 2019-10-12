#include "ExpLowerer.hpp"

#include "modelica/simulation/SimErrors.hpp"

using namespace std;
using namespace llvm;
using namespace modelica;

template<typename T>
Expected<AllocaInst*> lowerConstantTyped(
		IRBuilder<> builder, const SimConst<T>& constant, const SimType& type)
{
	if (constant.size() != type.flatSize())
		return make_error<TypeConstantSizeMissMatch>(constant, type);

	auto alloca = allocaSimType(builder, type);

	for (size_t i = 0; i < constant.size(); i++)
		storeConstantToArrayElement<T>(builder, constant.get(i), alloca, i);
	return alloca;
}

template<int argumentsSize>
static AllocaInst* elementWiseOperation(
		IRBuilder<>& builder,
		Function* fun,
		ArrayRef<Value*>& args,
		const SimType& operationOutType,
		std::function<Value*(IRBuilder<>&, ArrayRef<Value*>)> operation)
{
	auto alloca = allocaSimType(builder, operationOutType);

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
		IRBuilder<>& builder, ArrayRef<Value*> args, SimExpKind kind)
{
	assert(args[0]->getType()->isFloatTy());					// NOLINT
	assert(SimExp::Operation::arityOfOp(kind) == 2);	// NOLINT
	switch (kind)
	{
		case SimExpKind::add:
			return builder.CreateFAdd(args[0], args[1]);
		case SimExpKind::sub:
			return builder.CreateFSub(args[0], args[1]);
		case SimExpKind::mult:
			return builder.CreateFMul(args[0], args[1]);
		case SimExpKind::divide:
			return builder.CreateFDiv(args[0], args[1]);
		case SimExpKind::equal:
			return builder.CreateFCmpOEQ(args[0], args[1]);
		case SimExpKind::different:
			return builder.CreateFCmpONE(args[0], args[1]);
		case SimExpKind::greaterEqual:
			return builder.CreateFCmpOGE(args[0], args[1]);
		case SimExpKind::greaterThan:
			return builder.CreateFCmpOGT(args[0], args[1]);
		case SimExpKind::lessEqual:
			return builder.CreateFCmpOLE(args[0], args[1]);
		case SimExpKind::less:
			return builder.CreateFCmpOLT(args[0], args[1]);
		case SimExpKind::module: {
			// TODO
			assert(false && "module unsupported");	// NOLINT
			return nullptr;
		}
		case SimExpKind::elevation: {
			// TODO
			assert(false && "powerof unsupported");	 // NOLINT
			return nullptr;
		}
		case SimExpKind::zero:
		case SimExpKind::negate:
		case SimExpKind::conditional:
			assert(false && "unreachable");	 // NOLINT
			return nullptr;
	}
	assert(false && "unreachable");	 // NOLINT
	return nullptr;
}

static Value* createIntSingleBynaryOp(
		IRBuilder<>& builder, ArrayRef<Value*> args, SimExpKind kind)
{
	assert(args[0]->getType()->isIntegerTy());				// NOLINT
	assert(SimExp::Operation::arityOfOp(kind) == 2);	// NOLINT
	switch (kind)
	{
		case SimExpKind::add:
			return builder.CreateAdd(args[0], args[1]);
		case SimExpKind::sub:
			return builder.CreateSub(args[0], args[1]);
		case SimExpKind::mult:
			return builder.CreateMul(args[0], args[1]);
		case SimExpKind::divide:
			return builder.CreateSDiv(args[0], args[1]);
		case SimExpKind::equal:
			return builder.CreateICmpEQ(args[0], args[1]);
		case SimExpKind::different:
			return builder.CreateICmpNE(args[0], args[1]);
		case SimExpKind::greaterEqual:
			return builder.CreateICmpSGE(args[0], args[1]);
		case SimExpKind::greaterThan:
			return builder.CreateICmpSGT(args[0], args[1]);
		case SimExpKind::lessEqual:
			return builder.CreateICmpSLE(args[0], args[1]);
		case SimExpKind::less:
			return builder.CreateICmpSLT(args[0], args[1]);
		case SimExpKind::module: {
			// TODO
			assert(false && "module unsupported");	// NOLINT
			return nullptr;
		}
		case SimExpKind::elevation: {
			// TODO
			assert(false && "powerof unsupported");	 // NOLINT
			return nullptr;
		}
		case SimExpKind::zero:
		case SimExpKind::negate:
		case SimExpKind::conditional:
			assert(false && "unreachable");	 // NOLINT
			return nullptr;
	}
	assert(false && "unreachable");	 // NOLINT
	return nullptr;
}

static Value* createSingleBynaryOP(
		IRBuilder<>& builder, ArrayRef<Value*> args, SimExpKind kind)
{
	assert(SimExp::Operation::arityOfOp(kind) == 2);	// NOLINT
	auto type = args[0]->getType();
	if (type->isIntegerTy())
		return createIntSingleBynaryOp(builder, args, kind);
	if (type->isFloatTy())
		return createFloatSingleBynaryOp(builder, args, kind);

	assert(false && "unreachable");	 // NOLINT
	return nullptr;
}

static Value* createSingleUnaryOp(
		IRBuilder<>& builder, ArrayRef<Value*> args, SimExpKind kind)
{
	auto intType = IntegerType::getInt32Ty(builder.getContext());
	auto boolType = IntegerType::getInt1Ty(builder.getContext());
	auto zero = ConstantInt::get(boolType, 0);
	switch (kind)
	{
		case SimExpKind::negate:
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
		const SimExp& exp,
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
		IRBuilder<>& builder, Function* fun, const SimExp& exp)
{
	assert(exp.isTernary());	// NOLINT
	const auto leftHandLowerer = [&exp, fun](IRBuilder<>& builder) {
		return lowerExp(builder, fun, exp.getLeftHand());
	};
	const auto rightHandLowerer = [&exp, fun](IRBuilder<>& builder) {
		return lowerExp(builder, fun, exp.getRightHand());
	};
	const auto conditionLowerer =
			[&exp, fun](IRBuilder<>& builder) -> Expected<Value*> {
		auto& condition = exp.getCondition();
		assert(condition.getSimType() == SimType(BultinSimTypes::BOOL));	// NOLINT
		auto ptrToCond = lowerExp(builder, fun, condition);
		if (!ptrToCond)
			return ptrToCond;

		size_t zero = 0;
		return loadArrayElement(builder, *ptrToCond, zero);
	};

	auto type = exp.getLeftHand().getSimType();
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
		IRBuilder<>& builder, Function* fun, const SimExp& exp)
{
	assert(exp.isOperation());	// NOLINT

	if (SimExpKind::zero == exp.getKind())
	{
		SimType type(BultinSimTypes::INT);
		IntSimConst constant(0);
		return lowerConstantTyped(builder, constant, type);
	}

	if (exp.isTernary())
		return lowerTernary(builder, fun, exp);

	if (exp.isUnary())
	{
		SmallVector<Value*, 1> values;
		auto subexp = lowerExp(builder, fun, exp.getLeftHand());
		if (!subexp)
			return subexp.takeError();
		values.push_back(move(*subexp));

		return lowerUnOrBinOp<1>(builder, fun, exp, values);
	}

	if (exp.isBinary())
	{
		SmallVector<Value*, 2> values;
		auto leftHand = lowerExp(builder, fun, exp.getLeftHand());
		if (!leftHand)
			return leftHand.takeError();
		values.push_back(move(*leftHand));

		auto rightHand = lowerExp(builder, fun, exp.getRightHand());
		if (!rightHand)
			return rightHand.takeError();
		values.push_back(move(*rightHand));

		return lowerUnOrBinOp<2>(builder, fun, exp, values);
	}
	assert(false && "Unreachable");	 // NOLINT
	return nullptr;
}

static Expected<Value*> uncastedLowerExp(
		IRBuilder<>& builder, Function* fun, const SimExp& exp)
{
	if (!exp.areSubExpressionCompatibles())
		return make_error<TypeMissMatch>(exp);

	if (exp.isConstant())
		return lowerConstant(builder, exp);

	if (exp.isReference())
		return lowerReference(builder, exp.getReference());

	return lowerOperation(builder, fun, exp);
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
		IRBuilder<>& builder, Value* val, const SimType& type)
{
	auto ptrArrayType = dyn_cast<PointerType>(val->getType());
	auto arrayType = dyn_cast<ArrayType>(ptrArrayType->getContainedType(0));

	assert(arrayType->getNumElements() == type.flatSize());	 // NOLINT

	auto destType = typeToLLVMType(builder.getContext(), type);
	auto singleDestType = destType->getContainedType(0);

	if (destType == arrayType)
		return val;

	auto alloca = allocaSimType(builder, type);

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
	Expected<AllocaInst*> lowerConstant(IRBuilder<>& builder, const SimExp& exp)
	{
		if (exp.isConstant<int>())
			return lowerConstantTyped<int>(
					builder, exp.getConstant<int>(), exp.getSimType());
		if (exp.isConstant<float>())
			return lowerConstantTyped<float>(
					builder, exp.getConstant<float>(), exp.getSimType());
		if (exp.isConstant<bool>())
			return lowerConstantTyped<bool>(
					builder, exp.getConstant<bool>(), exp.getSimType());

		assert(false && "unreachable");	 // NOLINT
		return nullptr;
	}

	Expected<Value*> lowerExp(
			IRBuilder<>& builder, Function* fun, const SimExp& exp)
	{
		auto retVal = uncastedLowerExp(builder, fun, exp);
		if (!retVal)
			return retVal;

		return castReturnValue(builder, *retVal, exp.getSimType());
	}
}	 // namespace modelica
