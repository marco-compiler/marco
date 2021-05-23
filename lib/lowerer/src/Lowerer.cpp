#include "marco/lowerer/Lowerer.hpp"

#include <utility>

#include "CallLowerer.hpp"
#include "ExpLowerer.hpp"
#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "marco/model/Assigment.hpp"
#include "marco/model/ModErrors.hpp"
#include "marco/model/ModExp.hpp"
#include "marco/model/ModVariable.hpp"
#include "marco/passes/PassManager.hpp"
#include "marco/utils/Interval.hpp"

using namespace std;
using namespace marco;
using namespace llvm;

constexpr auto internalLinkage = GlobalValue::LinkageTypes::InternalLinkage;

template<typename... T>
static FunctionType* getVoidType(LLVMContext& context, T... argTypes)
{
	return FunctionType::get(Type::getVoidTy(context), { argTypes... }, false);
}

template<typename... T>
Expected<Function*> makePublicFunction(
		StringRef name, Module& module, T&&... argsTypes)
{
	if (module.getFunction(name) != nullptr)
		return make_error<FunctionAlreadyExists>(name.str());

	auto v = getVoidType(module.getContext(), argsTypes...);
	auto function = module.getOrInsertFunction(name, v);
	auto f = dyn_cast<llvm::Function>(function.getCallee());
	BasicBlock::Create(module.getContext(), "entry", f);
	return f;
}

template<typename... T>
Expected<Function*> makePrivateFunction(
		StringRef name, Module& module, T&&... argsTypes)
{
	auto F = makePublicFunction(name, module, std::forward<T>(argsTypes)...);
	if (not F)
		return F.takeError();
	(**F).setLinkage(internalLinkage);
	return F;
}

Error Lowerer::simExpToGlobalVar(
		LowererContext& info,
		StringRef name,
		const ModType& simType,
		GlobalValue::LinkageTypes linkage)
{
	auto* type = typeToLLVMType(module.getContext(), simType, info.useDoubles());
	auto* varDecl = module.getOrInsertGlobal(name, type);
	if (varDecl == nullptr)
		return make_error<GlobalVariableCreationFailure>(name.str());

	auto* global = dyn_cast<GlobalVariable>(varDecl);
	global->setLinkage(linkage);

	global->setInitializer(ConstantAggregateZero::get(type));
	return Error::success();
}

static Expected<Function*> populateMain(
		LowererContext& info,
		StringRef entryPointName,
		Function* init,
		Function* update,
		Function* printValues,
		unsigned simulationStop)
{
	assert(init != nullptr);				 // NOLINT
	assert(update != nullptr);			 // NOLINT
	assert(printValues != nullptr);	 // NOLINT

	auto expectedMain = makePublicFunction(entryPointName, info.getModule());
	if (!expectedMain)
		return expectedMain;
	auto* main = expectedMain.get();

	IRBuilder<>& builder = info.getBuilder();
	builder.SetInsertPoint(&main->getEntryBlock());
	info.setFunction(main);

	// call init
	builder.CreateCall(init);
	builder.CreateCall(printValues);

	// creates a for with simulationStop iterations what invokes
	// update and print values each time

	const auto& loopBody = [&](Value* index) {
		builder.CreateCall(update);
		builder.CreateCall(printValues);
	};

	const auto loopRange = Interval(0, simulationStop);
	auto* loopExit = info.createForCycle(loopRange, loopBody, true);

	// returns right after the loop
	builder.SetInsertPoint(loopExit);
	builder.CreateRet(nullptr);
	return main;
}

static void insertGlobalString(Module& m, StringRef name, StringRef content)
{
	auto* str = ConstantDataArray::getString(m.getContext(), content);
	auto* type = ArrayType::get(
			IntegerType::getInt8Ty(m.getContext()), content.size() + 1);
	auto* global = m.getOrInsertGlobal(name, type);
	dyn_cast<GlobalVariable>(global)->setInitializer(str);
}

Error Lowerer::insertGlobal(
		LowererContext& info,
		StringRef name,
		const ModExp& exp,
		GlobalValue::LinkageTypes linkage)
{
	const auto& type = exp.getModType();

	if (auto e = simExpToGlobalVar(info, name, type, linkage); e)
		return e;

	insertGlobalString(module, name.str() + "_str", name);
	return Error::success();
}

Error Lowerer::createAllGlobals(
		LowererContext& info, GlobalValue::LinkageTypes linkType)
{
	for (const auto& pair : variables)
		if (auto e =
						insertGlobal(info, pair.first(), pair.second.getInit(), linkType);
				e)
			return e;

	return Error::success();
}

void Lowerer::verify() { llvm::verifyModule(module, &llvm::outs()); }

static bool isZeroInitialized(const ModVariable& var)
{
	const auto& exp = var.getInit();
	if (exp.isConstant() && exp.getConstant().as<double>() == 0.0F)
		return true;

	if (!exp.isCall())
		return false;

	const auto& call = exp.getCall();
	if (call.getName() != "fill")
		return false;

	const auto& initExp = call.at(0);
	return initExp.isConstant() && initExp.getConstant().as<double>() == 0.0F;
}

static Error shortCallInit(LowererContext& ctx, const ModVariable& var)
{
	auto* loc = ctx.getModule().getGlobalVariable(var.getName(), true);
	auto res = lowerCall(loc, ctx, var.getInit().getCall());

	if (!res)
		return res.takeError();

	return Error::success();
}

static Error initGlobal(LowererContext& ctx, const ModVariable& var)
{
	auto& builder = ctx.getBuilder();
	if (isZeroInitialized(var))
		return Error::success();

	if (var.getInit().isCall())
		return shortCallInit(ctx, var);

	auto val = lowerExp(ctx, var.getInit());
	if (!val)
		return val.takeError();
	auto* loaded = builder.CreateLoad(*val);
	builder.CreateStore(
			loaded, ctx.getModule().getGlobalVariable(var.getName(), true));
	return Error::success();
}

static Expected<Function*> initializeGlobals(
		LowererContext& info, const StringMap<ModVariable>& vars)
{
	auto initFunctionExpected = makePublicFunction("init", info.getModule());
	if (!initFunctionExpected)
		return initFunctionExpected;
	auto* initFunction = initFunctionExpected.get();

	info.getBuilder().SetInsertPoint(&initFunction->getEntryBlock());
	info.setFunction(initFunction);
	for (const auto& pair : vars)
		if (auto error = initGlobal(info, pair.second); error)
			return move(error);

	info.getBuilder().CreateRet(nullptr);
	return initFunction;
}

static Error createAssigmentBody(
		LowererContext& info, const Assigment& assigment)
{
	auto left = lowerExp(info, assigment.getLeftHand());
	if (!left)
		return left.takeError();

	auto val = lowerExp(info, assigment.getExpression());
	if (!val)
		return val.takeError();

	val = castReturnValue(info, *val, assigment.getLeftHand().getModType());
	if (!val)
		return val.takeError();

	auto loaded = info.getBuilder().CreateLoad(*val);
	info.getBuilder().CreateStore(loaded, *left);
	return Error::success();
}

static Expected<Function*> createFunctionAssigment(
		LowererContext& ctx, const Assigment& assigment)
{
	IRBuilder<>& bld = ctx.getBuilder();
	auto oldFunction = ctx.getFunction();
	auto oldBlock = ctx.getBuilder().GetInsertBlock();
	auto oldInduction = ctx.getInductionVars();

	auto maybeFunction = makePrivateFunction(
			assigment.getTemplateName(), ctx.getModule(), oldInduction->getType());

	if (!maybeFunction)
		return maybeFunction.takeError();

	Function* fun = maybeFunction.get();
	bld.SetInsertPoint(&fun->getEntryBlock());
	ctx.setFunction(fun);
	ctx.setInductions(fun->arg_begin());
	auto error = createAssigmentBody(ctx, assigment);
	if (error)
		return move(error);

	bld.CreateRet(nullptr);
	bld.SetInsertPoint(oldBlock);
	ctx.setFunction(oldFunction);
	ctx.setInductions(oldInduction);

	return fun;
}

static Error createNormalAssigment(
		LowererContext& info, const Assigment& assigment)
{
	const auto& templName = assigment.getTemplateName();
	if (templName.empty())
		return createAssigmentBody(info, assigment);

	auto templFun = info.getModule().getFunction(templName);
	if (templFun == nullptr)
	{
		auto maybeFun = createFunctionAssigment(info, assigment);
		if (!maybeFun)
			return maybeFun.takeError();

		templFun = *maybeFun;
	}

	info.getBuilder().CreateCall(templFun, { info.getInductionVars() });
	return Error::success();
}

static Error createForAssigment(
		LowererContext& info, const Assigment& assigment)
{
	Error err = Error::success();

	const auto forLoopBody = [&](Value* inductionVars) {
		info.setInductions(inductionVars);
		auto error = createNormalAssigment(info, assigment);
		if (error)
			err = move(error);

		info.setInductions(nullptr);
	};

	info.createdNestedForCycle(assigment.getOrderedInductionsVar(), forLoopBody);
	return err;
}

static Error createAssigment(LowererContext& info, const Assigment& assigment)
{
	if (assigment.size() == 0)
		return createNormalAssigment(info, assigment);

	return createForAssigment(info, assigment);
}

static Expected<Function*> createUpdate(
		LowererContext& cont, const Assigment& assigment, size_t functionIndex)
{
	IRBuilder<>& bld = cont.getBuilder();

	auto expFun = makePrivateFunction(
			"update" + to_string(functionIndex), cont.getModule());
	if (!expFun)
		return expFun.takeError();

	auto fun = expFun.get();
	bld.SetInsertPoint(&fun->getEntryBlock());
	cont.setFunction(fun);

	auto err = createAssigment(cont, assigment);
	if (err)
		return move(err);

	bld.CreateRet(nullptr);
	return expFun;
}

static Expected<Function*> createUpdates(
		LowererContext& info,
		const SmallVector<variant<Assigment, ModBltBlock>, 0>& upds,
		const StringMap<ModVariable>& definitions)
{
	auto updateFunctionExpected = makePublicFunction("update", info.getModule());
	if (!updateFunctionExpected)
		return updateFunctionExpected;
	auto* updateFunction = updateFunctionExpected.get();
	IRBuilder<>& bld = info.getBuilder();
	bld.SetInsertPoint(&updateFunction->getEntryBlock());
	info.setFunction(updateFunction);

	size_t counter = 0;
	for (const auto& assigment : upds)
	{
		if (!holds_alternative<Assigment>(assigment))
			return make_error<UnsolvableAlgebraicLoop>();
		auto fun = createUpdate(info, get<Assigment>(assigment), counter++);
		if (!fun)
			return fun.takeError();
		bld.SetInsertPoint(&updateFunction->getEntryBlock());
		info.setFunction(updateFunction);
		bld.CreateCall(*fun);
	}

	bld.CreateRet(nullptr);
	return updateFunction;
}

static Type* baseType(Type* tp)
{
	if (isa<PointerType>(tp))
		tp = dyn_cast<PointerType>(tp)->getContainedType(0);

	while (isa<ArrayType>(tp))
		tp = dyn_cast<ArrayType>(tp)->getContainedType(0);
	return tp;
}

static void createPrintOfVar(
		LowererContext& context, GlobalValue* varString, GlobalValue* ptrToVar)
{
	size_t index = 0;
	auto ptrToStrName = context.getArrayElementPtr(varString, index);

	auto ptrType = ptrToVar->getType();
	auto arrayType = dyn_cast<ArrayType>(ptrType->getContainedType(0));
	auto basType = baseType(ptrToVar->getType());
	auto ptrToBaseType = basType->getPointerTo(0);

	auto charPtrType = ptrToStrName->getType();
	auto intType = IntegerType::getInt32Ty(context.getContext());

	SmallVector<Type*, 3> args({ charPtrType, ptrToBaseType, intType });
	auto printType =
			FunctionType::get(Type::getVoidTy(context.getContext()), args, false);

	const auto selectPrintName = [intType](Type* t) {
		if (t->isFloatTy())
			return "modelicaPrintFVector";
		if (t == intType)
			return "modelicaPrintIVector";
		if (t->isIntegerTy(1))
			return "modelicaPrintBVector";
		if (t->isDoubleTy())
			return "modelicaPrintDVector";

		assert(false && "unreachable");
		return "UKNOWN TYPE";
	};

	auto callee = context.getModule().getOrInsertFunction(
			selectPrintName(basType), printType);
	auto* externalPrint = dyn_cast<Function>(callee.getCallee());

	auto* numElements =
			ConstantInt::get(intType, modTypeFromLLVMType(arrayType).flatSize());
	auto* casted =
			context.getBuilder().CreatePointerCast(ptrToVar, ptrToBaseType);
	SmallVector<Value*, 3> argsVal({ ptrToStrName, casted, numElements });
	context.getBuilder().CreateCall(externalPrint, argsVal);
}

static Expected<Function*> populatePrint(
		LowererContext& info, StringMap<ModVariable> vars)
{
	Module& m = info.getModule();
	auto printFunctionExpected = makePrivateFunction("print", m);
	if (!printFunctionExpected)
		return printFunctionExpected;
	auto* printFunction = printFunctionExpected.get();
	IRBuilder<>& bld = info.getBuilder();
	bld.SetInsertPoint(&printFunction->getEntryBlock());
	info.setFunction(printFunction);

	for (const auto& pair : vars)
	{
		auto* varString = m.getGlobalVariable(pair.first().str() + "_str", true);
		auto* ptrToVar = m.getGlobalVariable(pair.first(), true);
		createPrintOfVar(info, varString, ptrToVar);
	}
	bld.CreateRet(nullptr);
	return printFunction;
}

Error Lowerer::lower()
{
	IRBuilder<> builder(module.getContext());
	LowererContext ctx(builder, module, useDoubles);

	if (auto e = createAllGlobals(ctx, getVarLinkage()); e)
		return e;

	auto initFunction = initializeGlobals(ctx, variables);

	if (!initFunction)
		return initFunction.takeError();

	auto updateFunction = createUpdates(ctx, updates, variables);

	if (!updateFunction)
		return updateFunction.takeError();

	auto printFunction = populatePrint(ctx, variables);

	if (!printFunction)
		return printFunction.takeError();

	auto e = populateMain(
			ctx,
			entryPointName,
			*initFunction,
			*updateFunction,
			*printFunction,
			stopTime);

	if (!e)
		return e.takeError();

	return Error::success();
}

void Lowerer::dump(raw_ostream& OS) const
{
	auto const dumpEquation = [&OS](const auto& couple) {
		OS << couple.first().data();
		OS << " = ";
		couple.second.getInit().dump(OS);
		OS << "\n";
	};

	OS << "init:\n";
	for_each(begin(variables), end(variables), dumpEquation);

	OS << "update:\n";
	for (const auto& update : updates)
	{
		if (holds_alternative<Assigment>(update))
			get<Assigment>(update).dump(OS);
		else
			get<ModBltBlock>(update).dump(OS);
	}
}

void Lowerer::dumpBC(raw_ostream& OS) const { WriteBitcodeToFile(module, OS); }

void Lowerer::dumpHeader(raw_ostream& OS) const
{
	OS << "#pragma once\n\n";

	for (const auto& var : variables)
	{
		const auto& type = var.second.getInit().getModType();

		OS << "extern ";
		type.dumpCSyntax(var.first(), useDoubles, OS);

		OS << ";\n";
	}

	OS << "extern \"C\"{";
	OS << "void " << entryPointName << "();";
	OS << "}";
}
