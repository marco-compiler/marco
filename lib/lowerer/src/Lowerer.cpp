#include "modelica/lowerer/Lowerer.hpp"

#include "CallLowerer.hpp"
#include "ExpLowerer.hpp"
#include "LowererUtils.hpp"
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
#include "modelica/model/Assigment.hpp"
#include "modelica/model/ModErrors.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/passes/PassManager.hpp"
#include "modelica/utils/Interval.hpp"

using namespace std;
using namespace modelica;
using namespace llvm;
constexpr auto internalLinkage = GlobalValue::LinkageTypes::InternalLinkage;

template<typename... T>
static FunctionType* getVoidType(LLVMContext& context, T... argTypes)
{
	return FunctionType::get(Type::getVoidTy(context), { argTypes... }, false);
}
template<typename... T>
Expected<Function*> makePrivateFunction(
		StringRef name, Module& module, T... argsTypes)
{
	if (module.getFunction(name) != nullptr)
		return make_error<FunctionAlreadyExists>(name.str());

	auto v = getVoidType(module.getContext(), argsTypes...);
	auto function = module.getOrInsertFunction(name, v);
	auto f = dyn_cast<llvm::Function>(function.getCallee());
	BasicBlock::Create(module.getContext(), "entry", f);
	f->setLinkage(internalLinkage);
	return f;
}

Error Lowerer::simExpToGlobalVar(
		StringRef name, const ModType& simType, GlobalValue::LinkageTypes linkage)
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
static Expected<Function*> populateMain(
		Module& m,
		StringRef entryPointName,
		Function* init,
		Function* update,
		Function* printValues,
		unsigned simulationStop)
{
	assert(init != nullptr);				 // NOLINT
	assert(update != nullptr);			 // NOLINT
	assert(printValues != nullptr);	 // NOLINT

	auto expectedMain = makePrivateFunction(entryPointName, m);
	if (!expectedMain)
		return expectedMain;
	auto main = expectedMain.get();
	main->setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);

	IRBuilder<> builder(&main->getEntryBlock());
	LowererContext lCont(builder, m);
	lCont.setFunction(main);
	// call init
	builder.CreateCall(init);
	builder.CreateCall(printValues);
	// creates a for with simulationStop iterations what invokes
	// update and print values each time
	//
	//
	const auto& loopBody = [&](Value* index) {
		builder.CreateCall(update);
		builder.CreateCall(printValues);
	};
	const auto loopRange = Interval(0, simulationStop);
	auto loopExit = lCont.createForCycle(loopRange, loopBody, true);

	// returns right after the loop
	builder.SetInsertPoint(loopExit);
	builder.CreateRet(nullptr);
	return main;
}

static void insertGlobalString(Module& m, StringRef name, StringRef content)
{
	auto str = ConstantDataArray::getString(m.getContext(), content);
	auto type = ArrayType::get(
			IntegerType::getInt8Ty(m.getContext()), content.size() + 1);
	auto global = m.getOrInsertGlobal(name, type);
	dyn_cast<GlobalVariable>(global)->setInitializer(str);
}

Error Lowerer::insertGlobal(
		StringRef name, const ModExp& exp, GlobalValue::LinkageTypes linkage)
{
	const auto& type = exp.getModType();

	if (auto e = simExpToGlobalVar(name, type, linkage); e)
		return e;

	insertGlobalString(module, name.str() + "_str", name);
	return Error::success();
}

Error Lowerer::createAllGlobals(GlobalValue::LinkageTypes linkType)
{
	for (const auto& pair : variables)
		if (auto e = insertGlobal(pair.first(), pair.second.getInit(), linkType); e)
			return e;

	return Error::success();
}

void Lowerer::verify() { llvm::verifyModule(module, &llvm::outs()); }

static bool isZeroInitialized(const ModVariable& var)
{
	const auto& exp = var.getInit();
	if (exp.isConstant() && exp.getConstant().as<float>() == 0.0F)
		return true;

	if (!exp.isCall())
		return false;

	const auto& call = exp.getCall();
	if (call.getName() != "fill")
		return false;

	const auto& initExp = call.at(0);
	return initExp.isConstant() && initExp.getConstant().as<float>() == 0.0F;
}

static Error shortCallInit(LowererContext& ctx, const ModVariable& var)
{
	auto loc = ctx.getModule().getGlobalVariable(var.getName(), true);
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
	auto loaded = builder.CreateLoad(*val);
	builder.CreateStore(
			loaded, ctx.getModule().getGlobalVariable(var.getName(), true));
	return Error::success();
}

static Expected<Function*> initializeGlobals(
		Module& m, const StringMap<ModVariable>& vars)
{
	auto initFunctionExpected = makePrivateFunction("init", m);
	if (!initFunctionExpected)
		return initFunctionExpected;
	auto initFunction = initFunctionExpected.get();
	IRBuilder builder(&initFunction->getEntryBlock());
	LowererContext cont(builder, m);

	cont.setFunction(initFunction);
	for (const auto& pair : vars)
		if (auto error = initGlobal(cont, pair.second); error)
			return move(error);

	builder.CreateRet(nullptr);
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
		auto maybFun = createFunctionAssigment(info, assigment);
		if (!maybFun)
			return maybFun.takeError();

		templFun = *maybFun;
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
		Module& m,
		const SmallVector<Assigment, 0>& upds,
		const StringMap<ModVariable>& definitions)
{
	auto updateFunctionExpected = makePrivateFunction("update", m);
	if (!updateFunctionExpected)
		return updateFunctionExpected;
	auto updateFunction = updateFunctionExpected.get();
	IRBuilder bld(&updateFunction->getEntryBlock());
	LowererContext cont(bld, m);
	cont.setFunction(updateFunction);

	size_t counter = 0;
	for (const auto& assigment : upds)
	{
		auto fun = createUpdate(cont, assigment, counter++);
		if (!fun)
			return fun.takeError();
		bld.SetInsertPoint(&updateFunction->getEntryBlock());
		cont.setFunction(updateFunction);
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

		return "modelicaPrintBVector";
	};

	auto callee = context.getModule().getOrInsertFunction(
			selectPrintName(basType), printType);
	auto externalPrint = dyn_cast<Function>(callee.getCallee());

	auto numElements =
			ConstantInt::get(intType, modTypeFromLLVMType(arrayType).flatSize());
	auto casted = context.getBuilder().CreatePointerCast(ptrToVar, ptrToBaseType);
	SmallVector<Value*, 3> argsVal({ ptrToStrName, casted, numElements });
	context.getBuilder().CreateCall(externalPrint, argsVal);
}

static Expected<Function*> populatePrint(Module& m, StringMap<ModVariable> vars)
{
	auto printFunctionExpected = makePrivateFunction("print", m);
	if (!printFunctionExpected)
		return printFunctionExpected;
	auto printFunction = printFunctionExpected.get();
	IRBuilder bld(&printFunction->getEntryBlock());
	LowererContext cont(bld, m);
	cont.setFunction(printFunction);

	for (const auto& pair : vars)
	{
		auto varString = m.getGlobalVariable(pair.first().str() + "_str", true);
		auto ptrToVar = m.getGlobalVariable(pair.first(), true);
		createPrintOfVar(cont, varString, ptrToVar);
	}
	bld.CreateRet(nullptr);
	return printFunction;
}

Error Lowerer::lower()
{
	if (auto e = createAllGlobals(getVarLinkage()); e)
		return e;

	auto initFunction = initializeGlobals(module, variables);
	if (!initFunction)
		return initFunction.takeError();

	auto updateFunction = createUpdates(module, updates, variables);
	if (!updateFunction)
		return updateFunction.takeError();

	auto printFunction = populatePrint(module, variables);
	if (!printFunction)
		return printFunction.takeError();

	auto e = populateMain(
			module,
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
		update.dump(OS);
}

void Lowerer::dumpBC(raw_ostream& OS) const { WriteBitcodeToFile(module, OS); }

void Lowerer::dumpHeader(raw_ostream& OS) const
{
	OS << "#pragma once\n\n";

	for (const auto& var : variables)
	{
		const auto& type = var.second.getInit().getModType();

		OS << "extern ";
		type.dumpCSyntax(var.first(), OS);

		OS << ";\n";
	}

	OS << "extern \"C\"{";
	OS << "void " << entryPointName << "();";
	OS << "}";
}
