#include "modelica/lowerer/Lowerer.hpp"

#include "ExpLowerer.hpp"
#include "LowererUtils.hpp"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IRBuilder.h"
#include "modelica/model/ModErrors.hpp"

using namespace std;
using namespace modelica;
using namespace llvm;

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
	// call init
	builder.CreateCall(init);

	const auto forBody = [&](Value* index) {
		builder.CreateCall(update);
		builder.CreateCall(printValues);
	};

	// creates a for with simulationStop iterations what invokes
	// update and print values each time
	auto loopExit = createForCycle(main, builder, 0, simulationStop, forBody);

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

static Error insertGlobal(
		Module& module,
		StringRef name,
		const ModExp& exp,
		GlobalValue::LinkageTypes linkage)
{
	const auto& type = exp.getModType();
	const string oldName = name.str() + "_old";

	if (auto e = simExpToGlobalVar(module, name, type, linkage); e)
		return e;
	if (auto e = simExpToGlobalVar(module, oldName, type, linkage); e)
		return e;
	insertGlobalString(module, name.str() + "_str", name);
	return Error::success();
}

static Error createAllGlobals(
		Module& m, StringMap<ModExp> vars, GlobalValue::LinkageTypes linkType)
{
	for (const auto& pair : vars)
		if (auto e = insertGlobal(m, pair.first(), pair.second, linkType); e)
			return e;

	return Error::success();
}

static Expected<Function*> initializeGlobals(
		Module& m, const StringMap<ModExp>& vars)
{
	auto initFunctionExpected = makePrivateFunction("init", m);
	if (!initFunctionExpected)
		return initFunctionExpected;
	auto initFunction = initFunctionExpected.get();
	IRBuilder builder(&initFunction->getEntryBlock());

	for (const auto& pair : vars)
	{
		LoweringInfo info = { builder, m, initFunction };
		auto val = lowerExp(info, pair.second, true);
		if (!val)
			return val.takeError();
		auto loaded = builder.CreateLoad(*val);
		builder.CreateStore(loaded, m.getGlobalVariable(pair.first(), true));
		builder.CreateStore(
				loaded, m.getGlobalVariable(pair.first().str() + "_old", true));
	}
	builder.CreateRet(nullptr);
	return initFunction;
}

static Error createNormalAssigment(
		LoweringInfo& info, const Assigment& assigment)
{
	auto left = lowerExp(info, assigment.getVarName(), false);
	if (!left)
		return left.takeError();

	auto val = lowerExp(info, assigment.getExpression(), true);
	if (!val)
		return val.takeError();

	auto loaded = info.builder.CreateLoad(*val);
	info.builder.CreateStore(loaded, *left);
	return Error::success();
}

static Error createForAssigment(LoweringInfo info, const Assigment& assigment)
{
	SmallVector<size_t, 3> inductionsBegin;
	SmallVector<size_t, 3> inductionsEnd;
	for (const auto& ind : assigment)
		inductionsBegin.push_back(ind.begin());
	for (const auto& ind : assigment)
		inductionsEnd.push_back(ind.end());
	Error err = Error::success();

	auto lowerBody = [&](Value* inductionVars) {
		info.inductionsVars = inductionVars;
		auto error = createNormalAssigment(info, assigment);
		if (error)
			err = move(error);

		info.inductionsVars = nullptr;
	};

	createdNestedForCycle(
			info.function, info.builder, inductionsBegin, inductionsEnd, lowerBody);
	return err;
}

static Error createAssigment(LoweringInfo& info, const Assigment& assigment)
{
	if (assigment.size() == 0)
		return createNormalAssigment(info, assigment);

	return createForAssigment(info, assigment);
}

static Expected<Function*> createUpdates(
		Module& m,
		const SmallVector<Assigment, 0>& upds,
		const StringMap<ModExp>& definitions)
{
	auto updateFunctionExpected = makePrivateFunction("update", m);
	if (!updateFunctionExpected)
		return updateFunctionExpected;
	auto updateFunction = updateFunctionExpected.get();
	IRBuilder bld(&updateFunction->getEntryBlock());

	size_t counter = 0;
	for (const auto& pair : upds)
	{
		auto expFun = makePrivateFunction("update" + to_string(counter), m);
		if (!expFun)
			return expFun;

		auto fun = expFun.get();
		bld.SetInsertPoint(&fun->getEntryBlock());

		LoweringInfo info = { bld, m, fun };
		auto err = createAssigment(info, pair);
		if (err)
			return move(err);

		bld.CreateRet(nullptr);

		bld.SetInsertPoint(&updateFunction->getEntryBlock());
		bld.CreateCall(fun);
		counter++;
	}
	for (const auto& pair : definitions)
	{
		auto globalVal = m.getGlobalVariable(pair.first(), true);
		auto val = bld.CreateLoad(globalVal);

		bld.CreateStore(
				val, m.getGlobalVariable(pair.first().str() + "_old", true));
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
		Module& m,
		IRBuilder<>& builder,
		GlobalValue* varString,
		GlobalValue* ptrToVar)
{
	size_t index = 0;
	auto ptrToStrName = getArrayElementPtr(builder, varString, index);

	auto ptrType = ptrToVar->getType();
	auto arrayType = dyn_cast<ArrayType>(ptrType->getContainedType(0));
	auto basType = baseType(ptrToVar->getType());
	auto ptrToBaseType = basType->getPointerTo(0);

	auto charPtrType = ptrToStrName->getType();
	auto intType = IntegerType::getInt32Ty(builder.getContext());

	SmallVector<Type*, 3> args({ charPtrType, ptrToBaseType, intType });
	auto printType =
			FunctionType::get(Type::getVoidTy(m.getContext()), args, false);

	const auto selectPrintName = [intType](Type* t) {
		if (t->isFloatTy())
			return "modelicaPrintFVector";
		if (t == intType)
			return "modelicaPrintIVector";

		return "modelicaPrintBVector";
	};

	auto callee = m.getOrInsertFunction(selectPrintName(basType), printType);
	auto externalPrint = dyn_cast<Function>(callee.getCallee());

	auto numElements = ConstantInt::get(intType, arrayType->getNumElements());
	auto casted = builder.CreatePointerCast(ptrToVar, ptrToBaseType);
	SmallVector<Value*, 3> argsVal({ ptrToStrName, casted, numElements });
	builder.CreateCall(externalPrint, argsVal);
}

static Expected<Function*> populatePrint(Module& m, StringMap<ModExp> vars)
{
	auto printFunctionExpected = makePrivateFunction("print", m);
	if (!printFunctionExpected)
		return printFunctionExpected;
	auto printFunction = printFunctionExpected.get();
	IRBuilder bld(&printFunction->getEntryBlock());

	for (const auto& pair : vars)
	{
		auto varString = m.getGlobalVariable(pair.first().str() + "_str", true);
		auto ptrToVar = m.getGlobalVariable(pair.first(), true);
		createPrintOfVar(m, bld, varString, ptrToVar);
	}
	bld.CreateRet(nullptr);
	return printFunction;
}

Error Lowerer::lower()
{
	if (auto e = createAllGlobals(module, variables, getVarLinkage()); e)
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
	auto const dumpAssigment = [&OS](const auto& couple) {
		couple.getVarName().dump(OS);
		OS << " = ";
		couple.getExpression().dump(OS);
		OS << "\n";
	};
	auto const dumpEquation = [&OS](const auto& couple) {
		OS << couple.first().data();
		OS << " = ";
		couple.second.dump(OS);
		OS << "\n";
	};

	OS << "Init:\n";
	for_each(begin(variables), end(variables), dumpEquation);

	OS << "Update:\n";
	for_each(begin(updates), end(updates), dumpAssigment);
}

void Lowerer::dumpBC(raw_ostream& OS) const { WriteBitcodeToFile(module, OS); }

void Lowerer::dumpHeader(raw_ostream& OS) const
{
	OS << "#pragma once\n\n";

	for (const auto& var : variables)
	{
		const auto& type = var.second.getModType();

		OS << "extern ";
		type.dumpCSyntax(var.first(), OS);

		OS << ";\n";
	}

	OS << "extern \"C\"{";
	OS << "void " << entryPointName << "();";
	OS << "}";
}
