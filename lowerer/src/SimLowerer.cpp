#include "llvm/IR/IRBuilder.h"
#include "modelica/lowerer/SimConst.hpp"
#include "modelica/lowerer/Simulation.hpp"

using namespace llvm;
using namespace std;

namespace modelica
{
	constexpr auto internalLinkage = GlobalValue::LinkageTypes::InternalLinkage;

	static FunctionType* getVoidType(LLVMContext& context)
	{
		return FunctionType::get(Type::getVoidTy(context), false);
	}

	Function* makePrivateFunction(StringRef name, Module& m)
	{
		auto function = m.getOrInsertFunction(name, getVoidType(m.getContext()));
		auto f = dyn_cast<Function>(function);
		BasicBlock::Create(function->getContext(), "entry", f);
		f->setLinkage(internalLinkage);
		return f;
	}

	Type* builtInToLLVMType(LLVMContext& context, BultinSimTypes type)
	{
		switch (type)
		{
			case BultinSimTypes::INT:
				return Type::getInt32Ty(context);
			case BultinSimTypes::BOOL:
				return Type::getInt1Ty(context);
			case BultinSimTypes::FLOAT:
				return Type::getFloatTy(context);
		}

		assert(false && "Unreachable");	// NOLINT
		return nullptr;
	}

	Type* typeToLLVMType(LLVMContext& context, const SimType& type)
	{
		auto baseType = builtInToLLVMType(context, type.getBuiltin());

		if (type.getDimensionsCount() == 0)
			return baseType;

		return ArrayType::get(baseType, type.flatSize());
	}

	Error simExpToGlobalVar(
			Module& module,
			StringRef name,
			const SimType& simType,
			GlobalValue::LinkageTypes linkage)
	{
		auto type = typeToLLVMType(module.getContext(), simType);
		auto varDecl = module.getOrInsertGlobal(name, type);
		if (varDecl == nullptr)
			return make_error<GlobalVariableCreationFailure>(name.str());

		auto global = dyn_cast<GlobalVariable>(varDecl);
		global->setLinkage(linkage);
		auto constant = ConstantInt::get(type, 0);
		global->setInitializer(constant);
		return Error::success();
	}

	Value* lowerReference(IRBuilder<>& builder, StringRef exp)
	{
		auto module = builder.GetInsertBlock()->getModule();
		auto global = module->getGlobalVariable(exp.str() + "_old", true);
		return builder.CreateLoad(global);
	}
}	// namespace modelica
