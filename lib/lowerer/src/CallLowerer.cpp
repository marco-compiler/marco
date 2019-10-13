#include "CallLowerer.hpp"

#include "ExpLowerer.hpp"

using namespace llvm;
using namespace std;

namespace modelica
{
	static Error invoke(
			IRBuilder<>& builder,
			Module& module,
			Function* fun,
			StringRef name,
			ArrayRef<Value*> args)
	{
		auto voidType = Type::getVoidTy(builder.getContext());
		SmallVector<Type*, 3> argsTypes;
		for (auto val : args)
			argsTypes.push_back(val->getType());

		auto functionType = FunctionType::get(voidType, argsTypes, false);
		auto externalFun = module.getOrInsertFunction(name, functionType);

		builder.CreateCall(externalFun, args);
		return Error::success();
	}

	Expected<Value*> lowerCall(
			IRBuilder<>& builder, Module& module, Function* fun, const SimCall& call)
	{
		SmallVector<Value*, 3> argsValue;

		auto alloca = allocaSimType(builder, call.getType());
		argsValue.push_back(alloca);
		argsValue.push_back(getTypeDimensionsArray(builder, call.getType()));

		for (size_t a = 0; a < call.argsSize(); a++)
		{
			auto arg = lowerExp(builder, module, fun, call.at(a));
			if (!arg)
				return arg;

			argsValue.push_back(*arg);
			argsValue.push_back(
					getTypeDimensionsArray(builder, call.at(a).getSimType()));
		}

		if (auto e = invoke(builder, module, fun, call.getName(), argsValue))
			return move(e);

		return alloca;
	}
}	 // namespace modelica
