#include "CallLowerer.hpp"

#include "ExpLowerer.hpp"

using namespace llvm;
using namespace std;

namespace modelica
{
	static Error invoke(LoweringInfo& info, StringRef name, ArrayRef<Value*> args)
	{
		auto voidType = Type::getVoidTy(info.builder.getContext());
		SmallVector<Type*, 3> argsTypes;
		for (auto val : args)
			argsTypes.push_back(val->getType());

		auto functionType = FunctionType::get(voidType, argsTypes, false);
		auto externalFun = info.module.getOrInsertFunction(name, functionType);

		info.builder.CreateCall(externalFun, args);
		return Error::success();
	}

	Expected<Value*> lowerCall(LoweringInfo& info, const ModCall& call)
	{
		SmallVector<Value*, 3> argsValue;

		auto alloca = allocaModType(info.builder, call.getType());
		argsValue.push_back(alloca);
		argsValue.push_back(getTypeDimensionsArray(info.builder, call.getType()));

		for (size_t a = 0; a < call.argsSize(); a++)
		{
			auto arg = lowerExp(info, call.at(a));
			if (!arg)
				return arg;

			argsValue.push_back(*arg);
			argsValue.push_back(
					getTypeDimensionsArray(info.builder, call.at(a).getModType()));
		}

		if (auto e = invoke(info, call.getName(), argsValue))
			return move(e);

		return alloca;
	}
}	 // namespace modelica
