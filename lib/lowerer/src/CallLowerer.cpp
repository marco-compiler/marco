#include "CallLowerer.hpp"

#include "ExpLowerer.hpp"

using namespace llvm;
using namespace std;

namespace modelica
{
	static Error invoke(
			LowererContext& info, StringRef name, ArrayRef<Value*> args)
	{
		auto voidType = Type::getVoidTy(info.getContext());
		SmallVector<Type*, 3> argsTypes;
		for (auto val : args)
			argsTypes.push_back(val->getType());

		auto functionType = FunctionType::get(voidType, argsTypes, false);
		auto externalFun = info.getModule().getOrInsertFunction(name, functionType);

		info.getBuilder().CreateCall(externalFun, args);
		return Error::success();
	}

	Expected<Value*> lowerCall(LowererContext& info, const ModCall& call)
	{
		SmallVector<Value*, 3> argsValue;

		auto alloca = info.allocaModType(call.getType());
		argsValue.push_back(alloca);
		call.dump();
		argsValue.push_back(info.getTypeDimensionsArray(call.getType()));

		for (size_t a = 0; a < call.argsSize(); a++)
		{
			auto arg = lowerExp(info, call.at(a));
			if (!arg)
				return arg;

			argsValue.push_back(*arg);
			argsValue.push_back(info.getTypeDimensionsArray(call.at(a).getModType()));
		}

		if (auto e = invoke(info, call.getName(), argsValue))
			return move(e);

		return alloca;
	}
}	 // namespace modelica
