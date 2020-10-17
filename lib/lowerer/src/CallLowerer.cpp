#include "CallLowerer.hpp"

#include <cstdlib>
#include <string>

#include "ExpLowerer.hpp"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace std;

namespace modelica
{
	static Error invoke(
			LowererContext& info, StringRef name, ArrayRef<Value*> args)
	{
		std::string realName = name;
		if (name == "fill" and info.useDoubles())
			realName = "filld";

		auto* voidType = Type::getVoidTy(info.getContext());
		SmallVector<Type*, 3> argsTypes;
		for (auto* val : args)
			argsTypes.push_back(val->getType());

		auto* functionType = FunctionType::get(voidType, argsTypes, false);
		auto externalFun =
				info.getModule().getOrInsertFunction(realName, functionType);

		info.getBuilder().CreateCall(externalFun, args);
		return Error::success();
	}

	Expected<Value*> lowerCall(
			Value* outLocation, LowererContext& info, const ModCall& call)
	{
		SmallVector<Value*, 3> argsValue;
		argsValue.push_back(outLocation);
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

		return outLocation;
	}

	Expected<Value*> lowerCall(LowererContext& info, const ModCall& call)
	{
		auto alloca = info.allocaModType(call.getType());
		return lowerCall(alloca, info, call);
	}
}	 // namespace modelica
