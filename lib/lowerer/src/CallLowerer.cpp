#include "marco/lowerer/CallLowerer.hpp"

#include <cstdlib>
#include <string>

#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include "marco/lowerer/ExpLowerer.hpp"

using namespace llvm;
using namespace std;

namespace marco
{
	Expected<Value*> invoke(
			LowererContext& info,
			StringRef name,
			ArrayRef<Value*> args,
			Type* returnType)
	{
		std::string realName = name.str();
		if (name == "fill" and info.useDoubles())
			realName = "filld";

		SmallVector<Type*, 3> argsTypes;
		for (auto* val : args)
			argsTypes.push_back(val->getType());

		if (not returnType)
			returnType = Type::getVoidTy(info.getContext());

		auto* functionType = FunctionType::get(returnType, argsTypes, false);
		auto externalFun =
				info.getModule().getOrInsertFunction(realName, functionType);

		return info.getBuilder().CreateCall(externalFun, args);
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

		if (auto e = invoke(info, call.getName(), argsValue); !e)
			return e.takeError();

		return outLocation;
	}

	Expected<Value*> lowerCall(LowererContext& info, const ModCall& call)
	{
		auto* alloca = info.allocaModType(call.getType());
		return lowerCall(alloca, info, call);
	}
}	 // namespace marco
