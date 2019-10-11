#pragma once

#include "llvm/IR/IRBuilder.h"
#include "modelica/lowerer/Simulation.hpp"

namespace modelica
{
	/**
	 * \brief create a function with internal linkage in the provided module and
	 * provided name.
	 *
	 * \return the created function, FunctionAlreadyExists if the function already
	 * existed
	 */
	llvm::Expected<llvm::Function*> makePrivateFunction(
			llvm::StringRef name, llvm::Module& m);

	/**
	 * creates a type from a SimType
	 * \return the created type
	 */
	[[nodiscard]] llvm::Type* typeToLLVMType(
			llvm::LLVMContext& context, const SimType& type);

	/**
	 * creates a type from a builtin type
	 * \return the created type.
	 */
	[[nodiscard]] llvm::Type* builtInToLLVMType(
			llvm::LLVMContext& context, BultinSimTypes type);

	/**
	 * allocates the global var into the module
	 * and initializes it with the provided value.
	 * the variable must not have been already inserted
	 *
	 * \return an error if the allocation failed
	 */
	llvm::Error simExpToGlobalVar(
			llvm::Module& module,
			llvm::StringRef name,
			const SimType& type,
			llvm::GlobalValue::LinkageTypes linkage);

	/**
	 * creates a store of an single variable to the provided location.
	 * \return the StoreInt
	 */
	template<typename T>
	llvm::Value* makeConstantStore(
			llvm::IRBuilder<>& builder, T value, llvm::Value* location)
	{
		auto ptrType = llvm::dyn_cast<llvm::PointerType>(location->getType());
		auto underlyingType = ptrType->getContainedType(0);

		if (underlyingType == llvm::Type::getFloatTy(builder.getContext()))

			return builder.CreateStore(
					llvm::ConstantFP::get(underlyingType, value), location);
		return builder.CreateStore(
				llvm::ConstantInt::get(underlyingType, value), location);
	}

	/**
	 * creates a load instruction to load the old value of a particular var.
	 *
	 * \return the loadInst
	 */
	llvm::Expected<llvm::Value*> lowerReference(
			llvm::IRBuilder<>& builder, llvm::StringRef exp);

	/**
	 * Creates a for cycle that last interationsCount iterations
	 * that will be inserted as closing instruction in the entryBlock
	 * The caller has to provide whileContent which is function that will
	 * produce the actual body of the loop.
	 *
	 * \return the exit point of the loop, that is the basic block that will
	 * always be executed at some point.
	 */
	llvm::BasicBlock* createForCycle(
			llvm::Function& function,
			llvm::BasicBlock& entryBlock,
			size_t iterationCount,
			std::function<void(llvm::IRBuilder<>&)> whileContent);

}	// namespace modelica
