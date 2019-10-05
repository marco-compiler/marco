#pragma once

#include "llvm/IR/IRBuilder.h"
#include "modelica/lowerer/Simulation.hpp"

namespace modelica
{
	/**
	 * \brief create a function with internal linkage in the provided module and
	 * provided name.
	 *
	 * \return the created function
	 */
	llvm::Function* makePrivateFunction(llvm::StringRef name, llvm::Module& m);

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
			llvm::IRBuilder<>& builder,
			T value,
			llvm::Type* llvmType,
			llvm::Value* location)
	{
		if constexpr (std::is_same<T, int>::value)
			return builder.CreateStore(
					llvm::ConstantInt::get(llvmType, value), location);
		else if constexpr (std::is_same<T, bool>::value)
			return builder.CreateStore(
					llvm::ConstantInt::get(llvmType, value), location);
		else if constexpr (std::is_same<T, float>::value)
			return builder.CreateStore(
					llvm::ConstantFP::get(llvmType, value), location);

		assert(false);	// NOLINT
		return nullptr;
	}

	/**
	 * creates a load instruction to load the old value of a particular var.
	 *
	 * \return the loadInst
	 */
	llvm::Value* lowerReference(llvm::IRBuilder<>& builder, llvm::StringRef exp);

}	// namespace modelica
