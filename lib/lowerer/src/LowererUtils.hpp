#pragma once

#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Error.h"
#include "modelica/lowerer/Lowerer.hpp"
#include "modelica/model/Assigment.hpp"
#include "modelica/utils/Interval.hpp"

namespace modelica
{
	class LowererContext
	{
		private:
		llvm::IRBuilder<>& builder;
		llvm::Module& module;
		llvm::Function* function;
		llvm::Value* inductionsVars;

		llvm::BasicBlock* createdNestedForCycleImp(
				const OrderedMultiDimInterval& iterationsCountBegin,
				std::function<void(llvm::Value*)> whileContent,
				llvm::SmallVector<llvm::Value*, 3>& indexes);

		llvm::Value* valueArrayFromArrayOfValues(
				llvm::SmallVector<llvm::Value*, 3> vals);

		public:
		LowererContext(llvm::IRBuilder<>& builder, llvm::Module& module)
				: builder(builder),
					module(module),
					function(nullptr),
					inductionsVars(nullptr)
		{
		}
		[[nodiscard]] llvm::IRBuilder<>& getBuilder() { return builder; }
		[[nodiscard]] llvm::Module& getModule() { return module; }
		[[nodiscard]] llvm::Function* getFunction() { return function; }
		[[nodiscard]] llvm::Value* getInductionVars() { return inductionsVars; }
		[[nodiscard]] llvm::LLVMContext& getContext()
		{
			return builder.getContext();
		}

		void setFunction(llvm::Function* fun) { function = fun; }
		void setInductions(llvm::Value* inds) { inductionsVars = inds; }
		/**
		 * \return the pointer to the index element inside the array pointed by
		 * arrayPtr
		 */
		llvm::Value* getArrayElementPtr(llvm::Value* arrayPtr, llvm::Value* index);

		/**
		 * \return the pointer to the index element inside the array pointed by
		 * arrayPtr
		 * \pre index in bounds
		 */
		llvm::Value* getArrayElementPtr(llvm::Value* arrayPtr, size_t index);

		/**
		 * \return a alloca instr that contains a zero terminated array of all
		 * dimensions of the type
		 */
		llvm::AllocaInst* getTypeDimensionsArray(const ModType& type);

		/**
		 * arrayPtr[index] = value;
		 */
		void storeToArrayElement(
				llvm::Value* value, llvm::Value* arrayPtr, llvm::Value* index);

		/**
		 * arrayPtr[index] = value;
		 *
		 */
		void storeToArrayElement(
				llvm::Value* value, llvm::Value* arrayPtr, size_t index);

		/**
		 * \return arrayPtr[index]
		 */
		llvm::Value* loadArrayElement(llvm::Value* arrayPtr, size_t index);

		/**
		 * \return arrayPtr[index]
		 */
		llvm::Value* loadArrayElement(llvm::Value* arrayPtr, llvm::Value* index);

		/**
		 * creates a store of an single variable to the provided location.
		 * \return the StoreInt
		 */
		template<typename T>
		llvm::Value* makeConstantStore(T value, llvm::Value* location)
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
		 * arrayPtr[index] = value
		 */
		template<typename T>
		void storeConstantToArrayElement(
				T value, llvm::Value* arrayPtr, size_t index)
		{
			auto ptrToElem = getArrayElementPtr(arrayPtr, index);
			makeConstantStore<T>(value, ptrToElem);
		}

		/**
		 * creates a load instruction to load the old value of a particular var.
		 *
		 * \return the loadInst
		 */
		llvm::Expected<llvm::Value*> lowerReference(llvm::StringRef exp);

		/**
		 * \return a llvm::type rappresenting the array of types of the provided
		 * ModType.
		 */
		llvm::AllocaInst* allocaModType(const ModType& type);

		/**
		 * Creates a for cycle that last interationsCount iterations
		 * that will be inserted in the provided builder
		 * The caller has to provide whileContent which is function that will
		 * produce the actual body of the loop.
		 *
		 * if inverse range is true it will create a range going from max to min
		 * rather than min to max
		 *
		 * \return the exit point of the loop, that is the basic block that will
		 * always be executed at some point. The builder will look at that basic
		 * block.
		 */
		llvm::BasicBlock* createForCycle(
				Interval induction,
				std::function<void(llvm::Value*)> whileContent,
				bool inverseRange);

		llvm::BasicBlock* createdNestedForCycle(
				const OrderedMultiDimInterval& induction,
				std::function<void(llvm::Value*)> whileContent);

		llvm::BasicBlock* createForArrayElement(
				const ModType& type, std::function<void(llvm::Value*)> whileContent);

		llvm::BasicBlock* createdNestedForCycle(
				llvm::ArrayRef<size_t> iterationsCountEnd,
				std::function<void(llvm::Value*)> body);

		using TernaryOpFunction = std::function<llvm::Expected<llvm::Value*>()>;

		/**
		 * Creates a if else branch based on the result value of condition()
		 * \pre the returned llvm::type of trueBlock() must be equal to the
		 * returned llvm::type of falseBlock() and to outType, the returned
		 * llvm::type of condition() bust be int1. \return the phi instruction
		 * that contains the result of the brach taken.
		 *
		 * builder will now point at the exit BB.
		 */
		llvm::Expected<llvm::Value*> createTernaryOp(
				llvm::Type* outType,
				TernaryOpFunction condition,
				TernaryOpFunction trueBlock,
				TernaryOpFunction falseBlock);
	};
	/**
	 * creates a type from a ModType
	 * \return the created type
	 */
	[[nodiscard]] llvm::ArrayType* typeToLLVMType(
			llvm::LLVMContext& context, const ModType& type);

	/**
	 * creates a type from a builtin type
	 * \return the created type.
	 */
	[[nodiscard]] llvm::Type* builtInToLLVMType(
			llvm::LLVMContext& context, BultinModTypes type);

	[[nodiscard]] BultinModTypes builtinTypeFromLLVMType(llvm::Type* tp);
	[[nodiscard]] ModType modTypeFromLLVMType(llvm::ArrayType* type);
}	 // namespace modelica
