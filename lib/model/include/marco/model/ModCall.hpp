#pragma once

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "marco/model/ModType.hpp"

namespace marco
{
	class ModExp;

	/**
	 *
	 * A ModCall is a rappresentation of a call of a external function,
	 * the external function will accept the following parameters:
	 * -> a ptr to the base tipe of the return type of this call
	 * -> a ptr to a zero terminated const array of size_t that contains the
	 * dimensions of the return type.
	 *
	 * for each parameter
	 * -> a ptr to the base type of the evalutaed expression
	 * -> a ptr to a const array of size_t to the dimensions of the type passed as
	 * argument
	 */
	class ModCall
	{
		public:
		using ArgsVec = llvm::SmallVector<std::unique_ptr<ModExp>, 3>;

		/**
		 * Buils a function call with the provided name, type and args
		 */
		ModCall(std::string name, ArgsVec args, ModType type)
				: name(std::move(name)), args(std::move(args)), type(std::move(type))
		{
		}

		/**
		 * Buils a function call with the provided name, type and args
		 */
		ModCall(std::string name, ArgsVec args, BultinModTypes type)
				: name(std::move(name)), args(std::move(args)), type(ModType(type))
		{
		}

		/**
		 * Overload to allow inizializers lists
		 */
		ModCall(
				std::string name,
				std::initializer_list<ModExp> arguments,
				ModType type);

		/**
		 * Overload to allow inizializers lists
		 */
		ModCall(
				std::string name,
				std::initializer_list<ModExp> arguments,
				BultinModTypes type);

		ModCall(const ModCall& other);

		ModCall(ModCall&& other) = default;

		ModCall& operator=(const ModCall& other);

		/**
		 * \return true if the two calls are deeply equal.
		 */
		[[nodiscard]] bool operator==(const ModCall& other) const;
		[[nodiscard]] bool operator!=(const ModCall& other) const
		{
			return !(*this == other);
		}

		ModCall& operator=(ModCall&& other) = default;
		~ModCall() = default;

		[[nodiscard]] const std::string& getName() const { return name; }

		[[nodiscard]] const ArgsVec& getArgs() const { return args; }

		[[nodiscard]] size_t argsSize() const { return args.size(); }

		[[nodiscard]] const ModExp& at(size_t index) const { return *args[index]; }

		[[nodiscard]] ModExp& at(size_t index) { return *args[index]; }

		void dump(llvm::raw_ostream& OS = llvm::outs()) const;

		void readableDump(llvm::raw_ostream& OS = llvm::outs()) const;
		void readableDump() const;

		[[nodiscard]] const ModType& getType() const { return type; }

		private:
		std::string name;
		ArgsVec args;
		ModType type;
	};

	template<typename ModCall, typename Callable>
	void forEachArgument(ModCall& call, Callable&& callable)
	{
		for (size_t a = 0; a < call.argsSize(); a++)
			callable(call.at(a));
	}
}	 // namespace marco