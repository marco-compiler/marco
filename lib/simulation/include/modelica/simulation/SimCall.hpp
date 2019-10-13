#pragma once

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/simulation/SimType.hpp"

namespace modelica
{
	class SimExp;

	/**
	 *
	 * A SimCall is a rappresentation of a call of a external function,
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
	class SimCall
	{
		public:
		using ArgsVec = llvm::SmallVector<std::unique_ptr<SimExp>, 3>;

		/**
		 * Buils a function call with the provided name, type and args
		 */
		SimCall(std::string name, ArgsVec args, SimType type)
				: name(std::move(name)), args(std::move(args)), type(std::move(type))
		{
		}

		/**
		 * Overload to allow inizializers lists
		 */
		SimCall(
				std::string name,
				std::initializer_list<SimExp> arguments,
				SimType type);

		SimCall(const SimCall& other);

		SimCall(SimCall&& other) = default;

		SimCall& operator=(const SimCall& other);

		/**
		 * \return true if the two calls are deeply equal.
		 */
		[[nodiscard]] bool operator==(const SimCall& other) const;
		[[nodiscard]] bool operator!=(const SimCall& other) const
		{
			return !(*this == other);
		}

		SimCall& operator=(SimCall&& other) = default;
		~SimCall() = default;

		[[nodiscard]] const std::string& getName() const { return name; }

		[[nodiscard]] const ArgsVec& getArgs() const { return args; }

		[[nodiscard]] size_t argsSize() const { return args.size(); }

		[[nodiscard]] const SimExp& at(size_t index) const { return *args[index]; }

		[[nodiscard]] SimExp& at(size_t index) { return *args[index]; }

		void dump(llvm::raw_ostream& OS = llvm::outs()) const;

		[[nodiscard]] const SimType& getType() const { return type; }

		private:
		std::string name;
		ArgsVec args;
		SimType type;
	};

	template<typename SimCall, typename Visitor>
	void visitCall(SimCall& call, Visitor& visitor)
	{
		for (size_t a = 0; a < call.argsSize(); a++)
			visit(call.at(a), visitor);
	}
}	 // namespace modelica
