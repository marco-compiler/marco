#pragma once

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace modelica
{
	class SimExp;
	class SimCall
	{
		public:
		using ArgsVec = llvm::SmallVector<std::unique_ptr<SimExp>, 3>;
		SimCall(std::string name, ArgsVec args)
				: name(std::move(name)), args(std::move(args))
		{
		}

		SimCall(std::string name, std::initializer_list<SimExp> arguments);

		SimCall(const SimCall& other);

		SimCall(SimCall&& other) = default;

		SimCall& operator=(const SimCall& other);

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

		private:
		std::string name;
		ArgsVec args;
	};

	template<typename SimCall, typename Visitor>
	void visitCall(SimCall& call, Visitor& visitor)
	{
		for (size_t a = 0; a < call.argsSize(); a++)
			visit(call.at(a), visitor);
	}
}	 // namespace modelica
