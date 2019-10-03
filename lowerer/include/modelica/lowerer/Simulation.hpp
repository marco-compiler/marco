#pragma once

#include "llvm/ADT/StringMap.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "modelica/lowerer/SimExp.hpp"

namespace modelica
{
	class Simulation
	{
		public:
		Simulation(
				llvm::LLVMContext& context,
				llvm::StringMap<SimExp> vars,
				llvm::StringMap<SimExp> updates,
				std::string name = "Modelica Module",
				unsigned stopTime = 10)
				: context(context),
					module(std::move(name), context),
					variables(std::move(vars)),
					updates(std::move(updates)),
					stopTime(stopTime)
		{
		}

		Simulation(
				llvm::LLVMContext& context,
				std::string name = "Modelica Module",
				unsigned stopTime = 10)
				: context(context), module(std::move(name), context), stopTime(stopTime)
		{
		}

		[[nodiscard]] bool addVar(std::string name, SimExp exp)
		{
			if (variables.find(name) != variables.end())
				return false;
			variables.try_emplace(std::move(name), std::move(exp));
			return true;
		}

		[[nodiscard]] bool addUpdate(std::string name, SimExp exp)
		{
			if (updates.find(name) != updates.end())
				return false;
			updates.try_emplace(std::move(name), std::move(exp));
			return true;
		}

		void lower();
		void dump(llvm::raw_ostream& OS = llvm::outs()) const;
		void dumpBC(llvm::raw_ostream& OS) const;
		[[nodiscard]] unsigned getStopTime() const { return stopTime; }

		private:
		llvm::Function* makePrivateFunction(llvm::StringRef name);
		llvm::LLVMContext& context;
		llvm::Module module;
		llvm::StringMap<SimExp> variables;
		llvm::StringMap<SimExp> updates;
		unsigned stopTime;
	};
}	// namespace modelica
