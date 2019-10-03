#pragma once

#include "llvm/ADT/StringMap.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "modelica/lowerer/SimExp.hpp"

namespace modelica
{
	constexpr int defaultSimulationIterations = 10;
	class Simulation
	{
		public:
		Simulation(
				llvm::LLVMContext& context,
				llvm::StringMap<SimExp> vars,
				llvm::StringMap<SimExp> updates,
				std::string name = "Modelica Module",
				std::string entryPointName = "main",
				unsigned stopTime = defaultSimulationIterations)
				: context(context),
					module(std::move(name), context),
					variables(std::move(vars)),
					updates(std::move(updates)),
					stopTime(stopTime),
					entryPointName(std::move(entryPointName)),
					varsLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage)
		{
		}

		Simulation(
				llvm::LLVMContext& context,
				std::string name = "Modelica Module",
				std::string entryPointName = "main",
				unsigned stopTime = defaultSimulationIterations)
				: context(context),
					module(std::move(name), context),
					stopTime(stopTime),
					entryPointName(std::move(entryPointName)),
					varsLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage)
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
		[[nodiscard]] llvm::GlobalValue::LinkageTypes getVarLinkage() const
		{
			return varsLinkage;
		}

		void setVarsLinkage(llvm::GlobalValue::LinkageTypes newLinkage)
		{
			varsLinkage = newLinkage;
		}

		private:
		llvm::Function* makePrivateFunction(llvm::StringRef name);
		llvm::LLVMContext& context;
		llvm::Module module;
		llvm::StringMap<SimExp> variables;
		llvm::StringMap<SimExp> updates;
		unsigned stopTime;

		std::string entryPointName;
		llvm::GlobalValue::LinkageTypes varsLinkage;
	};
}	// namespace modelica
