#pragma once

#include "llvm/ADT/StringMap.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "modelica/lowerer/SimExp.hpp"

namespace modelica
{
	/**
	 * The default number of iterations to be performed.
	 * Totaly arbitrary number.
	 */
	constexpr int defaultSimulationIterations = 10;

	/**
	 * A simulation is the main container of the library. A simulation can be
	 * lowered and dumped into a bc file to be later compiled.
	 *
	 * A simulation is made of a inizialization section and of an update section.
	 * The generated file will invoke the initialization values once and then
	 * update a certain number of time and will print the values of the vars at
	 * each update.
	 */
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

		/**
		 * adds a var to the simulation that will be intialized with the provided
		 * expression. Notice that in the initialization is undefined behaviour to
		 * use references to other variables.
		 *
		 * \return true if there were no other vars with the same name already.
		 */
		[[nodiscard]] bool addVar(std::string name, SimExp exp)
		{
			if (variables.find(name) != variables.end())
				return false;
			variables.try_emplace(std::move(name), std::move(exp));
			return true;
		}

		/**
		 * Add an update expression for a particular variable.
		 * notice that if a expression is referring to a missing
		 * variable then it's lower that will fail, not addUpdate
		 *
		 * \return  true if there were no other updates referring to the same var.
		 */
		[[nodiscard]] bool addUpdate(std::string name, SimExp exp)
		{
			if (updates.find(name) != updates.end())
				return false;
			updates.try_emplace(std::move(name), std::move(exp));
			return true;
		}

		/**
		 * \requires lower was not already invoked on this object.
		 * Populate the module with the simulation.
		 *
		 * \return a error if there were missing references or if a type missmatch
		 * was encountered.
		 */
		void lower();

		/**
		 * dumpds a human readable rappresentation of the simulation to OS, standard
		 * out by default
		 */
		void dump(llvm::raw_ostream& OS = llvm::outs()) const;

		/**
		 * \requires lower has been invoked and it returned success.
		 * dumps the simulation as bytecode
		 */
		void dumpBC(llvm::raw_ostream& OS) const;

		/**
		 * \return the number of updates of the simulation
		 */
		[[nodiscard]] unsigned getStopTime() const { return stopTime; }

		/**
		 * \return the kind of linkage specified to the variables.
		 *
		 * This is usefull if you need to expose variables that are by default
		 * internal so that other programs can link to the simulation and drive it.
		 */
		[[nodiscard]] llvm::GlobalValue::LinkageTypes getVarLinkage() const
		{
			return varsLinkage;
		}

		/**
		 *
		 * Set the linkage type of the variables
		 * This is usefull if you need to expose variables that are by default
		 * internal so that other programs can link to the simulation and drive it.
		 */
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
