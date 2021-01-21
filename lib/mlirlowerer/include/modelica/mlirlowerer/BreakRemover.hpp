#pragma once

#include <modelica/frontend/ClassContainer.hpp>

namespace modelica
{
	class BreakRemover
	{
		public:
		void fix(modelica::ClassContainer& cls);
		void fix(modelica::Class& cls);
		void fix(modelica::Function& function);
		void fix(modelica::Algorithm& algorithm);

		template<typename T>
		bool fix(modelica::Statement& statement);

		private:
		int nestLevel = 0;
	};

	template<>
	bool BreakRemover::fix<modelica::Statement>(modelica::Statement& statement);

	template<>
	bool BreakRemover::fix<modelica::AssignmentStatement>(modelica::Statement& statement);

	template<>
	bool BreakRemover::fix<modelica::IfStatement>(modelica::Statement& statement);

	template<>
	bool BreakRemover::fix<modelica::ForStatement>(modelica::Statement& statement);

	template<>
	bool BreakRemover::fix<modelica::WhileStatement>(modelica::Statement& statement);

	template<>
	bool BreakRemover::fix<modelica::WhenStatement>(modelica::Statement& statement);

	template<>
	bool BreakRemover::fix<modelica::BreakStatement>(modelica::Statement& statement);

	template<>
	bool BreakRemover::fix<modelica::ReturnStatement>(modelica::Statement& statement);
}
