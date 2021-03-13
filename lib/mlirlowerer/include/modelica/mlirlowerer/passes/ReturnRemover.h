#pragma once

#include <modelica/frontend/ClassContainer.hpp>
#include <modelica/frontend/Statement.hpp>

namespace modelica
{
	class ReturnRemover
	{
		public:
		void fix(modelica::ClassContainer& cls);

		private:
		void fix(modelica::Class& cls);
		void fix(modelica::Function& function);
		void fix(modelica::Algorithm& algorithm);

		template<typename T>
		bool fix(modelica::Statement& statement);
	};

	template<>
	bool ReturnRemover::fix<modelica::Statement>(modelica::Statement& statement);

	template<>
	bool ReturnRemover::fix<modelica::AssignmentStatement>(modelica::Statement& statement);

	template<>
	bool ReturnRemover::fix<modelica::IfStatement>(modelica::Statement& statement);

	template<>
	bool ReturnRemover::fix<modelica::ForStatement>(modelica::Statement& statement);

	template<>
	bool ReturnRemover::fix<modelica::WhileStatement>(modelica::Statement& statement);

	template<>
	bool ReturnRemover::fix<modelica::WhenStatement>(modelica::Statement& statement);

	template<>
	bool ReturnRemover::fix<modelica::BreakStatement>(modelica::Statement& statement);

	template<>
	bool ReturnRemover::fix<modelica::ReturnStatement>(modelica::Statement& statement);
}
