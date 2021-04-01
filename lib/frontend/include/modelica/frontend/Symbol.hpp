#pragma once

#include <variant>

namespace modelica
{
	class Class;
	class Function;
	class Induction;
	class Member;
	class Package;
	class Record;

	class Symbol
	{
		public:
		Symbol();
		explicit Symbol(Function& function);
		explicit Symbol(Class& model);
		explicit Symbol(Package& package);
		explicit Symbol(Record& record);
		explicit Symbol(Member& member);
		explicit Symbol(Induction& induction);

		template<typename T>
		[[nodiscard]] bool isA() const
		{
			return std::holds_alternative<T*>(content);
		}

		template<typename T>
		[[nodiscard]] const T& get() const
		{
			assert(isA<T>());
			return *std::get<T*>(content);
		}

		private:
		std::variant<Function*, Class*, Package*, Record*, Member*, Induction*> content;
	};
}
