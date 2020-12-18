#pragma once

#include <variant>

namespace modelica
{
	class Class;
	class Function;
	class Induction;
	class Member;

	class Symbol
	{
		public:
		Symbol();
		explicit Symbol(Function& function);
		explicit Symbol(Class& model);
		explicit Symbol(Member& mem);
		explicit Symbol(Induction& mem);

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
		std::variant<Function*, Class*, Member*, Induction*> content;
	};
}
