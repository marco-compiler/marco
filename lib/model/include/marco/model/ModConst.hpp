#pragma once

#include <type_traits>
#include <utility>
#include <variant>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "marco/model/ModType.hpp"
#include "marco/utils/IRange.hpp"

namespace marco
{
	/**
	 * A ModConst is a way to indicate values known at compile times
	 * They are the basic blocks of expressions.
	 *
	 * There are 3 template specialization that modelica knows how to
	 * lower, and they are one for each foudamental type, int, bool and float.
	 *
	 * Notice that a Const can be an array, but in that case the const is not able
	 * to have multiple dimensions, it's always just a vector and it's up to
	 * the user to determin how the array best fits the type he need.
	 */
	class ModConst
	{
		public:
		template<typename T>
		using Content = llvm::SmallVector<T, 3>;
		/**
		 * Builds a single value constant by providing that value.
		 */
		template<typename T>
		explicit ModConst(T val): content(Content<T>({ val }))
		{
		}

		ModConst(): content(Content<bool>({ false })) {}

		template<typename T>
		ModConst(Content<T> args): content(std::move(args))
		{
		}

		ModConst(int d): content(Content<long>({ static_cast<long>(d) })) {}

		ModConst(float d): content(Content<double>({ static_cast<double>(d) })) {}

		template<typename First, typename... T>
		explicit ModConst(First f, T&&... args)
				: content(Content<First>({ f, std::forward<T>(args)... }))
		{
		}

		template<typename... T>
		explicit ModConst(int f, T&&... args)
				: content(Content<long>({ static_cast<long>(f), static_cast<long>(std::forward<T>(args))... }))
		{
		}

		template<typename Callable>
		void visit(Callable&& c)
		{
			if (isA<long>())
				c(getContent<long>());
			else if (isA<double>())
				c(getContent<double>());
			else if (isA<bool>())
				c(getContent<bool>());
			else
				assert(false && "unreachable");
		}

		template<typename Callable>
		auto map(Callable&& c)
		{
			using ReturnType = decltype(c(Content<long>()));

			ReturnType returnValue;
			visit([&](const auto& content) { returnValue = c(content); });
			return returnValue;
		}

		template<typename Callable>
		[[nodiscard]] auto map(Callable&& c) const
		{
			using ReturnType = decltype(c(Content<long>()));

			ReturnType returnValue;
			visit([&](const auto& content) { returnValue = c(content); });
			return returnValue;
		}

		template<typename Callable>
		void visit(Callable&& c) const
		{
			if (isA<long>())
				c(getContent<long>());
			else if (isA<double>())
				c(getContent<double>());
			else if (isA<bool>())
				c(getContent<bool>());
			else
				assert(false && "unreachable");
		}

		/**
		 * \require index < size()
		 *
		 * \return the element at the indexth position.
		 */
		template<typename T>
		[[nodiscard]] const T& get(size_t index) const
		{
			const auto& cont = std::get<Content<T>>(content);
			assert(index < size());	 // NOLINT
			return cont[index];
		}

		/**
		 * \require index < size()
		 *
		 * \return the element at the indexth position.
		 */
		template<typename T>
		[[nodiscard]] T& get(size_t index)
		{
			auto& cont = std::get<Content<T>>(content);
			assert(index < size());	 // NOLINT
			return cont[index];
		}

		template<typename T>
		[[nodiscard]] Content<T>& getContent()
		{
			assert(isA<T>());
			return std::get<Content<T>>(content);
		}

		template<typename T>
		[[nodiscard]] const Content<T>& getContent() const
		{
			assert(isA<T>());
			return std::get<Content<T>>(content);
		}

		template<typename T>
		[[nodiscard]] ModConst as() const
		{
			const auto copyContent = [](const auto& currContent) {
				Content<T> newContent;
				for (auto e : currContent)
					newContent.emplace_back(e);
				return newContent;
			};

			return map(copyContent);
		}

		[[nodiscard]] ModConst as(BultinModTypes builtin) const
		{
			if (builtin == BultinModTypes::BOOL)
				return as<bool>();
			if (builtin == BultinModTypes::FLOAT)
				return as<double>();
			if (builtin == BultinModTypes::INT)
				return as<long>();

			assert(false && "unreachable");
			return *this;
		}

		template<typename T>
		[[nodiscard]] bool isA() const
		{
			return std::holds_alternative<Content<T>>(content);
		}

		[[nodiscard]] ModType getModTypeOfLiteral() const
		{
			return ModType(getBuiltinType(), size());
		}

		[[nodiscard]] BultinModTypes getBuiltinType() const;

		[[nodiscard]] size_t size() const
		{
			return map([](const auto& content) { return content.size(); });
		}

		void negateAll()
		{
			for (auto index : irange<size_t>(size()))
				negate(index);
		}

		void negate(size_t index)
		{
			visit([index](auto& content) { content[index] = -content[index]; });
		}

		/**
		 * \return true iff every element is equal and have the same size
		 */
		bool operator==(const ModConst& other) const
		{
			return other.content == content;
		}
		/**
		 * \return Negation of operator ==
		 */
		bool operator!=(const ModConst& other) const { return !(*this == other); }

		void dump(llvm::raw_ostream& OS = llvm::outs()) const
		{
			OS << '{';

			visit([&OS](const auto& content) {
				for (auto a : irange(content.size()))
				{
					OS << content[a];
					if (a != content.size() - 1)
						OS << ", ";
				}
			});
			OS << '}';
		}

		static ModConst sum(const ModConst& left, const ModConst& right);
		static ModConst sub(const ModConst& left, const ModConst& right);
		static ModConst mult(const ModConst& left, const ModConst& right);
		static ModConst divide(const ModConst& left, const ModConst& right);
		static ModConst greaterThan(const ModConst& left, const ModConst& right);
		static ModConst greaterEqual(const ModConst& left, const ModConst& right);
		static ModConst equal(const ModConst& left, const ModConst& right);
		static ModConst different(const ModConst& left, const ModConst& right);
		static ModConst lessThan(const ModConst& left, const ModConst& right);
		static ModConst lessEqual(const ModConst& left, const ModConst& right);
		static ModConst elevate(const ModConst& left, const ModConst& right);
		static ModConst module(const ModConst& left, const ModConst& right);

		private:
		std::variant<Content<long>, Content<double>, Content<bool>> content;
	};

	template<>
	[[nodiscard]] inline bool ModConst::isA<int>() const
	{
		return isA<long>();
	}

	/**
	 * This template is used to check if
	 * a class is a instance of a template.
	 *
	 * This is usefull to determin the kind of
	 * constant you are receiving.
	 */
	template<class, template<class> class>
	struct is_instance: public std::false_type	// NOLINT
	{
	};

	template<class T, template<class> class U>
	struct is_instance<U<T>, U>: public std::true_type
	{
	};

}	 // namespace marco
