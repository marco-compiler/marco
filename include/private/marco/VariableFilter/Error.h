#ifndef MARCO_VARIABLEFILTER_ERROR_H
#define MARCO_VARIABLEFILTER_ERROR_H

#include "marco/VariableFilter/Token.h"
#include "marco/Utils/LogMessage.h"
#include "marco/Utils/SourcePosition.h"
#include "llvm/Support/Error.h"
#include <system_error>

namespace marco::vf::detail
{
  enum class ParsingErrorCode
  {
    success = 0,
    unexpected_token,
    empty_regex,
    invalid_regex
  };
}

namespace std
{
  template<>
  struct is_error_condition_enum<marco::vf::detail::ParsingErrorCode>
    : public std::true_type
  {
  };
}

namespace marco::vf
{
  namespace detail
  {
    class ParsingErrorCategory : public std::error_category
    {
      public:
        static ParsingErrorCategory category;

        std::error_condition default_error_condition(int ev) const noexcept override
        {
          if (ev == 1)
            return std::error_condition(ParsingErrorCode::unexpected_token);

          if (ev == 2)
            return std::error_condition(ParsingErrorCode::empty_regex);

          if (ev == 3)
            return std::error_condition(ParsingErrorCode::invalid_regex);

          return std::error_condition(ParsingErrorCode::success);
        }

        const char* name() const noexcept override
        {
          return "Parsing error";
        }

        bool equivalent(const std::error_code& code, int condition) const noexcept override
        {
          bool equal = *this == code.category();
          auto v = default_error_condition(code.value()).value();
          equal = equal && static_cast<int>(v) == condition;
          return equal;
        }

        std::string message(int ev) const noexcept override
        {
          switch (ev) {
            case (0):
              return "Success";

            case (1):
              return "Unexpected Token";

            case (2):
              return "Empty regex";

            case (3):
              return "Invalid regex";

            default:
              return "Unknown Error";
          }
        }
    };

    std::error_condition make_error_condition(ParsingErrorCode errc);
  }

  class UnexpectedToken
      : public ErrorMessage,
        public llvm::ErrorInfo<UnexpectedToken>
  {
    public:
      static char ID;

      UnexpectedToken(SourceRange location, Token token)
          : location(std::move(location)),
            token(token)
      {
      }

      SourceRange getLocation() const override
      {
        return location;
      }

      void printMessage(llvm::raw_ostream& os) const override
      {
        os << "unexpected token [";
        os.changeColor(llvm::raw_ostream::SAVEDCOLOR, true);
        os << token;
        os << "]";
      }

      void log(llvm::raw_ostream& os) const override
      {
        print(os);
      }

      std::error_code convertToErrorCode() const override
      {
        return std::error_code(
            static_cast<int>(detail::ParsingErrorCode::unexpected_token),
            detail::ParsingErrorCategory::category);
      }

    private:
      SourceRange location;
      Token token;
  };

  class EmptyRegex
      : public ErrorMessage,
        public llvm::ErrorInfo<EmptyRegex>
  {
    public:
      static char ID;

      EmptyRegex(SourceRange location)
          : location(std::move(location))
      {
      }

      SourceRange getLocation() const override
      {
        return location;
      }

      void printMessage(llvm::raw_ostream& os) const override
      {
        os << "empty regex";
      }

      void log(llvm::raw_ostream& os) const override
      {
        print(os);
      }

      std::error_code convertToErrorCode() const override
      {
        return std::error_code(
            static_cast<int>(detail::ParsingErrorCode::empty_regex),
            detail::ParsingErrorCategory::category);
      }

    private:
      SourceRange location;
  };

  class InvalidRegex
      : public ErrorMessage,
        public llvm::ErrorInfo<InvalidRegex>
  {
    public:
      static char ID;

      InvalidRegex(SourceRange location)
          : location(std::move(location))
      {
      }

      SourceRange getLocation() const override
      {
        return location;
      }

      void printMessage(llvm::raw_ostream& os) const override
      {
        os << "invalid regex";
      }

      void log(llvm::raw_ostream& os) const override
      {
        print(os);
      }

      std::error_code convertToErrorCode() const override
      {
        return std::error_code(
            static_cast<int>(detail::ParsingErrorCode::invalid_regex),
            detail::ParsingErrorCategory::category);
      }

    private:
      SourceRange location;
  };
}

#endif // MARCO_VARIABLEFILTER_ERROR_H
