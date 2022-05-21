#include "marco/VariableFilter/Error.h"

using namespace ::marco;
using namespace ::marco::vf;

namespace marco::vf
{
  char UnexpectedToken::ID;
  char EmptyRegex::ID;
  char InvalidRegex::ID;

  namespace detail
  {
    ParsingErrorCategory ParsingErrorCategory::category;

    std::error_condition make_error_condition(ParsingErrorCode errc)
    {
      return std::error_condition(
          static_cast<int>(errc), detail::ParsingErrorCategory::category);
    }
  }
}
