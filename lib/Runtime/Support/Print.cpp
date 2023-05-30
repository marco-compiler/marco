#include "marco/Runtime/Support/Print.h"
#include "marco/Runtime/Support/Utils.h"
#include <iomanip>

//===----------------------------------------------------------------------===//
// print
//===----------------------------------------------------------------------===//

namespace
{
  template<typename T>
  void print_void(T value)
  {
    std::cout << std::scientific << value << std::endl;
  }

  template<>
  void print_void<bool>(bool value)
  {
    std::cout << std::boolalpha << value << std::endl;
  }
}

RUNTIME_FUNC_DEF(print, void, bool)
RUNTIME_FUNC_DEF(print, void, int32_t)
RUNTIME_FUNC_DEF(print, void, int64_t)
RUNTIME_FUNC_DEF(print, void, float)
RUNTIME_FUNC_DEF(print, void, double)

namespace
{
  template<typename T>
  void print_void(UnrankedMemRefType<T>* array)
  {
    DynamicMemRefType memRef(*array);
    std::cout << std::scientific << memRef << std::endl;
  }

  template<>
  void print_void<bool>(UnrankedMemRefType<bool>* array)
  {
    DynamicMemRefType memRef(*array);
    std::cout << std::boolalpha << memRef << std::endl;
  }
}

RUNTIME_FUNC_DEF(print, void, ARRAY(bool))
RUNTIME_FUNC_DEF(print, void, ARRAY(int32_t))
RUNTIME_FUNC_DEF(print, void, ARRAY(int64_t))
RUNTIME_FUNC_DEF(print, void, ARRAY(float))
RUNTIME_FUNC_DEF(print, void, ARRAY(double))
