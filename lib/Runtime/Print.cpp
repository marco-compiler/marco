#include "marco/Runtime/Print.h"
#include <iostream>
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
  void print_void(UnsizedArrayDescriptor<T> array)
  {
    std::cout << std::scientific << array << std::endl;
  }

  template<>
  void print_void<bool>(UnsizedArrayDescriptor<bool> array)
  {
    std::cout << std::boolalpha << array << std::endl;
  }
}

RUNTIME_FUNC_DEF(print, void, ARRAY(bool))
RUNTIME_FUNC_DEF(print, void, ARRAY(int32_t))
RUNTIME_FUNC_DEF(print, void, ARRAY(int64_t))
RUNTIME_FUNC_DEF(print, void, ARRAY(float))
RUNTIME_FUNC_DEF(print, void, ARRAY(double))

//===----------------------------------------------------------------------===//
// print_csv
//===----------------------------------------------------------------------===//

namespace
{
  void print_csv_newline_void()
  {
    std::cout << "\n";
  }

  void print_csv_separator_void()
  {
    std::cout << ";";
  }

  void print_csv_name_void(void* name, int64_t rank, int64_t* indices)
  {
    std::cout << static_cast<char*>(name);

    if (rank != 0) {
      std::cout << "[";

      for (int64_t i = 0; i < rank; ++i) {
        if (i != 0) {
          std::cout << ",";
        }

        std::cout << indices[i];
      }

      std::cout << "]";
    }
  }

  template<typename T>
  void print_csv_void(T value)
  {
    std::cout << std::fixed << std::setprecision(9) << value;
  }

  template<>
  void print_csv_void<bool>(bool value)
  {
    std::cout << std::boolalpha << value;
  }
}

RUNTIME_FUNC_DEF(print_csv_newline, void)
RUNTIME_FUNC_DEF(print_csv_separator, void)
RUNTIME_FUNC_DEF(print_csv_name, void, PTR(void), int64_t, PTR(int64_t))

RUNTIME_FUNC_DEF(print_csv, void, bool)
RUNTIME_FUNC_DEF(print_csv, void, int32_t)
RUNTIME_FUNC_DEF(print_csv, void, int64_t)
RUNTIME_FUNC_DEF(print_csv, void, float)
RUNTIME_FUNC_DEF(print_csv, void, double)
