#include "marco/Runtime/Print.h"
#include <iostream>
#include <iomanip>

#ifdef MARCO_PROFILING

#include "marco/Runtime/Profiling.h"

namespace
{
  class PrintProfiler : public Profiler
  {
    public:
      PrintProfiler() : Profiler("Simulation data printing")
      {
        registerProfiler(*this);
      }

      void reset() override
      {
        booleanValues.reset();
        integerValues.reset();
        floatValues.reset();
        stringValues.reset();
      }

      void print() const override
      {
        std::cerr << "Time spent on printing boolean values: " << booleanValues.totalElapsedTime() << " ms\n";
        std::cerr << "Time spent on printing integer values: " << integerValues.totalElapsedTime() << " ms\n";
        std::cerr << "Time spent on printing float values: " << floatValues.totalElapsedTime() << " ms\n";
        std::cerr << "Time spent on printing strings: " << stringValues.totalElapsedTime() << " ms\n";
      }

    public:
      Timer booleanValues;
      Timer integerValues;
      Timer floatValues;
      Timer stringValues;
  };

  PrintProfiler& profiler()
  {
    static PrintProfiler obj;
    return obj;
  }
}

  #define PROFILER_BOOL_START ::profiler().booleanValues.start()
  #define PROFILER_BOOL_STOP ::profiler().booleanValues.stop()

  #define PROFILER_INT_START ::profiler().integerValues.start()
  #define PROFILER_INT_STOP ::profiler().integerValues.stop()

  #define PROFILER_FLOAT_START ::profiler().floatValues.start()
  #define PROFILER_FLOAT_STOP ::profiler().floatValues.stop()

  #define PROFILER_STRING_START ::profiler().stringValues.start()
  #define PROFILER_STRING_STOP ::profiler().stringValues.start()

#else
  #define PROFILER_BOOL_START
  #define PROFILER_BOOL_STOP

  #define PROFILER_INT_START
  #define PROFILER_INT_STOP

  #define PROFILER_FLOAT_START
  #define PROFILER_FLOAT_STOP

  #define PROFILER_STRING_START
  #define PROFILER_STRING_STOP
#endif

PrinterConfig& printerConfig()
{
  static PrinterConfig obj;
  return obj;
}

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
    PROFILER_STRING_START;
    std::cout << "\n";
    PROFILER_STRING_STOP;
  }

  void print_csv_separator_void()
  {
    PROFILER_STRING_START;
    std::cout << ";";
    PROFILER_STRING_STOP;
  }

  void print_csv_name_void(void* name, int64_t rank, int64_t* indices)
  {
    PROFILER_STRING_START;
    std::cout << static_cast<char*>(name);
    PROFILER_STRING_STOP;

    if (rank != 0) {
      PROFILER_STRING_START;
      std::cout << "[";
      PROFILER_STRING_STOP;

      for (int64_t i = 0; i < rank; ++i) {
        if (i != 0) {
          PROFILER_STRING_START;
          std::cout << ",";
          PROFILER_STRING_STOP;
        }

        PROFILER_INT_START;
        std::cout << indices[i];
        PROFILER_INT_STOP;
      }

      PROFILER_STRING_START;
      std::cout << "]";
      PROFILER_STRING_STOP;
    }
  }

  void print_csv_void(bool value)
  {
    auto& config = printerConfig();

    if (config.scientificNotation) {
      PROFILER_BOOL_START;
      std::cout << std::scientific;
      PROFILER_BOOL_STOP;
    } else {
      PROFILER_BOOL_START;
      std::cout << std::boolalpha;
      PROFILER_BOOL_STOP;
    }

    PROFILER_BOOL_START;
    std::cout << value;
    PROFILER_BOOL_STOP;
  }

  void print_csv_void(int32_t value)
  {
    auto& config = printerConfig();

    if (config.scientificNotation) {
      PROFILER_INT_START;
      std::cout << std::scientific;
      PROFILER_INT_STOP;
    } else {
      PROFILER_INT_START;
      std::cout << std::fixed << std::setprecision(config.precision);
      PROFILER_INT_STOP;
    }

    PROFILER_INT_START;
    std::cout << value;
    PROFILER_INT_STOP;
  }

  void print_csv_void(int64_t value)
  {
    auto& config = printerConfig();

    if (config.scientificNotation) {
      PROFILER_INT_START;
      std::cout << std::scientific;
      PROFILER_INT_STOP;
    } else {
      PROFILER_INT_START;
      std::cout << std::fixed << std::setprecision(config.precision);
      PROFILER_INT_STOP;
    }

    PROFILER_INT_START;
    std::cout << value;
    PROFILER_INT_STOP;
  }

  void print_csv_void(float value)
  {
    auto& config = printerConfig();

    if (config.scientificNotation) {
      PROFILER_FLOAT_START;
      std::cout << std::scientific;
      PROFILER_FLOAT_STOP;
    } else {
      PROFILER_FLOAT_START;
      std::cout << std::fixed << std::setprecision(config.precision);
      PROFILER_FLOAT_STOP;
    }

    PROFILER_INT_START;
    std::cout << value;
    PROFILER_INT_STOP;
  }

  void print_csv_void(double value)
  {
    auto& config = printerConfig();

    if (config.scientificNotation) {
      PROFILER_FLOAT_START;
      std::cout << std::scientific;
      PROFILER_FLOAT_STOP;
    } else {
      PROFILER_FLOAT_START;
      std::cout << std::fixed << std::setprecision(config.precision);
      PROFILER_FLOAT_STOP;
    }

    PROFILER_INT_START;
    std::cout << value;
    PROFILER_INT_STOP;
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
