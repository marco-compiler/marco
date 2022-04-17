#include "marco/Runtime/IO.h"
#include <iostream>
#include <iomanip>

#ifdef MARCO_PROFILING

#include "marco/Runtime/Profiling.h"

namespace
{
  class InputOutputProfiler : public Profiler
  {
    public:
      InputOutputProfiler() : Profiler("Input / output")
      {
        registerProfiler(*this);
      }

      void reset() override
      {
        timer.reset();
      }

      void print() const override
      {
        std::cerr << "Time spent for input / output operations: " << time() << " ms\n";
      }

      void startTimer()
      {
        timer.start();
      }

      void stopTimer()
      {
        timer.stop();
      }

      double time() const
      {
        return timer.totalElapsedTime();
      }

    private:
      Timer timer;
  };

  InputOutputProfiler& profiler()
  {
    static InputOutputProfiler obj;
    return obj;
  }
}

#endif

//===----------------------------------------------------------------------===//
// print
//===----------------------------------------------------------------------===//

namespace
{
  template<typename T>
  void print_void(T value)
  {
    #ifdef MARCO_PROFILING
    ::profiler().startTimer();
    #endif

    std::cout << std::scientific << value << std::endl;

    #ifdef MARCO_PROFILING
    ::profiler().stopTimer();
    #endif
  }

  template<>
  void print_void<bool>(bool value)
  {
    #ifdef MARCO_PROFILING
    ::profiler().startTimer();
    #endif

    std::cout << std::boolalpha << value << std::endl;

    #ifdef MARCO_PROFILING
    ::profiler().stopTimer();
    #endif
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
    #ifdef MARCO_PROFILING
    ::profiler().startTimer();
    #endif

    std::cout << std::scientific << array << std::endl;

    #ifdef MARCO_PROFILING
    ::profiler().stopTimer();
    #endif
  }

  template<>
  void print_void<bool>(UnsizedArrayDescriptor<bool> array)
  {
    #ifdef MARCO_PROFILING
    ::profiler().startTimer();
    #endif

    std::cout << std::boolalpha << array << std::endl;

    #ifdef MARCO_PROFILING
    ::profiler().stopTimer();
    #endif
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
    #ifdef MARCO_PROFILING
    ::profiler().startTimer();
    #endif

    std::cout << "\n";

    #ifdef MARCO_PROFILING
    ::profiler().stopTimer();
    #endif
  }

  void print_csv_separator_void()
  {
    #ifdef MARCO_PROFILING
    ::profiler().startTimer();
    #endif

    std::cout << ";";

    #ifdef MARCO_PROFILING
    ::profiler().stopTimer();
    #endif
  }

  void print_csv_name_void(void* name, int64_t rank, int64_t* indices)
  {
    #ifdef MARCO_PROFILING
    ::profiler().startTimer();
    #endif

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

    #ifdef MARCO_PROFILING
    ::profiler().stopTimer();
    #endif
  }

  template<typename T>
  void print_csv_void(T value)
  {
    #ifdef MARCO_PROFILING
    ::profiler().startTimer();
    #endif

    std::cout << std::fixed << std::setprecision(9) << value;

    #ifdef MARCO_PROFILING
    ::profiler().stopTimer();
    #endif
  }

  template<>
  void print_csv_void<bool>(bool value)
  {
    #ifdef MARCO_PROFILING
    ::profiler().startTimer();
    #endif

    std::cout << std::boolalpha << value;

    #ifdef MARCO_PROFILING
    ::profiler().stopTimer();
    #endif
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
