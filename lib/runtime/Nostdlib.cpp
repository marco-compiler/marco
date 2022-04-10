#ifdef WINDOWS_NOSTDLIB
#include "marco/runtime/Printing.h"

#include <cstddef> // to define std::size_t
#include <Windows.h>

///////////////////////////////////////////////////////////////////////////////////
// The following code is needed to substitute the c library outside the runtime
// library, by the executable compiled by marco.
///////////////////////////////////////////////////////////////////////////////////

extern "C" int printf(const char* format, ...)
{
	va_list arg;
	int done;

	va_start(arg, format);
	done = runtimePrintfInternal(format, arg);
	va_end(arg);

	return done;
}

extern "C" int putchar(int c)
{
	printChar(c);
	return 1;
}

// This function is called before the main function in a nostdlib executable
// compiled with gcc.
extern "C" int __main()
{
	return 0;
}

// The following code is needed when building with MSVC.
#ifdef MSVC_BUILD
extern "C" int main();

// This is the first function called in the Windows executable.
extern "C" void mainCRTStartup()
{
	main();
	return;
}

extern "C" __declspec(noreturn) void __cdecl 
__imp__invalid_parameter_noinfo_noreturn(void)
{
	ExitProcess(1);
}

extern "C" __declspec(noreturn) void __cdecl
__imp__invoke_watson(
    wchar_t const* const expression,
    wchar_t const* const function_name,
    wchar_t const* const file_name,
    unsigned int const line_number,
    uintptr_t const reserved)
{
	ExitProcess(1);
}

extern "C" int _fltused = 0;
#endif // MSVC_BUILD

///////////////////////////////////////////////////////////////////////////////////
// The following code is needed to substitute the c runtime inside the runtime
// library.
///////////////////////////////////////////////////////////////////////////////////

// Error throwing functions needed for vectors.
namespace std {
	void __throw_bad_array_new_length() {
		ExitProcess(1);
	}

	void __throw_length_error(char const*) {
		ExitProcess(1);
	}

	void __throw_bad_alloc() {
		ExitProcess(1);
	}
}

// Memory allocation functions needed for vectors.
void* operator new(std::size_t sz)
{
    if (sz == 0)
        ++sz;
 
    if (void *ptr = HeapAlloc(GetProcessHeap(), 0x0, sz))
        return ptr;
	else 
		return NULL;
}

void operator delete(void* ptr) noexcept
{
    HeapFree(GetProcessHeap(), 0x0, ptr);
}

void operator delete(void* ptr, std::size_t sz)
{
	::operator delete(ptr);
}

// Needed by the built in functions.
void* memmove(void* dstpp, const void* srcpp, size_t len)
{
	char* dstp = (char*)dstpp;
	const char* srcp = (const char*)srcpp;

	// The two cases are needed to allow for the dst and src to be overlapping:
	// copy left to right if dstp < srcp, otherwise right to left.
	if(dstp < srcp) {
		for (size_t i = 0; i < len; i++)
			*(dstp + i) = *(srcp + i);
	} else {
		for (size_t i = 0; i < len; i++)
			*(dstp + len - 1 - i) = *(srcp + len - 1 - i);
	}
	return dstpp;
}

// Needed by the fixed to double conversion.
void* memcpy(void* dstpp, const void* srcpp, size_t len)
{
	char* dstp = (char*)dstpp;
	const char* srcp = (const char*)srcpp;

	for (size_t i = 0; i < len; i++)
		*(dstp + i) = *(srcp + i);

	return dstpp;
}

// Needed by the fixed to double conversion.
void* memset(void* s, int c,  size_t len)
{
	size_t i = 0;
	// The volatile keyword is needed because optimizations here make the
	// application crash.
    volatile unsigned char* p = (unsigned char*) s;
	while(i < len)
	{
		*p = c;
		p = p + 1;
		i = i + 1;
	}
    return s;
}

#endif // WINDOWS_NOSTDLIB