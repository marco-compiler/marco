#ifdef WINDOWS_NOSTDLIB
#include <Windows.h>
#include "marco/runtime/Nostdlib.h"
#include "marco/runtime/UtilityFunctions.h"

BOOL WINAPI DllMain(
    HINSTANCE hinstDLL,
    DWORD fdwReason,
    LPVOID lpReserved)
{
    return TRUE;
}

extern "C" BOOL WINAPI DllMainCRTStartup(
	HINSTANCE hinstDLL,
	DWORD fdwReason,
	LPVOID lpvReserved)
{
	return TRUE;
}

namespace std {
	void __throw_bad_array_new_length() {
		ExitProcess(1);
	}

	void __throw_bad_cast() {
		ExitProcess(1);
	}

	void __throw_length_error(char const*) {
		ExitProcess(1);
	}

	void __throw_bad_alloc() {
		ExitProcess(1);
	}
}

void* operator new(std::size_t sz)
{
    if (sz == 0)
        ++sz;
 
    if (void *ptr = HeapAlloc(GetProcessHeap(), 0x0, sz))
        return ptr;
	else 
		return NULL;
    //throw std::bad_alloc{};
}
void operator delete(void* ptr) noexcept
{
    HeapFree(GetProcessHeap(), 0x0, ptr);
}

void operator delete(void* ptr, std::size_t sz)
{
	::operator delete(ptr);
}

void* memmove(void* dstpp, const void* srcpp, size_t len)
{
	char* dstp = (char*)dstpp;
	const char* srcp = (const char*)srcpp;

	if(dstp < srcp) {
		for (size_t i = 0; i < len; i++)
			*(dstp + i) = *(srcp + i);
	} else {
		for (size_t i = 0; i < len; i++)
			*(dstp + len - 1 - i) = *(srcp + len - 1 - i);
	}
	return dstpp;
}

void* memcpy(void* dstpp, const void* srcpp, size_t len)
{
	char* dstp = (char*)dstpp;
	const char* srcp = (const char*)srcpp;

	for (size_t i = 0; i < len; i++)
		*(dstp + i) = *(srcp + i);

	return dstpp;
}

void* memset(void* s, int c,  size_t len)
{
	size_t i = 0;
    volatile unsigned char* p = (unsigned char*) s;
	while(i < len)
	{
		*p = c;
		p = p + 1;
		i = i + 1;
	}
    return s;
}

void runtimeMemset(char *p, char c, int l)
{
      for(int i = 0; i < l; i++)
        *(p + i) = '0';
}
#endif