#ifdef WINDOWS_NOSTDLIB
#include <Windows.h>
#include "marco/runtime/Printing.h"

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
	}

	void __throw_bad_cast() {
	}

	void __throw_length_error(char const*) {
	}

	void __throw_bad_alloc() {
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
	ryuPrintf("%d\n", len);
    unsigned char* p= (unsigned char*) s;
    while(len--)
    {
        *p++ = (unsigned char)c;
    }
    return s;
}
#endif