/* { dg-do compile } */
/* { dg-options "-funroll-loops -O3" } */

#include <new>

// Based on the method insert from xalanc/include/XalanVector.hpp
// https://apache.github.io/xalan-c/api/XalanVector_8hpp_source.html
// Insert at the end of the vector all elements referenced by the
// iterator theFirst upto the iterator theLast.
void insert(unsigned short *thePointer, const unsigned short *theFirst, const unsigned short *theLast)
{
    while (theFirst != theLast)
    {
        (unsigned short *)new (thePointer) unsigned short(*theFirst);
        ++thePointer;
        ++theFirst;
    }
}

/* { dg-final { scan-assembler "ld" } } */
/* { dg-final { scan-assembler "addi" } } */
/* { dg-final { scan-assembler "addi" } } */
/* { dg-final { scan-assembler "sd" } } */
